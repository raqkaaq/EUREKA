"""Private-network FreeToken provider: LAN-only model configuration,
PydanticAI model construction, and deterministic client lifetime.

Owns no prompts, no analysis bounds (see :mod:`radar.config.runtime`),
no inference HTTP (PydanticAI owns that); this module owns the
underlying client lifetime.
"""

from __future__ import annotations

import asyncio as _asyncio
import ipaddress as _ipaddress
import os as _os
import socket as _socket
import urllib.parse as _parse
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx as _httpx

if TYPE_CHECKING:
    import httpx2 as _httpx2
    from pydantic_ai.models import Model as _Model

EXAMPLE_BASE_URL = "http://192.168.1.20:1919/v1"
MODELS_TIMEOUT_S = 5.0
MODEL_TIMEOUT_S = 60.0
DISABLE_THINKING_ENV_VAR = "FREETOKEN_DISABLE_THINKING"


def resolve_disable_thinking(explicit: bool = False) -> bool:
    """Whether to send the server-specific thinking-disable key.

    Opt-in only: explicit flag wins, else the ``FREETOKEN_DISABLE_THINKING``
    env var (1/true/yes/on). Default omits the key entirely -- no backend
    is assumed to support it.
    """
    if explicit:
        return True
    return _os.environ.get(DISABLE_THINKING_ENV_VAR, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def thinking_extra_body() -> dict[str, object]:
    """Server-specific key disabling vLLM-style reasoning output (opt-in)."""
    return {"chat_template_kwargs": {"enable_thinking": False}}

_LOCALHOST_NAMES = frozenset({"localhost"})
_ALLOWED_V4_NETWORKS = tuple(
    _ipaddress.ip_network(value)
    for value in (
        "10.0.0.0/8",
        "100.64.0.0/10",  # RFC 6598, commonly used by private overlay LANs.
        "127.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
    )
)
_ALLOWED_V6_NETWORKS = tuple(
    _ipaddress.ip_network(value) for value in ("::1/128", "fc00::/7")
)


class FreeTokenError(RuntimeError):
    """Actionable local-model failure (endpoint down, no model, bad config)."""


def _is_allowed_address(value: str) -> bool:
    """Return whether an IP literal belongs to an allowed local network."""
    try:
        address = _ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.is_link_local or address.is_multicast or address.is_unspecified:
        return False
    if isinstance(address, _ipaddress.IPv4Address):
        return any(address in network for network in _ALLOWED_V4_NETWORKS)
    if address.ipv4_mapped is not None:
        return _is_allowed_address(str(address.ipv4_mapped))
    return any(address in network for network in _ALLOWED_V6_NETWORKS)


def _resolved_addresses(host: str) -> set[str]:
    try:
        answers = _socket.getaddrinfo(host, None, type=_socket.SOCK_STREAM)
    except OSError as exc:
        raise FreeTokenError(
            f"Cannot resolve FreeToken host {host!r}; use a LAN IP address or "
            "a hostname that resolves on this machine."
        ) from exc
    return {str(answer[4][0]).split("%", 1)[0] for answer in answers if answer[4]}


def check_local_network(base_url: str) -> str:
    """Validate that *base_url* targets loopback or a private LAN.

    Returns the normalized URL. Raises :class:`FreeTokenError` for empty,
    unparseable, non-http(s), or non-local destinations. Hostnames are resolved
    and accepted only when every returned address is in an allowed network.
    """
    raw = (base_url or "").strip()
    if not raw:
        raise FreeTokenError(
            "FreeToken base URL is not configured; set FREETOKEN_BASE_URL or "
            f"pass --base-url with a loopback/private-LAN URL (for example {EXAMPLE_BASE_URL})."
        )
    try:
        parts = _parse.urlsplit(raw)
    except ValueError as exc:
        raise FreeTokenError(f"FreeToken base URL is not a valid URL ({raw!r}): {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise FreeTokenError(
            f"FreeToken base URL must use http(s) (got {raw!r}); "
            f"for example {EXAMPLE_BASE_URL}."
        )
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host or parts.username is not None or parts.password is not None:
        raise FreeTokenError(
            f"FreeToken base URL must contain a host and no embedded credentials ({raw!r})."
        )
    try:
        _ = parts.port
    except ValueError as exc:
        raise FreeTokenError(f"FreeToken base URL has an invalid port ({raw!r}).") from exc
    if host in _LOCALHOST_NAMES:
        addresses = {"127.0.0.1"}
    else:
        try:
            _ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            addresses = _resolved_addresses(host)
        else:
            addresses = {host.split("%", 1)[0]}
    if not addresses or any(not _is_allowed_address(address) for address in addresses):
        raise FreeTokenError(
            f"Refusing non-local FreeToken base URL {raw!r}: the host must resolve "
            "only to loopback, RFC 1918, IPv6 ULA, or RFC 6598 addresses."
        )
    return raw.rstrip("/")


def resolve_base_url(explicit: str | None = None) -> str:
    """Resolve and local-network-check the FreeToken base URL.

    Precedence: explicit arg > ``FREETOKEN_BASE_URL`` env > default.
    """
    env = _os.environ.get("FREETOKEN_BASE_URL", "").strip()
    return check_local_network(explicit or env)


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def list_models(base_url: str, timeout: float = MODELS_TIMEOUT_S) -> list[str]:
    """List model ids from the local ``/models`` endpoint (httpx, bounded).

    Never includes credentials, headers, or raw bodies in errors.
    """
    base_url = check_local_network(base_url)
    if not (0 < timeout <= 30):
        raise FreeTokenError("models lookup timeout must be within (0, 30]s")
    try:
        with _httpx.Client(trust_env=False, follow_redirects=False) as client:
            resp = client.get(
                _models_url(base_url),
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
    except _httpx.HTTPError as exc:
        raise FreeTokenError(
            f"Cannot reach FreeToken at {base_url} ({type(exc).__name__}). "
            "Is the user-owned FreeToken server running at that LAN endpoint? "
            "this tool never starts it for you."
        ) from None
    except ValueError as exc:
        raise FreeTokenError(
            f"FreeToken at {base_url} returned a non-JSON /models payload."
        ) from exc
    except Exception as exc:
        raise FreeTokenError(
            f"Cannot reach FreeToken at {base_url} ({type(exc).__name__}). "
            "Is the user-owned FreeToken server running at that LAN endpoint? "
            "this tool never starts it for you."
        ) from None
    ids: list[str] = []
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                mid = entry.get("id")
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
    return ids


def resolve_model(base_url: str, explicit: str | None = None) -> str:
    """Resolve the model id: explicit > ``FREETOKEN_MODEL`` env > /models[0]."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = _os.environ.get("FREETOKEN_MODEL", "").strip()
    if env:
        return env
    ids = list_models(base_url)
    if not ids:
        raise FreeTokenError(
            f"FreeToken at {base_url} listed no models. Set FREETOKEN_MODEL "
            "to a model id served by your local server."
        )
    return ids[0]


@dataclass(frozen=True)
class FreeTokenConfig:
    """Resolved, private-network-checked FreeToken connection settings."""

    base_url: str
    model: str
    api_key: str = "freetoken-local"
    timeout_s: float = MODEL_TIMEOUT_S

    @classmethod
    def resolve(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = MODEL_TIMEOUT_S,
    ) -> "FreeTokenConfig":
        base = resolve_base_url(base_url)
        name = resolve_model(base, model)
        key = (api_key or _os.environ.get("FREETOKEN_API_KEY", "") or "freetoken-local").strip()
        if not (0 < timeout_s <= 300):
            raise FreeTokenError("model timeout must be within (0, 300]s")
        return cls(base_url=base, model=name, api_key=key, timeout_s=float(timeout_s))


def _provider_http_client(timeout_s: float) -> "_httpx2.AsyncClient":
    """Build the owned inference HTTP client (typed httpx2 seam).

    PydanticAI 2.52 accepts ``httpx2.AsyncClient`` without warnings (the
    legacy ``httpx.AsyncClient`` path warns); ``httpx2>=2.7`` is a declared
    dependency of ``pydantic-ai-slim``. LAN safety: no proxy env, no
    redirects, bounded timeout. The caller owns the client and must close
    it via :func:`close_session` when the run settles.
    """
    import httpx2

    return httpx2.AsyncClient(
        timeout=timeout_s, follow_redirects=False, trust_env=False
    )


@dataclass
class FreeTokenSession:
    """Owned inference session: PydanticAI model plus its HTTP client."""

    model: "_Model"
    http_client: "_httpx2.AsyncClient"


def build_session(config: FreeTokenConfig) -> FreeTokenSession:
    """Build a PydanticAI Chat Completions model for the FreeToken endpoint.

    Uses ``OpenAIChatModel`` (Chat Completions path, correct for
    OpenAI-compatible local servers) with ``OpenAIProvider``. No direct
    OpenAI SDK imports or calls; PydanticAI owns all inference HTTP, while
    this module owns the underlying client lifetime.
    """
    # Local imports so modules without a model dependency stay light.
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client = _provider_http_client(config.timeout_s)
    provider = OpenAIProvider(
        base_url=config.base_url, api_key=config.api_key, http_client=client
    )
    return FreeTokenSession(
        model=OpenAIChatModel(config.model, provider=provider),  # type: ignore[arg-type]
        http_client=client,
    )


def close_session(session: FreeTokenSession) -> None:
    """Deterministically close a session's HTTP client (sync-safe).

    Safe when no loop is running (fresh loop) and when called inside a
    running loop (scheduled, never nested). Only call once the run has
    settled; prefer the in-loop cleanup in ``analyze`` for live runs.
    """
    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        loop.create_task(session.http_client.aclose())
    else:
        _asyncio.run(session.http_client.aclose())
