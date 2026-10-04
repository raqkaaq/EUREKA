"""Private-network Strata provider: LAN-only model configuration,
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

EXAMPLE_BASE_URL = "http://192.168.1.20:8080/v1"
MODELS_TIMEOUT_S = 5.0
MODEL_TIMEOUT_S = 60.0
DISABLE_THINKING_ENV_VAR = "STRATA_DISABLE_THINKING"


def _setting(name: str) -> str:
    """Strata settings take precedence; retired FreeToken names are fallbacks."""
    return (_os.environ.get(f"STRATA_{name}", "").strip()
            or _os.environ.get(f"FREETOKEN_{name}", "").strip())


def model_from_env() -> str:
    """Configured model name without network I/O, including legacy fallback."""
    return _setting("MODEL")


def resolve_disable_thinking(explicit: bool = False) -> bool:
    """Whether to send the server-specific thinking-disable key.

    Opt-in only: explicit flag wins, else the ``STRATA_DISABLE_THINKING``
    env var (1/true/yes/on). Default omits the key entirely -- no backend
    is assumed to support it.
    """
    if explicit:
        return True
    return _setting("DISABLE_THINKING").lower() in {
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


class StrataError(RuntimeError):
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
        raise StrataError(
            f"Cannot resolve Strata host {host!r}; use a LAN IP address or "
            "a hostname that resolves on this machine."
        ) from exc
    return {str(answer[4][0]).split("%", 1)[0] for answer in answers if answer[4]}


def _checked_host(base_url: str) -> str:
    """Validate URL syntax without DNS or other I/O."""
    raw = (base_url or "").strip()
    if not raw:
        raise StrataError(
            "Strata base URL is not configured; set STRATA_BASE_URL (legacy FREETOKEN_BASE_URL) or "
            f"pass --base-url with a loopback/private-LAN URL (for example {EXAMPLE_BASE_URL})."
        )
    try:
        parts = _parse.urlsplit(raw)
    except ValueError as exc:
        raise StrataError(f"Strata base URL is not a valid URL ({raw!r}): {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise StrataError(
            f"Strata base URL must use http(s) (got {raw!r}); "
            f"for example {EXAMPLE_BASE_URL}."
        )
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host or parts.username is not None or parts.password is not None:
        raise StrataError(
            f"Strata base URL must contain a host and no embedded credentials ({raw!r})."
        )
    try:
        _ = parts.port
    except ValueError as exc:
        raise StrataError(f"Strata base URL has an invalid port ({raw!r}).") from exc
    return host


def _checked_addresses(raw: str, addresses: set[str]) -> str:
    if not addresses or any(not _is_allowed_address(address) for address in addresses):
        raise StrataError(
            f"Refusing non-local Strata base URL {raw!r}: the host must resolve "
            "only to loopback, RFC 1918, IPv6 ULA, or RFC 6598 addresses."
        )
    return raw.rstrip("/")


def check_local_network(base_url: str) -> str:
    """Accept private LAN URLs only when every resolved address is allowed."""
    raw = (base_url or "").strip()
    host = _checked_host(raw)
    if host in _LOCALHOST_NAMES:
        addresses = {"127.0.0.1"}
    else:
        try:
            _ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            addresses = _resolved_addresses(host)
        else:
            addresses = {host.split("%", 1)[0]}
    return _checked_addresses(raw, addresses)


def resolve_base_url(explicit: str | None = None) -> str:
    """Resolve and local-network-check the Strata base URL.

    Precedence: explicit arg > ``STRATA_BASE_URL`` env > legacy ``FREETOKEN_BASE_URL`` env > default.
    """
    env = _setting("BASE_URL")
    return check_local_network(explicit or env)


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def list_models(base_url: str, timeout: float = MODELS_TIMEOUT_S,
                *, api_key: str | None = None) -> list[str]:
    """List model ids from the local ``/models`` endpoint (httpx, bounded).

    Never includes credentials, headers, or raw bodies in errors.
    """
    base_url = check_local_network(base_url)
    if not (0 < timeout <= 30):
        raise StrataError("models lookup timeout must be within (0, 30]s")
    headers = {"Accept": "application/json"}
    key = (api_key or _setting("API_KEY")).strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with _httpx.Client(trust_env=False, follow_redirects=False) as client:
            resp = client.get(
                _models_url(base_url),
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
    except _httpx.HTTPError as exc:
        raise StrataError(
            f"Cannot reach Strata at {base_url} ({type(exc).__name__}). "
            "Is the user-owned Strata server running at that LAN endpoint? "
            "this tool never starts it for you."
        ) from None
    except ValueError as exc:
        raise StrataError(
            f"Strata at {base_url} returned a non-JSON /models payload."
        ) from exc
    except Exception as exc:
        raise StrataError(
            f"Cannot reach Strata at {base_url} ({type(exc).__name__}). "
            "Is the user-owned Strata server running at that LAN endpoint? "
            "this tool never starts it for you."
        ) from None
    return _model_ids(payload)


def _model_ids(payload: object) -> list[str]:
    ids: list[str] = []
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                mid = entry.get("id")
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
    return ids


def resolve_model(base_url: str, explicit: str | None = None,
                  *, api_key: str | None = None) -> str:
    """Resolve the model id: explicit > ``STRATA_MODEL`` env > legacy ``FREETOKEN_MODEL`` env > /models[0]."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = model_from_env()
    if env:
        return env
    ids = list_models(base_url, api_key=api_key)
    if not ids:
        raise StrataError(
            f"Strata at {base_url} listed no models. Set STRATA_MODEL "
            "to a model id served by your local server."
        )
    return ids[0]


@dataclass(frozen=True)
class StrataConfig:
    """Resolved, private-network-checked Strata connection settings."""

    base_url: str
    model: str
    api_key: str = "strata-local"
    timeout_s: float = MODEL_TIMEOUT_S

    @classmethod
    async def resolve_async(
        cls, base_url: str | None = None, model: str | None = None,
        api_key: str | None = None, timeout_s: float = MODEL_TIMEOUT_S,
    ) -> "StrataConfig":
        """Asynchronous screening setup with a cancellable /models lookup.

        The caller owns the enclosing stage deadline. OS hostname resolution
        uses the event loop resolver; its executor may still delay loop shutdown
        after cancellation. A configured LAN IP bypasses that limitation.
        """
        if not (0 < timeout_s <= 300):
            raise StrataError("model timeout must be within (0, 300]s")
        raw = (base_url or _setting("BASE_URL")).strip()
        host = _checked_host(raw)
        if host in _LOCALHOST_NAMES:
            addresses = {"127.0.0.1"}
        else:
            try:
                _ipaddress.ip_address(host.split("%", 1)[0])
                addresses = {host.split("%", 1)[0]}
            except ValueError:
                try:
                    answers = await _asyncio.get_running_loop().getaddrinfo(
                        host, None, type=_socket.SOCK_STREAM)
                except OSError as exc:
                    raise StrataError("Cannot resolve the configured Strata LAN host.") from exc
                addresses = {str(a[4][0]).split("%", 1)[0] for a in answers if a[4]}
        base = _checked_addresses(raw, addresses)
        key = (api_key or _setting("API_KEY") or "strata-local").strip()
        name = (model or "").strip() or model_from_env()
        if not name:
            try:
                async with _httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
                    response = await client.get(
                        _models_url(base), headers={"Authorization": f"Bearer {key}"},
                        timeout=min(timeout_s, MODELS_TIMEOUT_S))
                    response.raise_for_status()
                    ids = _model_ids(response.json())
            except (_httpx.HTTPError, ValueError) as exc:
                raise StrataError("Strata model discovery failed; check the user-owned LAN server.") from exc
            if not ids:
                raise StrataError("Strata listed no models; configure STRATA_MODEL.")
            name = ids[0]
        return cls(base_url=base, model=name, api_key=key, timeout_s=float(timeout_s))

    @classmethod
    def resolve(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = MODEL_TIMEOUT_S,
    ) -> "StrataConfig":
        base = resolve_base_url(base_url)
        key = (api_key or _setting("API_KEY") or "strata-local").strip()
        name = resolve_model(base, model, api_key=key)
        if not (0 < timeout_s <= 300):
            raise StrataError("model timeout must be within (0, 300]s")
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
class StrataSession:
    """Owned inference session: PydanticAI model plus its HTTP client."""

    model: "_Model"
    http_client: "_httpx2.AsyncClient"


def build_session(config: StrataConfig) -> StrataSession:
    """Build a PydanticAI Chat Completions model for the Strata endpoint.

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
    # Stage owners budget requests/retries. The SDK's default two network
    # retries otherwise turn a 60-second request into roughly three minutes.
    provider.client.max_retries = 0
    return StrataSession(
        model=OpenAIChatModel(config.model, provider=provider),  # type: ignore[arg-type]
        http_client=client,
    )


def close_session(session: StrataSession) -> None:
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


# Compatibility contracts for callers migrating from the retired provider.
FreeTokenConfig = StrataConfig
FreeTokenError = StrataError
FreeTokenSession = StrataSession
