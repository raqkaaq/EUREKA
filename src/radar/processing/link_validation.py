"""Pure link validation shared by source identities and evidence attachment.

No network resolution or fetching. Invalid metadata returns False, never an
exception carrying an untrusted URL. This is link syntax, not URL reachability
or protection for a crawler (the radar does not fetch evidence links).
"""

from urllib.parse import urlsplit


def is_http_link(value: str) -> bool:
    """HTTP(S), a valid authority, no credentials or unsafe raw characters."""
    if not value or any(char.isspace() or ord(char) < 32 or ord(char) == 127
                        or char in '\\<>"' for char in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port  # Access validates malformed and out-of-range ports.
        return (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                and parsed.username is None and parsed.password is None)
    except ValueError:
        return False


def is_openalex_work_link(value: str) -> bool:
    """An HTTP(S) OpenAlex work identity, not a deceptive domain substring."""
    if not is_http_link(value):
        return False
    parsed = urlsplit(value)
    suffix = parsed.path.removeprefix("/W")
    return (parsed.hostname == "openalex.org" and parsed.port is None
            and parsed.path.startswith("/W") and bool(suffix)
            and suffix.isascii() and suffix.isalnum()
            and not parsed.query and not parsed.fragment)
