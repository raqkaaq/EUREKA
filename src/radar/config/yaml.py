"""Safe, typed package-resource configuration loading; no service dependencies."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import yaml
from pydantic import BaseModel, ValidationError

MAX_CONFIG_BYTES = 64 * 1024


class ConfigurationError(ValueError):
    """Invalid packaged YAML, with no raw configuration values in the message."""


class _UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in keys:
                raise yaml.YAMLError("mapping keys must be unique strings")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_yaml[T: BaseModel](text: str, model: type[T], *, name: str = "configuration") -> T:
    """Parse data-only YAML and validate its complete typed contract."""
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ConfigurationError(f"{name}: configuration exceeds 64 KiB")
    try:
        data = yaml.load(text, Loader=_UniqueSafeLoader)
    except (yaml.YAMLError, RecursionError):
        raise ConfigurationError(f"{name}: malformed or unsafe YAML") from None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        categories = ", ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'document'}: {error['type']}"
            for error in exc.errors(include_input=False, include_url=False)[:8]
        )
        raise ConfigurationError(f"{name}: invalid configuration ({categories})") from None


@lru_cache(maxsize=16)
def load_yaml[T: BaseModel](package: str, filename: str, model: type[T]) -> T:
    """Load immutable defaults once per process, independent of working directory."""
    try:
        data = files(package).joinpath(filename).read_bytes()
        if len(data) > MAX_CONFIG_BYTES:
            raise ConfigurationError(f"{filename}: configuration exceeds 64 KiB")
        text = data.decode("utf-8")
    except (OSError, UnicodeError):
        raise ConfigurationError(f"{filename}: cannot read packaged configuration") from None
    return parse_yaml(text, model, name=filename)
