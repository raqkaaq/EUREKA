"""Compatibility imports for the retired FreeToken provider.

The canonical local chat adapter and configuration now live in
:mod:`radar.provider.strata`. No duplicate transport implementation.
"""

from radar.provider.strata import (
    EXAMPLE_BASE_URL, MODELS_TIMEOUT_S, MODEL_TIMEOUT_S, DISABLE_THINKING_ENV_VAR,
    StrataConfig as FreeTokenConfig, StrataError as FreeTokenError,
    StrataSession as FreeTokenSession, build_session, check_local_network,
    close_session, list_models, resolve_base_url, resolve_disable_thinking,
    resolve_model, thinking_extra_body,
)
