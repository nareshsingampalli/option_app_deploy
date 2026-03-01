"""Custom exceptions for option_app."""


class OptionAppError(Exception):
    """Base exception for all option_app errors."""


class SpotPriceFetchError(OptionAppError):
    """Could not resolve spot price for the requested date/time."""


class InstrumentResolutionError(OptionAppError):
    """No instruments could be resolved for the given symbol/date."""


class CandleDataError(OptionAppError):
    """Candle data fetch returned empty or invalid data."""


class StorageError(OptionAppError):
    """Failed to persist data to the storage backend."""


class ConfigurationError(OptionAppError):
    """Missing or invalid configuration (e.g. missing API token)."""


class StrategyNotFoundError(OptionAppError):
    """No suitable data strategy found for the given context."""
