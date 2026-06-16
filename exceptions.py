"""
Subsystem: Application exception hierarchy.
--------------------------------------------------------------------------
Function:   Define a small set of domain-specific exceptions that carry the
            HTTP status code and user-facing message appropriate to each
            failure, so the API layer can translate any of them into a clean
            response through a single handler.
Algorithms & techniques:
  * Every exception derives from ``AppError`` and exposes a ``status_code`` and
    ``detail``. A single FastAPI exception handler maps the whole hierarchy to
    JSON responses, removing repetitive ``raise HTTPException(...)`` calls from
    the route logic.
Role in pipeline:
  * Raised by the sentiment and API layers when a stage cannot complete;
    converted to HTTP responses centrally so behaviour stays consistent.
--------------------------------------------------------------------------
"""


class AppError(Exception):
    """Base class for expected application errors.

    Subclasses set ``status_code`` and a default ``detail``; a single FastAPI
    handler then renders any of them as ``{"detail": ...}`` with the right HTTP
    status. Raising, for example, ``DataSourceUnavailable()`` anywhere in a
    request yields a clean ``503`` response without the route needing to know
    about HTTP at all.

    Attributes:
        status_code: HTTP status to return for this error.
        detail: Human-readable message safe to show the client.
    """

    status_code: int = 500
    detail: str = "Internal server error."

    def __init__(self, detail: str | None = None):
        if detail is not None:
            self.detail = detail
        super().__init__(self.detail)


class DataSourceUnavailable(AppError):
    """Raised when no data source could be reached for a search."""

    status_code = 503
    detail = ("Could not fetch data from any source. "
              "Check your internet connection and try again.")


class NoUsableData(AppError):
    """Raised when every crawled item was removed by deduplication."""

    status_code = 422
    detail = "All crawled items were removed by deduplication. Try a different keyword."


class ModelError(AppError):
    """Raised when sentiment classification fails."""

    status_code = 500
    detail = "Sentiment analysis failed. Please try again."


class ModelLoadError(ModelError):
    """Raised when a sentiment model cannot be loaded into memory."""

    detail = "Could not load the sentiment model."


class PersistenceError(AppError):
    """Raised when results cannot be written to the database."""

    status_code = 500
    detail = "Failed to save results. Please try again."


class NoResultsFound(AppError):
    """Raised when stored results are requested for an unknown keyword."""

    status_code = 404
    detail = "No results found."
