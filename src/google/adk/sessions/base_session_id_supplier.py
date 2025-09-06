from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseSessionIdSupplier(ABC):
  @abstractmethod
  async def get_session_id(self, app_name: str, user_id: str, initial_state: Optional[dict[str, Any]] = None) -> str:
    """
    Get a session id for a given app name and user id.
    """
    pass
