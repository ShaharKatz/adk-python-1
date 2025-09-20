from typing import Optional
from typing import Any
from typing_extensions import override
from google.adk.sessions.base_session_id_supplier import BaseSessionIdSupplier
from google.adk.sessions.base_session_service import BaseSessionService


class PreCreatedSessionIdSupplier(BaseSessionIdSupplier):
  def __init__(self, session_service: BaseSessionService):
    self.session_service = session_service

  @override
  async def get_session_id(self, app_name: Optional[str], user_id: Optional[str], initial_state: Optional[dict[str, Any]] = None) -> str:
    created_session = await self.session_service.create_session(
      app_name=app_name, 
      user_id=user_id, 
      state=initial_state
    )
    return created_session.id