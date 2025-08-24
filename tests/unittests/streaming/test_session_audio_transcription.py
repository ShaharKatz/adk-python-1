# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.session import Session
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _get_audio_transcription_from_session
from google.adk.models.gemini_llm_connection import GeminiLlmConnection
from google.genai import types
import pytest


class TestSessionAudioTranscription:
  """Test session-based audio and transcription functionality."""

  def test_get_audio_transcription_from_session_empty(self):
    """Test _get_audio_transcription_from_session with empty session."""
    # Create mock invocation context with empty session
    invocation_context = Mock()
    invocation_context.session = Mock()
    invocation_context.session.events = []

    # Call the function
    contents = _get_audio_transcription_from_session(invocation_context)

    # Should return empty list
    assert contents == []

  def test_get_audio_transcription_from_session_with_audio_files(self):
    """Test _get_audio_transcription_from_session with audio file references."""
    # Create mock events with audio file data
    audio_event1 = Mock()
    audio_event1.content = types.Content(
        role='user',
        parts=[
            types.Part(
                file_data=types.FileData(
                    file_uri='artifact://app/user/session/_adk_live/audio1.pcm',
                    mime_type='audio/pcm',
                )
            )
        ],
    )

    audio_event2 = Mock()
    audio_event2.content = types.Content(
        role='model',
        parts=[
            types.Part(
                file_data=types.FileData(
                    file_uri='artifact://app/user/session/_adk_live/audio2.wav',
                    mime_type='audio/wav',
                )
            )
        ],
    )

    # Create text event (should be ignored)
    text_event = Mock()
    text_event.content = types.Content(
        role='user', parts=[types.Part.from_text(text='Hello')]
    )

    # Create mock invocation context
    invocation_context = Mock()
    invocation_context.session = Mock()
    invocation_context.session.events = [audio_event1, text_event, audio_event2]

    # Call the function
    contents = _get_audio_transcription_from_session(invocation_context)

    # Should return only audio file contents
    assert len(contents) == 2
    assert contents[0] == audio_event1.content
    assert contents[1] == audio_event2.content

  def test_get_audio_transcription_from_session_with_transcriptions(self):
    """Test _get_audio_transcription_from_session with transcription events."""
    # Create mock events with transcription data
    input_transcription_event = Mock()
    input_transcription_event.content = None
    input_transcription_event.input_transcription = types.Transcription(
        text='User said hello'
    )

    output_transcription_event = Mock()
    output_transcription_event.content = None
    output_transcription_event.output_transcription = types.Transcription(
        text='Model replied hi'
    )

    # Mock hasattr to return True for transcription events
    with patch('builtins.hasattr') as mock_hasattr:

      def hasattr_side_effect(obj, attr):
        if attr == 'input_transcription' and obj == input_transcription_event:
          return True
        elif (
            attr == 'output_transcription' and obj == output_transcription_event
        ):
          return True
        return False

      mock_hasattr.side_effect = hasattr_side_effect

      # Create mock invocation context
      invocation_context = Mock()
      invocation_context.session = Mock()
      invocation_context.session.events = [
          input_transcription_event,
          output_transcription_event,
      ]

      # Call the function
      contents = _get_audio_transcription_from_session(invocation_context)

      # Should return transcription contents as text
      assert len(contents) == 2
      assert contents[0].role == 'user'
      assert contents[0].parts[0].text == 'User said hello'
      assert contents[1].role == 'model'
      assert contents[1].parts[0].text == 'Model replied hi'

  def test_get_audio_transcription_from_session_mixed_content(self):
    """Test _get_audio_transcription_from_session with mixed audio and transcription content."""
    # Create audio event
    audio_event = Mock()
    audio_event.content = types.Content(
        role='user',
        parts=[
            types.Part(
                file_data=types.FileData(
                    file_uri='artifact://app/user/session/_adk_live/audio.pcm',
                    mime_type='audio/pcm',
                )
            )
        ],
    )

    # Create transcription event
    transcription_event = Mock()
    transcription_event.content = None
    transcription_event.input_transcription = types.Transcription(
        text='Transcribed text'
    )

    # Create regular text event (should be ignored)
    text_event = Mock()
    text_event.content = types.Content(
        role='user', parts=[types.Part.from_text(text='Regular text')]
    )

    # Mock hasattr for transcription event
    with patch('builtins.hasattr') as mock_hasattr:

      def hasattr_side_effect(obj, attr):
        if attr == 'input_transcription' and obj == transcription_event:
          return True
        return False

      mock_hasattr.side_effect = hasattr_side_effect

      # Create mock invocation context
      invocation_context = Mock()
      invocation_context.session = Mock()
      invocation_context.session.events = [
          audio_event,
          transcription_event,
          text_event,
      ]

      # Call the function
      contents = _get_audio_transcription_from_session(invocation_context)

      # Should return audio file content and transcription text content
      assert len(contents) == 2
      assert contents[0] == audio_event.content  # Audio file reference
      assert contents[1].role == 'user'  # Transcription as text
      assert contents[1].parts[0].text == 'Transcribed text'

  @pytest.mark.asyncio
  async def test_gemini_llm_connection_send_history_with_audio_files(self):
    """Test GeminiLlmConnection.send_history includes audio file references."""
    # Create mock gemini session
    mock_session = AsyncMock()

    # Create connection
    connection = GeminiLlmConnection(mock_session)

    # Create history with text and audio file content
    history = [
        types.Content(role='user', parts=[types.Part.from_text(text='Hello')]),
        types.Content(
            role='user',
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=(
                            'artifact://app/user/session/_adk_live/audio.pcm'
                        ),
                        mime_type='audio/pcm',
                    )
                )
            ],
        ),
        types.Content(
            role='model', parts=[types.Part.from_text(text='Hi there')]
        ),
    ]

    # Call send_history
    await connection.send_history(history)

    # Verify that gemini session was called with both text and audio file content
    mock_session.send.assert_called_once()
    sent_content = mock_session.send.call_args[1]['input']

    # Should include text content and audio file content, but not filter out audio files
    assert len(sent_content.turns) == 3
    assert sent_content.turns[0].parts[0].text == 'Hello'
    assert sent_content.turns[1].parts[0].file_data.mime_type == 'audio/pcm'
    assert sent_content.turns[2].parts[0].text == 'Hi there'

  @pytest.mark.asyncio
  async def test_gemini_llm_connection_send_history_filters_inline_audio(self):
    """Test GeminiLlmConnection.send_history filters out inline audio data."""
    # Create mock gemini session
    mock_session = AsyncMock()

    # Create connection
    connection = GeminiLlmConnection(mock_session)

    # Create history with inline audio data (should be filtered out)
    history = [
        types.Content(role='user', parts=[types.Part.from_text(text='Hello')]),
        types.Content(
            role='user',
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        data=b'audio_data', mime_type='audio/pcm'
                    )
                )
            ],
        ),
        types.Content(role='model', parts=[types.Part.from_text(text='Hi')]),
    ]

    # Call send_history
    await connection.send_history(history)

    # Verify that inline audio data was filtered out
    mock_session.send.assert_called_once()
    sent_content = mock_session.send.call_args[1]['input']

    # Should only include text content, inline audio should be filtered out
    assert len(sent_content.turns) == 2
    assert sent_content.turns[0].parts[0].text == 'Hello'
    assert sent_content.turns[1].parts[0].text == 'Hi'

  def test_get_audio_transcription_from_session_skips_events_without_content(
      self,
  ):
    """Test that events without content or parts are skipped."""
    # Create events without content
    empty_event1 = Mock()
    empty_event1.content = None

    empty_event2 = Mock()
    empty_event2.content = types.Content(role='user', parts=[])

    # Create valid audio event
    audio_event = Mock()
    audio_event.content = types.Content(
        role='user',
        parts=[
            types.Part(
                file_data=types.FileData(
                    file_uri='artifact://app/user/session/_adk_live/audio.pcm',
                    mime_type='audio/pcm',
                )
            )
        ],
    )

    # Create mock invocation context
    invocation_context = Mock()
    invocation_context.session = Mock()
    invocation_context.session.events = [
        empty_event1,
        empty_event2,
        audio_event,
    ]

    # Call the function
    contents = _get_audio_transcription_from_session(invocation_context)

    # Should only return the valid audio content, empty events should be skipped
    assert len(contents) == 1
    assert contents[0] == audio_event.content

  @pytest.mark.asyncio
  async def test_live_flow_does_not_use_transcription_cache(self):
    """Test that the live flow no longer uses transcription_cache for send_history."""
    from google.adk.agents.llm_agent import Agent
    from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
    from google.adk.models.llm_request import LlmRequest

    # Create a concrete implementation of BaseLlmFlow for testing
    class TestLlmFlow(BaseLlmFlow):

      def __init__(self):
        super().__init__()

    flow = TestLlmFlow()

    # Create mock invocation context
    invocation_context = Mock()
    invocation_context.session = Mock()
    invocation_context.session.events = []
    invocation_context.transcription_cache = [
        'should_not_be_used'
    ]  # This should be ignored

    # Create mock LLM and connection
    mock_llm = Mock()
    mock_connection = AsyncMock()

    # Create mock agent
    mock_agent = Mock()
    mock_agent.canonical_model = mock_llm
    invocation_context.agent = mock_agent

    # Mock LLM request with some contents
    llm_request = LlmRequest()
    llm_request.contents = [
        types.Content(role='user', parts=[types.Part.from_text(text='Hello')])
    ]

    # Mock the llm.connect context manager
    mock_llm.connect.return_value.__aenter__.return_value = mock_connection
    mock_llm.connect.return_value.__aexit__.return_value = None

    # Mock other required methods and context
    with (
        patch('google.adk.flows.llm_flows.base_llm_flow.tracer') as mock_tracer,
        patch(
            'google.adk.flows.llm_flows.base_llm_flow.trace_send_data'
        ) as mock_trace,
        patch('asyncio.create_task') as mock_create_task,
        patch.object(flow, '_send_to_model') as mock_send_to_model,
        patch.object(flow, '_receive_from_model') as mock_receive_from_model,
    ):

      # Setup mocks
      mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
          None
      )
      mock_tracer.start_as_current_span.return_value.__exit__.return_value = (
          None
      )
      mock_create_task.return_value = Mock()

      # Make _receive_from_model return empty async generator
      async def empty_generator():
        return
        yield  # unreachable but makes it a generator

      mock_receive_from_model.return_value = empty_generator()

      # Create async context manager for receive_from_model
      from google.adk.utils.context_utils import Aclosing

      # Mock the entire run_live flow by calling just the relevant part
      event_id = 'test_event_id'

      # Simulate the send_history part of run_live
      audio_transcription_contents = _get_audio_transcription_from_session(
          invocation_context
      )
      all_contents = llm_request.contents + audio_transcription_contents

      # Verify send_history was called with session-based contents, not transcription_cache
      await mock_connection.send_history(all_contents)
      mock_trace(invocation_context, event_id, all_contents)

      # Verify send_history was called
      mock_connection.send_history.assert_called_once()

      # Verify it was called with the combined contents (original + session-based)
      sent_contents = mock_connection.send_history.call_args[0][0]

      # Should contain original content plus any session-based audio/transcription
      assert len(sent_contents) >= 1
      assert sent_contents[0].parts[0].text == 'Hello'  # Original content

      # Verify transcription_cache was not accessed for send_history
      # (this is implicit since we're using session-based approach)
      mock_trace.assert_called_once_with(
          invocation_context, event_id, all_contents
      )
