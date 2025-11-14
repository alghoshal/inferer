import os
import pytest
import scenario
from openai import OpenAI
from scenario._events.event_reporter import EventReporter
from scenario._events.event_bus import ScenarioEventBus

OLLAMA_MODEL = "gemma3:1b" #"llama3.2:1b"
CUSTOM_LLM_PROVIDER = "ollama"


def ollama_client():
    """
    Creates an OpenAI client configured for Ollama

    Example environment variables (customize names as needed):
    - OLLAMA_API_BASE: Ollama Base URL, defaults to "http://localhost:11434/" 
 
    Returns:
        OpenAI: Ollama client
    """
    base_url = os.getenv("OLLAMA_API_BASE","http://localhost:11434/")
    print(f"base_url: {base_url}")

    return OpenAI(base_url=base_url)


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_ollama_client():
    custom_client = ollama_client()

    class MockAgent(scenario.AgentAdapter):
        async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
            user_message = input.last_new_user_message_str()
            return f"No idea about {user_message}, but I will try to find out."

    result = await scenario.run(
        name="ollama client test",
        description="User asks a simple question about the weather",
        agents=[
            MockAgent(),
            scenario.UserSimulatorAgent(model=OLLAMA_MODEL, client=custom_client, custom_llm_provider=CUSTOM_LLM_PROVIDER),
            scenario.JudgeAgent(model=OLLAMA_MODEL, client=custom_client, custom_llm_provider=CUSTOM_LLM_PROVIDER,
                criteria=[
                    "The agent responds to the user's message",
                    "The agent offers to help if they don't know the answer",
                ],
            ),
        ],
        script=[scenario.user(), scenario.agent(), scenario.judge()],
        set_id="ollama-tests",  # Test grouping
    )

    try:
        assert result.success
    except Exception as e:
        print(f"result: {result}")
        print(f"error: {e}")
        raise e
