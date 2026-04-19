from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.workflow import Workflow, Step, Loop, StepOutput

import dotenv
import os
dotenv.load_dotenv()

llm = OpenAILike(
    id="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

BASE_INSTRUCTIONS = [
    "You are an expert story writer.",
    "Create engaging and imaginative stories based on the user request.",
    "Strictly respect any word-limit instructions given in the prompt.",
]

# ========================== Agents ==========================

story_generation_agent = Agent(
    id="story-generation-agent",
    name="Story Generation Agent",
    instructions=BASE_INSTRUCTIONS,
    model=llm,
)

# ========================== Condition =======================

def word_count_condition(step_output: list[StepOutput]) -> bool:
    """
    Returns True (stop) when story is <= 300 words.
    On failure, injects feedback into the agent's instructions for the next iteration.
    """
    if not step_output:
        return False

    for output in step_output:
        word_count = len(output.content.split())
        if word_count <= 300:
            story_generation_agent.instructions = BASE_INSTRUCTIONS
            return True
        else:
            excess = word_count - 300
            story_generation_agent.instructions = BASE_INSTRUCTIONS + [
                f"FEEDBACK: Your previous story was {word_count} words — {excess} words over the 300-word limit. "
                "Rewrite it to be strictly under 300 words while keeping it engaging."
            ]
            return False

    return False

# ========================== Steps ===========================

story_generation_step = Step(
    name="Story Generation Step",
    agent=story_generation_agent,
    description="Generates a short story based on the user's prompt",
)

looping_step = Loop(
    steps=[story_generation_step],
    name="Story Generation Loop",
    description="Generates stories in a loop until the story is <= 300 words",
    end_condition=word_count_condition,
    max_iterations=5,
)

# ========================= Workflow =========================

workflow = Workflow(
    id="story-generation-workflow-feedback",
    name="Story Generation Workflow with Feedback",
    steps=[looping_step],
    description=(
        "Generates short stories and uses per-iteration feedback to enforce "
        "the 300-word limit via a looping mechanism."
    ),
)

workflow.print_response(
    input="Title: A magical adventure in a castle",
    stream=True,
    markdown=True,
)
