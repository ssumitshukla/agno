from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.workflow import Workflow, Step, Loop, StepOutput


# ======================== LLM ==============================
import dotenv
import os
dotenv.load_dotenv()
llm = OpenAILike(
    id="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

def word_count_condition(step_output: list[StepOutput]) -> bool:
    """Condition to check if the story is less than 300 words"""
    # check if there is output from the prev step
    if step_output:
        # looping through the list of step output
        for output in step_output:
            # fetching the content
            story_content: str = output.content
            # calculating the word count
            word_count: int = len(story_content.split(" "))
            # checking if word count is less than 300 words
            if word_count <= 300:
                return True
            else:
                return False
    else:
        return False


# ========================== Agent =======================

# story generation agent
story_generation_agent = Agent(
    id="story-generation-agent",
    name="Story Generation Agent",
    instructions=["You are an expert story writer.",
                  "Create engaging and imaginative stories based on user request",
                  "stick to the word limit requested by user"],
    model=llm
)


# =============================== Steps ============================

# create agent step
story_generation_step = Step(
    name="Story Generation Step",
    agent=story_generation_agent,
    description="Generates a short story based on user's prompt"
)

# define the looping step
looping_step = Loop(
    steps=[story_generation_step],
    name="Story generation loop",
    description="Generates stories in loop till condition is met",
    end_condition=word_count_condition,
    max_iterations=5
)

# ========================= Workflow ================================
workflow = Workflow(
    id="story-generation-workflow",
    name="Story Generation Workflow",
    steps=[looping_step],
    description="A worflow that generates short stories and ensures they are less than 300 words using a looping mechanism"
)


# input to the workflow
workflow.print_response(input="Title: A magical adventure in a castle",
                        stream=True,
                        markdown=True)