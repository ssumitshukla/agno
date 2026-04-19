from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.workflow import Step, Workflow
import os 


import dotenv
dotenv.load_dotenv()

llm = OpenAILike(
    id="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# ================= Agents =======================
# essay writing agent
essay_writing_agent = Agent(
    id="essay-writing-agent",
    name="Essay Writing Agent",
    instructions=["You are an expert in writing essays",
                  "Write well structured essays on a variety of topics",
                  "Limit your response to a maximum of 500 words"],
    model=llm,
)

# extraction agent to extract imp points from the essay
extraction_agent = Agent(
    id="extraction-agent",
    name="Extraction Agent",
    instructions=["You are an expert at extracting important points from the generated essay",
                  "Summarize the key points in a concise manner",
                  "Your output should be in a good format"],
    model=llm,
    markdown=True
)

# ================= Step =====================
essay_writing_step = Step(
    name="Essay Writing Step",
    agent=essay_writing_agent,
    description="Generates an essay based on the user's topic"
)

extraction_step = Step(
    name="Information Extraction Step",
    agent=extraction_agent,
    description="Extracts important points from the essay generated in the previous step"
)


# =================== Workflow =======================

workflow = Workflow(
    id="essay-workflow",
    name="Essay Writing and Point Extraction Workflow",
    steps=[essay_writing_step, extraction_step],
    description="A workflow that first writes an essay on a given topic and then extracts important points from that essay"
)

# execute the workflow
workflow.print_response(input="The topic is: Impact of technology on education. Respond in a proper format",
                        stream=True, markdown=True)