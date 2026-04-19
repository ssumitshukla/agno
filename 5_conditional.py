from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.workflow import Step, Workflow, Condition, StepInput, StepOutput

# ======================== LLM ==============================
import dotenv
import os
dotenv.load_dotenv()
llm = OpenAILike(
    id="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

def review_email_condition(step_input: StepInput) -> bool:
    """Condition to check if the email has the subject line"""
    email_content = step_input.previous_step_content or ""
    if email_content:
        # content of the email
        email_content = email_content.lower()
        # check if subject is present
        if "subject" in email_content:
            return True
        else:
            return False
    else:
        return False


def email_output(step_input: StepInput) -> StepOutput:
    """Returns the output of the drafting step"""
    email_content = step_input.get_step_content("Email Draft Step")
    
    return StepOutput(content=email_content,
                      step_name="Email Output Step",
                      executor_type="function")
    
    
# ===================== Agents =============================
# email draft agent

email_draft_agent = Agent(
    id="email-draft-agent",
    name="Email Draft Agent",
    instructions=["You are an expert in drafting emails",
                  "Draft clear and professional emails based on the user's input"],
    model=llm
)

# ======================= Steps ===========================
# email drafting step
email_draft_step = Step(
    name="Email Draft Step",
    agent=email_draft_agent,
    description="Drafts and email based on user's input prompt"
)

# email output step
email_output_step = Step(
    name="Email output Step",
    executor=email_output,
    description="Outputs my email to the end user"
)

# conditional step
review_email_step = Condition(
    evaluator=review_email_condition,
    steps=[email_output_step],
    name="Review Email Step",
    description="Reviews the drafted email if it contains the subject line"
)

# =================== Workflow ============================
workflow = Workflow(
    id="email-workflow",
    name="Email Drafting and Review Workflow",
    steps=[email_draft_step,review_email_step],
    description="A workflow that drafts an email based on user and reviews it if the subject line is present or not"
)


# execute the workflow
workflow.print_response(input="Draft an email to schedule a meeting with my technical team at 6pm and do not give a subject")