from agno.agent import Agent
from agno.models.openai import OpenAILike
import os 
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team import Team


import dotenv
dotenv.load_dotenv()

llm = OpenAILike(
    id="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

news_agent = Agent(
    name = "news_agent",
    role = "Get the latest news about a topic",
    tools = [DuckDuckGoTools()],
    instructions="""You are a news agent that can search the web for the latest news about a topic. 
Use the DuckDuckGoTools to search for news articles and summarize them for the user.""",
    model = llm
)

web_agent = Agent(
    name = "web_agent",
    role = "Get information about a topic from the web",
    tools = [DuckDuckGoTools()],
    instructions="""You are a web agent that can search the web for information about a topic. 
Use the DuckDuckGoTools to search for information and answer the user's questions.""",
    model = llm
)

team = Team(
    name = "news_and_web_team",
    members = [news_agent, web_agent],
    role = "You are a team of agents that can work together to answer the user's questions. The news_agent can search for the latest news about a topic, while the web_agent can search for general information about a topic. When a user asks a question, the team will decide which agent is best suited to answer the question and will use that agent to find the answer. If the question is about the latest news, the team will use the news_agent. If the question is about general information, the team will use the web_agent.",
    instructions="""You are a team of agents that can work together to answer the user's questions.
The news_agent can search for the latest news about a topic, while the web_agent can search for general information about a topic.
When a user asks a question, the team will decide which agent is best suited to answer the question and will use that agent to find the answer. If the question is about the latest news, the team will use the news_agent. If the question is about general information, the team will use the web_agent.""",
    model = llm,
    markdown=True,
    stream=True,
    debug_mode=True)

team.cli_app(stream=True, markdown=True)
