from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
from smolagents.models import OpenAIServerModel

model = OpenAIServerModel(
    model_id="llama3",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool()
    ],
    model=model
)

agent.run("Whats the owner of google")
