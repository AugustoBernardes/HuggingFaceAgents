from smolagents import CodeAgent, GoogleSearchTool, InferenceClientModel
from dotenv import load_dotenv
load_dotenv()

from Tools.city_coordinates import cities_coordinates
from Tools.map_generator import map_generation

model = InferenceClientModel()

web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(provider="serper"),
    ],
    name="web_agent",
    description="Searches the web for factual information",
    verbosity_level=1,
    max_steps=5,
)

manager_agent = CodeAgent(
    model=model,
    tools=[cities_coordinates, map_generation],
    managed_agents=[web_agent],
    verbosity_level=3,
    max_steps=8,
    instructions="""
You are a geography assistant.

To answer questions about cities:
1. Use web_agent to find the cities.
2. Use cities_coordinates to get their coordinates.
3. Use map_generation to generate the map.
"""
)

manager_agent.run("""
Which cities will host the FIFA World Cup 2026 games in Canada ? Show them on a map.
""")