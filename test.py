# Create web agent and manager agent structure
web_agent = ToolCallingAgent(
    tools=[],           # Add required tools
    model=None,         # Add model
    max_steps=5,        # Adjust steps
    name="",           # Add name
    description=""      # Add description
)

manager_agent = CodeAgent(
     model=InferenceClientModel(),
    tools=[],
    managed_agents=[web_agent],
    max_steps=15,
)