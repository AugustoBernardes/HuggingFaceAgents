import asyncio

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import Context

# Tools
from llama_index.core.tools import FunctionTool
from Tools.weather import get_weather
from Tools.nhl_tool import build_nhl_tool
from Tools.company_documentation_tool import build_company_documentation_tool

async def main():

    # LLM
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
    )

    # # External community tool (Not used)
    # tool_spec = GmailToolSpec()
    # gmail_tools = tool_spec.to_tool_list()

    nhl_tool = await build_nhl_tool(llm=llm)
    company_documentation_tool = await build_company_documentation_tool(llm=llm)

    # Agent with Tools (RAG & Function)
    agent = AgentWorkflow.from_tools_or_functions(
        [nhl_tool,company_documentation_tool,FunctionTool.from_defaults(get_weather)],
        llm=llm,
        verbose=False
    )

    context = Context(agent)

    hockey_response = await agent.run(user_msg="Hello my name is Augusto and i would like to know the teams that are on NHL playoffs ?", ctx=context)
    print(hockey_response)

    company_docs = await agent.run(user_msg="Whats the owner of this company ? Whats the company goals ?", ctx=context)
    print(company_docs)

    weather_response = await agent.run(user_msg="Whats the weather in Quirinoópolis today ? And whats my name ?", ctx=context)
    print(weather_response)

asyncio.run(main())