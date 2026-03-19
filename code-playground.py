import asyncio

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b


async def main():
# initialize llm
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    # initialize agent
    agent = AgentWorkflow.from_tools_or_functions(
        [FunctionTool.from_defaults(multiply)],
        llm=llm
    )

    response = await agent.run("What is 2 times 2?")
    print(response)


asyncio.run(main())
