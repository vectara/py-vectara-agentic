import unittest

from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.sub_query_workflow import SubQuestionQueryWorkflow

def mult(x: float, y: float):
    """
    Multiply two numbers.
    """
    return x * y

def add(x: float, y: float):
    """
    Add two numbers.
    """
    return x + y

class TestWorkflowPackage(unittest.IsolatedAsyncioTestCase):

    async def test_workflow(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            workflow_cls = SubQuestionQueryWorkflow,
        )

        inputs = SubQuestionQueryWorkflow.InputsModel(
            query="Compute 5 times 3, then add 7 to the result. respond with the final answer only."
        )
        res = await agent.run(inputs=inputs)
        self.assertEqual(res.response, "22")


if __name__ == "__main__":
    unittest.main()
