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

class TestWorkflowPackage(unittest.TestCase):

    def test_workflow(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            use_structured_planning = True,
            workflow_cls = SubQuestionQueryWorkflow
        )

        inputs = SubQuestionQueryWorkflow.InputsModel(
            query="What is 5 times 3 times 7. Only give the answer, nothing else."
        )
        res = agent.run(inputs)
        self.assertEqual(res.response, "1050")


if __name__ == "__main__":
    unittest.main()
