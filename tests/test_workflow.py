import unittest

from pydantic import BaseModel

from llama_index.core.workflow import WorkflowTimeoutError

from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.sub_query_workflow import SubQuestionQueryWorkflow, SequentialSubQuestionsWorkflow

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

    async def test_sub_query_workflow(self):
        tools = [ToolsFactory().create_tool(mult)] + [ToolsFactory().create_tool(add)]
        topic = "AI topic"
        instructions = "You are a helpful AI assistant."
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            workflow_cls = SubQuestionQueryWorkflow,
        )

        inputs = SubQuestionQueryWorkflow.InputsModel(
            query="Compute 5 times 3, then add 7 to the result."
        )
        res = await agent.run(inputs=inputs)
        self.assertIn("22", res.response)

        inputs = SubQuestionQueryWorkflow.InputsModel(
            query="what is the sum of 10 with 21, and the multiplication of 3 and 6?"
        )
        res = await agent.run(inputs=inputs)
        self.assertIn("31", res.response)
        self.assertIn("18", res.response)

    async def test_seq_sub_query_workflow(self):
        tools = [ToolsFactory().create_tool(mult)] + [ToolsFactory().create_tool(add)]
        topic = "AI topic"
        instructions = "You are a helpful AI assistant."
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            workflow_cls = SequentialSubQuestionsWorkflow,
        )

        inputs = SequentialSubQuestionsWorkflow.InputsModel(
            query="Compute 5 times 3, then add 7 to the result."
        )
        res = await agent.run(inputs=inputs, verbose=True)
        self.assertIn("22", res.response)

class TestWorkflowFailure(unittest.IsolatedAsyncioTestCase):

    async def test_workflow_failure(self):
        tools = [ToolsFactory().create_tool(mult)] + [ToolsFactory().create_tool(add)]
        topic = "AI topic"
        instructions = "You are a helpful AI assistant."
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            workflow_cls = SubQuestionQueryWorkflow,
            workflow_timeout = 1
        )

        inputs = SubQuestionQueryWorkflow.InputsModel(
            query="Compute 5 times 3, then add 7 to the result."
        )

        res = None

        try:
            res = await agent.run(inputs=inputs)
        except Exception as e:
            self.assertIsInstance(e, WorkflowTimeoutError)

        self.assertIsNone(res)

    async def test_workflow_with_fail_class(self):
        tools = [ToolsFactory().create_tool(mult)] + [ToolsFactory().create_tool(add)]
        topic = "AI topic"
        instructions = "You are a helpful AI assistant."

        class SubQuestionQueryWorkflowWithFailClass(SubQuestionQueryWorkflow):
            class OutputModelOnFail(BaseModel):
                """
                In case of failure, returns the user's original query
                """
                original_query: str

        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            workflow_cls = SubQuestionQueryWorkflowWithFailClass,
            workflow_timeout = 1
        )

        inputs = SubQuestionQueryWorkflow.InputsModel(
            query="Compute 5 times 3, then add 7 to the result."
        )

        res = None

        try:
            res = await agent.run(inputs=inputs)
        except Exception as e:
            assert isinstance(e, WorkflowTimeoutError)

        self.assertIsInstance(res, SubQuestionQueryWorkflowWithFailClass.OutputModelOnFail)
        self.assertEqual(res.original_query, "Compute 5 times 3, then add 7 to the result.")


if __name__ == "__main__":
    unittest.main()
