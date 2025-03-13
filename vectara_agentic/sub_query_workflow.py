"""
This module contains the SubQuestionQueryEngine workflow, which is a workflow
that takes a user question and a list of tools, and outputs a list of sub-questions.
"""
import json
import asyncio
from typing import Any, List, Optional
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.agent.types import BaseAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM

from vectara_agentic.types import AgentResponse

class QueryEvent(Event):
    """Event for a query."""
    question: str

class AnswerEvent(Event):
    """Event for an answer."""
    question: str
    answer: str

class SubQuestionQueryEngine(Workflow):
    """Workflow for sub-question query engine."""
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        Given a user question, and a list of tools, output a list of relevant
        sub-questions, such that the answers to all the sub-questions put together
        will answer the question.
        """
        if hasattr(ev, "query"):
            await ctx.set("original_query", ev.query)
            print(f"Query is {await ctx.get('original_query')}")

        if hasattr(ev, "agent"):
            await ctx.set("agent", ev.agent)

        if hasattr(ev, "llm"):
            await ctx.set("llm", ev.llm)

        if hasattr(ev, "tools"):
            await ctx.set("tools", ev.tools)

        if hasattr(ev, "verbose"):
            await ctx.set("verbose", ev.verbose)

        response = (await ctx.get("llm")).complete(
            f"""
            Given a user question, and a list of tools, output a list of
            relevant sub-questions, such that the answers to all the
            sub-questions put together will answer the question.
            Make sure sub-questions do not result in duplicate tool calling.
            Respond in pure JSON without any markdown, like this:
            {{
                "sub_questions": [
                    "What is the population of San Francisco?",
                    "What is the budget of San Francisco?",
                    "What is the GDP of San Francisco?"
                ]
            }}
            Here is the user question: {await ctx.get('original_query')}

            And here is the list of tools: {await ctx.get('tools')}
            """
        )

        if await ctx.get("verbose"):
            print(f"Sub-questions are {response}")

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        await ctx.set("sub_question_count", len(sub_questions))

        for question in sub_questions:
            self.send_event(QueryEvent(question=question))

        return None

    @step
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        """
        Given a sub-question, return the answer to the sub-question, using the agent.
        """
        if await ctx.get("verbose"):
            print(f"Sub-question is {ev.question}")
        agent = await ctx.get("agent")
        response = agent.chat(ev.question)
        return AnswerEvent(question=ev.question, answer=str(response))

    @step
    async def combine_answers(
        self, ctx: Context, ev: AnswerEvent
    ) -> StopEvent | None:
        """
        Given a list of answers to sub-questions, combine them into a single answer.
        """
        ready = ctx.collect_events(
            ev, [AnswerEvent] * await ctx.get("sub_question_count")
        )
        if ready is None:
            return None

        answers = "\n\n".join(
            [
                f"Question: {event.question}: \n Answer: {event.answer}"
                for event in ready
            ]
        )

        prompt = f"""
            You are given an overall question that has been split into sub-questions,
            each of which has been answered. Combine the answers to all the sub-questions
            into a single answer to the original question.

            Original question: {await ctx.get('original_query')}

            Sub-questions and answers:
            {answers}
        """

        if await ctx.get("verbose"):
            print(f"Final prompt is {prompt}")

        response = (await ctx.get("llm")).complete(prompt)

        if await ctx.get("verbose"):
            print("Final response is", response)

        return StopEvent(result=str(response))

class SubQueryAgent(BaseAgent):
    """
    An agent that uses the SubQuestionQueryEngine workflow to answer questions.
    """
    def __init__(
        self,
        agent: Any,
        tools: list[FunctionTool],
        llm: LLM,
        verbose: bool = False
    ):
        self.agent = agent
        self.tools = tools
        self.verbose = verbose
        self.llm = llm
        self._chat_history = []
        self.engine = SubQuestionQueryEngine(timeout=120, verbose=self.verbose)

    async def run(self, query: str) -> str:
        """
        Run the SubQuestionQueryEngine workflow with the given query.
        """
        result = await self.engine.run(
            agent=self.agent,
            tools=self.tools,
            llm=self.llm,
            verbose=self.verbose,
            query=query
        )
        self._chat_history.append(ChatMessage.from_str(content=query, role=MessageRole.USER))
        self._chat_history.append(ChatMessage.from_str(content=result, role=MessageRole.ASSISTANT))
        return AgentResponse(response=result)

    def reset(self):
        """
        Reset the agent.
        """
        self._chat_history = []
        if hasattr(self.engine, "reset"):
            self.engine.reset()

    def clear_memory(self):
        """
        Clear the memory of the agent.
        """
        self.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """
        Return the chat history of the agent.
        """
        return self._chat_history

    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Chat with the agent.
        """
        if chat_history is not None:
            self._chat_history = chat_history
        return asyncio.run(self.run(message))

    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Chat with the agent asynchronously.
        """
        if chat_history is not None:
            self._chat_history = chat_history
        return await self.run(message)
