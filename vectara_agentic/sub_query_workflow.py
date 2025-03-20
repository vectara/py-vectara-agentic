"""
This module contains the SubQuestionQueryEngine workflow, which is a workflow
that takes a user question and a list of tools, and outputs a list of sub-questions.
"""
import json
from pydantic import BaseModel

from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)

class SubQuestionQueryWorkflow(Workflow):
    """
    Workflow for sub-question query engine.
    """

    # Workflow inputs/outputs
    class InputsModel(BaseModel):
        """
        Inputs for the workflow.
        """
        query: str

    class OutputsModel(BaseModel):
        """
        Outputs for the workflow.
        """
        response: str

    # Workflow Event types
    class QueryEvent(Event):
        """Event for a query."""
        question: str

    class AnswerEvent(Event):
        """Event for an answer."""
        question: str
        answer: str

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        Given a user question, and a list of tools, output a list of relevant
        sub-questions, such that the answers to all the sub-questions put together
        will answer the question.
        """
        if not hasattr(ev, "inputs"):
            raise ValueError("No inputs provided to workflow Start Event.")
        if hasattr(ev, "inputs") and not isinstance(ev.inputs, self.InputsModel):
            raise ValueError(f"Expected inputs to be of type {self.InputsModel}")
        if hasattr(ev, "inputs"):
            query = ev.inputs.query
            await ctx.set("original_query", query)
            print(f"Query is {await ctx.get('original_query')}")

        if hasattr(ev, "agent"):
            await ctx.set("agent", ev.agent)
        chat_history = [str(msg) for msg in ev.agent.memory.get()]

        if hasattr(ev, "llm"):
            await ctx.set("llm", ev.llm)

        if hasattr(ev, "tools"):
            await ctx.set("tools", ev.tools)

        if hasattr(ev, "verbose"):
            await ctx.set("verbose", ev.verbose)

        llm = await ctx.get("llm")
        response = llm.complete(
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
            As an example, for the question
            "what is the name of the mayor of the largest city within 50 miles of San Francisco?",
            the sub-questions could be:
            - What is the largest city within 50 miles of San Francisco? (answer is San Jose)
            - What is the name of the mayor of San Jose?
            Here is the user question: {await ctx.get('original_query')}.
            Here are previous chat messages: {chat_history}.
            And here is the list of tools: {await ctx.get('tools')}
            """,
        )

        if await ctx.get("verbose"):
            print(f"Sub-questions are {response}")

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        await ctx.set("sub_question_count", len(sub_questions))

        for question in sub_questions:
            self.send_event(self.QueryEvent(question=question))

        return None

    @step
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        """
        Given a sub-question, return the answer to the sub-question, using the agent.
        """
        if await ctx.get("verbose"):
            print(f"Sub-question is {ev.question}")
        agent = await ctx.get("agent")
        response = await agent.achat(ev.question)
        return self.AnswerEvent(question=ev.question, answer=str(response))

    @step
    async def combine_answers(
        self, ctx: Context, ev: AnswerEvent
    ) -> StopEvent | None:
        """
        Given a list of answers to sub-questions, combine them into a single answer.
        """
        ready = ctx.collect_events(
            ev, [self.AnswerEvent] * await ctx.get("sub_question_count")
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

        llm = await ctx.get("llm")
        response = llm.complete(prompt)

        if await ctx.get("verbose"):
            print("Final response is", response)

        output = self.OutputsModel(response=str(response))
        return StopEvent(result=output)
