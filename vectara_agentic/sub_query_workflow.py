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
        else:
            raise ValueError("Agent not provided to workflow Start Event.")
        chat_history = [str(msg) for msg in ev.agent.memory.get()]

        if hasattr(ev, "llm"):
            await ctx.set("llm", ev.llm)
        else:
            raise ValueError("LLM not provided to workflow Start Event.")

        if hasattr(ev, "tools"):
            await ctx.set("tools", ev.tools)
        else:
            raise ValueError("Tools not provided to workflow Start Event.")

        if hasattr(ev, "verbose"):
            await ctx.set("verbose", ev.verbose)
        else:
            await ctx.set("verbose", False)

        llm = await ctx.get("llm")
        response = llm.complete(
            f"""
            Given a user question, and a list of tools, output a list of
            relevant sub-questions, such that the answers to all the
            sub-questions put together will answer the question.
            Order the sub-questions in the right order if there are dependencies.
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
            And here is the list of tools: {ev.tools}
            """,
        )

        if await ctx.get("verbose"):
            print(f"Sub-questions are {response}")

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        await ctx.set("sub_question_count", len(sub_questions))

        for question in sub_questions:
            ctx.send_event(self.QueryEvent(question=question))

        return None

    @step(num_workers=3)
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

class SequentialSubQuestionsWorkflow(Workflow):
    """
    Workflow for breaking a query into sequential sub-questions
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
        prev_answer: str
        num: int

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        Given a user question, and a list of tools, output a list of relevant
        sub-questions, such that each question depends on the response of the
        previous question, to answer the original user question.
        """
        if not hasattr(ev, "inputs"):
            raise ValueError("No inputs provided to workflow Start Event.")
        if hasattr(ev, "inputs") and not isinstance(ev.inputs, self.InputsModel):
            raise ValueError(f"Expected inputs to be of type {self.InputsModel}")
        if hasattr(ev, "inputs"):
            query = ev.inputs.query
            await ctx.set("original_query", query)

        if hasattr(ev, "agent"):
            await ctx.set("agent", ev.agent)
        else:
            raise ValueError("Agent not provided to workflow Start Event.")
        chat_history = [str(msg) for msg in ev.agent.memory.get()]

        if hasattr(ev, "llm"):
            await ctx.set("llm", ev.llm)
        else:
            raise ValueError("LLM not provided to workflow Start Event.")

        if hasattr(ev, "tools"):
            await ctx.set("tools", ev.tools)
        else:
            raise ValueError("Tools not provided to workflow Start Event.")

        if hasattr(ev, "verbose"):
            await ctx.set("verbose", ev.verbose)
        else:
            await ctx.set("verbose", False)
        if ev.verbose:
            print(f"Query is {await ctx.get('original_query')}")

        llm = await ctx.get("llm")
        orig_query = await ctx.get("original_query")
        response = llm.complete(
            f"""
            Given a user question, and a list of tools, output a list of
            relevant sequential sub-questions, such that the answers to all the
            sub-questions in sequence will answer the question, and the output
            of each question can be used as input to the subsequent question.
            Respond in pure JSON without any markdown, like this:
            {{
                "sub_questions": [
                    "What is the population of San Francisco?",
                    "Is that population larger than the population of San Jose?",
                ]
            }}
            As an example, for the question
            "what is the name of the mayor of the largest city within 50 miles of San Francisco?",
            the sub-questions could be:
            - What is the largest city within 50 miles of San Francisco?
            - Who is the mayor of this city?
            The answer to the first question is San Jose, which is given as context to the second question.
            The answer to the second question is Matt Mahan.
            Here is the user question: {orig_query}.
            Here are previous chat messages: {chat_history}.
            And here is the list of tools: {ev.tools}
            """,
        )

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        await ctx.set("sub_questions", sub_questions)
        if await ctx.get("verbose"):
            print(f"Sub-questions are {sub_questions}")

        return self.QueryEvent(question=sub_questions[0], prev_answer="", num=0)

    @step
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> StopEvent | QueryEvent:
        """
        Given a sub-question, return the answer to the sub-question, using the agent.
        """
        if await ctx.get("verbose"):
            print(f"Sub-question is {ev.question}")
        agent = await ctx.get("agent")
        sub_questions = await ctx.get("sub_questions")
        if ev.prev_answer:
            prev_question = sub_questions[ev.num - 1]
            prompt = f"""
                The answer to the question '{prev_question}' is: '{ev.prev_answer}'
                Now answer the following question: '{ev.question}'
            """
            response = await agent.achat(prompt)
        else:
            response = await agent.achat(ev.question)
        if await ctx.get("verbose"):
            print(f"Answer is {response}")

        sub_questions = await ctx.get("sub_questions")
        if ev.num + 1 < len(sub_questions):
            return self.QueryEvent(
                question=sub_questions[ev.num + 1],
                prev_answer = response.response,
                num=ev.num + 1
            )

        output = self.OutputsModel(response=response.response)
        return StopEvent(result=output)
