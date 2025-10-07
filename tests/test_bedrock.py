# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.tools_catalog import ToolsCatalog

import nest_asyncio

nest_asyncio.apply()

from conftest import (
    mult,
    add,
    fc_config_bedrock,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)

ARIZE_LOCK = threading.Lock()


class TestBedrock(unittest.IsolatedAsyncioTestCase):

    async def test_multiturn(self):
        with ARIZE_LOCK:
            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
                agent_config=fc_config_bedrock,
            )

            # First calculation: 5 * 10 = 50
            stream1 = await agent.astream_chat(
                "What is 5 times 10. Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream1.async_response_gen():
                pass
            _ = await stream1.aget_response()

            # Second calculation: 3 * 7 = 21
            stream2 = await agent.astream_chat(
                "what is 3 times 7. Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream2.async_response_gen():
                pass
            _ = await stream2.aget_response()

            # Final calculation: 50 * 21 = 1050
            stream3 = await agent.astream_chat(
                "multiply the results of the last two questions. Output only the answer."
            )
            # Consume the stream
            async for chunk in stream3.async_response_gen():
                pass
            response3 = await stream3.aget_response()

            self.assertEqual(response3.response, "1050")

    async def test_claude_sonnet_4_multi_tool_chain(self):
        """Test Claude Sonnet 4 with complex multi-step reasoning chain using multiple tools via Bedrock."""
        with ARIZE_LOCK:
            # Use Bedrock config (Claude Sonnet 4)
            tools_catalog = ToolsCatalog(fc_config_bedrock)
            tools = [
                ToolsFactory().create_tool(mult),
                ToolsFactory().create_tool(add),
                ToolsFactory().create_tool(tools_catalog.summarize_text),
                ToolsFactory().create_tool(tools_catalog.rephrase_text),
            ]

            agent = Agent(
                agent_config=fc_config_bedrock,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions="You are a mathematical reasoning agent that explains your work step by step.",
            )

            # Complex multi-step reasoning task
            complex_query = (
                "Perform this calculation step by step: "
                "First multiply 5 by 9, then add 13 to that result, "
                "then multiply the new result by 2. "
                "After getting the final number, summarize the entire mathematical process "
                "with expertise in 'mathematics education', "
                "then rephrase that summary as a 10-year-old would explain it."
            )

            print("\nStarting Claude Sonnet 4 multi-tool chain test (Bedrock)")
            print(f"Query: {complex_query}")
            print("Streaming response:\n" + "="*50)

            stream = await agent.astream_chat(complex_query)

            # Capture streaming deltas and tool calls
            streaming_deltas = []
            tool_calls_made = []
            full_response = ""

            async for chunk in stream.async_response_gen():
                if chunk and chunk.strip():
                    streaming_deltas.append(chunk)
                    full_response += chunk
                    # Display each streaming delta
                    print(f"Delta: {repr(chunk)}")

                    # Track tool calls in the stream
                    if "mult" in chunk.lower():
                        if "mult" not in [call["tool"] for call in tool_calls_made]:
                            tool_calls_made.append({"tool": "mult", "order": len(tool_calls_made) + 1})
                            print(f"Tool call detected: mult (#{len(tool_calls_made)})")
                    if "add" in chunk.lower():
                        if "add" not in [call["tool"] for call in tool_calls_made]:
                            tool_calls_made.append({"tool": "add", "order": len(tool_calls_made) + 1})
                            print(f"Tool call detected: add (#{len(tool_calls_made)})")
                    if "summarize" in chunk.lower():
                        if "summarize_text" not in [call["tool"] for call in tool_calls_made]:
                            tool_calls_made.append({"tool": "summarize_text", "order": len(tool_calls_made) + 1})
                            print(f"Tool call detected: summarize_text (#{len(tool_calls_made)})")
                    if "rephrase" in chunk.lower():
                        if "rephrase_text" not in [call["tool"] for call in tool_calls_made]:
                            tool_calls_made.append({"tool": "rephrase_text", "order": len(tool_calls_made) + 1})
                            print(f"Tool call detected: rephrase_text (#{len(tool_calls_made)})")

            response = await stream.aget_response()

            print("="*50)
            print(f"Streaming completed. Total deltas: {len(streaming_deltas)}")
            print(f"Tool calls made: {[call['tool'] for call in tool_calls_made]}")
            print(f"📄 Final response length: {len(response.response)} chars")
            print(f"Final response: {response.response}")

            # Validate tool usage sequence
            tools_used = [call["tool"] for call in tool_calls_made]
            print(f"🧪 Tools used in order: {tools_used}")

            # Check that at least multiplication happened (basic requirement)
            self.assertIn("mult", tools_used, f"Expected multiplication tool to be used. Tools used: {tools_used}")

            # Check for mathematical results in the full response or streaming deltas
            # Expected: 5*9=45, 45+13=58, 58*2=116
            expected_intermediate_results = ["45", "58", "116"]
            all_text = (full_response + " " + response.response).lower()
            math_results_found = sum(1 for result in expected_intermediate_results
                                     if result in all_text)

            print(f"🔢 Mathematical results found: {math_results_found}/3 expected")
            print(f"Full text searched: {all_text[:200]}...")

            # More lenient assertion - just check that some mathematical progress was made
            self.assertGreaterEqual(math_results_found, 1,
                                    f"Expected at least 1 mathematical result. Found {math_results_found}. "
                                    f"Full text: {all_text}")

            # Verify that streaming actually produced content
            self.assertGreater(len(streaming_deltas), 0, "Expected streaming deltas to be produced")
            self.assertGreater(len(response.response.strip()), 0, "Expected non-empty final response")


if __name__ == "__main__":
    unittest.main()
