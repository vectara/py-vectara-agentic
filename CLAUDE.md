# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Linting and Type Checking
- `make lint` - Run pylint, flake8, and codespell on the codebase
- `make mypy` - Run mypy type checking
- `make all` - Run lint, mypy, and tests together

### Testing
- `make test` - Run all tests using unittest discovery
- `python tests/run_tests.py` - Run tests with custom runner (suppresses Pydantic warnings)
- `python tests/run_tests.py test_specific.py` - Run specific test file
- `python tests/run_tests.py --verbose` - Run tests with verbose output
- `python tests/run_tests.py "test_agent*"` - Run tests matching pattern

### Documentation
- `mkdocs serve` - Serve documentation locally for development
- `mkdocs build` - Build documentation

## Code Architecture

### Core Components

**Agent System (vectara_agentic/agent.py)**
- Main `Agent` class that orchestrates AI assistants using LlamaIndex agent framework
- Supports multiple agent types: ReAct, Function Calling
- Built around tools, custom instructions, and configurable LLM providers
- Handles both sync/async chat interactions and streaming responses

**Configuration (vectara_agentic/agent_config.py)**
- `AgentConfig` dataclass centralizes all configuration
- Supports multiple LLM providers: OpenAI, Anthropic, Together.AI, GROQ, Cohere, Bedrock, Gemini, Private
- Configurable agent types and model names for both main agent and tools

**Tools System (vectara_agentic/tools.py)**
- `VectaraToolFactory` creates RAG and search tools for Vectara corpora
- `ToolsFactory` creates general-purpose tools from Python functions
- Extensive integration with LlamaIndex ToolSpecs for external services
- Support for custom tool creation with VHC (Vectara Hallucination Correction) eligibility

**Tools Catalog (vectara_agentic/tools_catalog.py)**
- Pre-built domain-specific tools (legal, financial, database)
- Standard utility tools (summarize_text, rephrase_text)
- Integration with external services (Yahoo Finance, etc.)

**Agent Core (vectara_agentic/agent_core/)**
- `factory.py` - Agent factory for different agent types
- `prompts.py` - General instructions and prompt templates
- `streaming.py` - Streaming response handlers
- `serialization.py` - Agent state serialization/deserialization
- `utils/` - Shared utilities for hallucination detection, logging, prompt formatting

### Key Design Patterns

**Tool Creation**
- Tools are created via factories (`VectaraToolFactory`, `ToolsFactory`)
- RAG tools automatically include `query` parameter plus custom schema fields
- Support for metadata filtering using Pydantic schemas
- VHC eligibility controls whether tools participate in hallucination correction

**Agent Workflows**
- Built-in workflows: `SubQuestionQueryWorkflow`, `SequentialSubQuestionsWorkflow`
- Custom workflows supported via LlamaIndex Workflow framework
- Workflows handle complex multi-step interactions with input/output validation

**LLM Provider Abstraction**
- Unified interface across multiple LLM providers
- Separate configuration for main agent LLM vs tool LLM
- Support for private/self-hosted LLM endpoints

### Dependencies and Environment

**Python Requirements**
- Python ≥3.10 required
- Core dependencies: LlamaIndex, Pydantic, Vectara SDK
- Development tools: pylint, flake8, mypy, codespell, black

**Environment Variables**
- `VECTARA_API_KEY` - Vectara API credentials
- `VECTARA_CORPUS_KEY` - Corpus identifier(s), comma-separated for multiple
- Provider-specific API keys (OpenAI, Anthropic, etc.)
- AWS credentials for Bedrock (AWS_PROFILE, AWS_REGION)

### Testing Strategy
- Unit tests for all major components in `tests/` directory
- Custom test runner suppresses Pydantic deprecation warnings
- Tests cover different agent types, LLM providers, and tool integrations
- Mock endpoints for testing without external API calls