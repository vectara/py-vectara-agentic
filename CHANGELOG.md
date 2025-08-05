# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-08-06

### Breaking Changes

- **Removed LATS and LLMCOMPILER agent types** - These agent types are no longer supported as LlamaIndex has deprecated the underlying `LATSAgentWorker` and `LLMCompilerAgentWorker` classes
- **Updated default model for GROQ** - using the new OpenAI openai/gpt-oss-20b.

## [0.4.0] - 2025-07-31

### Breaking Changes

- **Removed `update_func` functionality** - This feature has been removed entirely
- **Deprecated `StructuredPlanning`** - No longer supported due to LlamaIndex compatibility changes
- **Removed Fireworks LLM support** - Provider removed due to lack of support in the new LlamaIndex version
- **Deprecated token counting functionality** - This feature has been removed
- **Removed `compact_docstring` option** - This configuration option is no longer available

### Migration Guide

If you are upgrading from v0.3.x:

1. **Update_func users**: This functionality has been removed with no direct replacement
2. **StructuredPlanning users**: Please migrate to the standard Agent workflows or custom workflows
3. **Fireworks users**: Migrate to one of the supported providers: OpenAI, Anthropic, Together.AI, GROQ, Cohere, Bedrock, or Gemini
4. **Token counting users**: This functionality has been removed with no direct replacement
5. **Compact_docstring users**: Remove this option from your configuration
6. **OpenAI agent type**: use FUNCTION_CALLING instead
7. **LATS and LLMCOMPILER users**: Migrate to REACT or FUNCTION_CALLING agent types, or implement custom workflows using the new LlamaIndex Workflows framework

### Added

- Enhanced workflow support with improved error handling
- Better LlamaIndex integration

### Changed

- Updated to newer LlamaIndex version with improved stability
- Streamlined configuration options
- **Updated default models**: Gemini now defaults to gemini-2.5-flash-lite

## [0.3.x] - Previous Versions

For changes in previous versions, please refer to the git history.