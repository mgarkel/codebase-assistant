# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a codebase assistant powered by LLM + LangGraph + LangChain that enables Q&A interaction with codebases. The system ingests repository code, creates embeddings using vector stores, and provides three types of interactions through specialized agents: retrieve, explain, and navigate.

## Development Commands

### Poetry Environment
```bash
# Install dependencies
poetry install

# Run ingestion (build embeddings from codebase)
poetry run python main.py ingest

# Run interactive chat
poetry run python main.py chat

# Streamlit demo
cd streamlit_app && poetry run streamlit run app.py
```

### Linting and Formatting
```bash
# Format code with ruff
poetry run ruff format .

# Sort imports with isort
poetry run isort .

# Run pre-commit hooks manually
poetry run pre-commit run --all-files

# Install pre-commit hooks
poetry run pre-commit install
```

### Testing
```bash
# Run tests
poetry run pytest
```

## Architecture Overview

### Core Components

1. **Ingestion Pipeline** (`ingestion/`)
   - `chunk_code.py`: Splits code into chunks for embedding
   - `embed_chunks_into_vectorstore.py`: Creates embeddings and stores in vector database
   - `ingest_repo.py`: Main ingestion orchestrator
   - `load_vectorstore.py`: Vector store loading utilities

2. **LangGraph Flow** (`langgraph_flow/`)
   - **Agents**: Specialized agents for different interaction patterns
     - `intent_classifier.py`: Determines user intent (retrieve/explain/navigate)
     - `retriever_agent.py`: Fetches relevant code snippets
     - `explainer_agent.py`: Provides detailed code explanations
     - `navigator_agent.py`: Helps navigate and understand codebase structure
   - **Models**: State management and OpenAI model wrappers
     - `assistant_state.py`: Pydantic state schema for LangGraph flow
     - `openai_model.py`: OpenAI API integration

3. **Utilities** (`utils/`)
   - `agent_utils.py`: Common agent functionality
   - `util.py`: Main flow orchestration (ingest_flow, chat_flow)
   - `constants.py`: Project constants and enums

### Agent Flow

The system uses intent classification to route queries to appropriate specialized agents:
- **RETRIEVE**: Returns relevant code snippets without explanation
- **EXPLAIN**: Provides detailed explanations of code functionality 
- **NAVIGATE**: Helps understand codebase structure and relationships

### Configuration

Configuration is managed through `config/settings.toml` with sections for:
- OpenAI models (inference and embedding)
- Repository settings (URL, local path)
- Vector store configuration (Chroma)
- Agent-specific top_k settings

## Key Files to Understand

- `main.py`: CLI entry point with ingest/chat commands
- `langgraph_flow/agents/agent.py`: Base agent class with common inference logic
- `utils/util.py`: Core flow functions that orchestrate the entire process
- `langgraph_flow/models/assistant_state.py`: State schema passed between agents

## Development Notes

- Uses Poetry for dependency management with Python 3.11+
- Ruff configured for formatting (80 char line length) with pre-commit hooks
- Vector store currently supports Chroma with configurable persistence
- All agents inherit from base `Agent` class with standardized inference patterns
- LangGraph manages the multi-agent conversation flow and state transitions