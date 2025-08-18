# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-assistant customer support system for automotive diagnostic equipment. The system uses OpenAI's Assistant API with specialized AI assistants that work together to answer customer queries about technical equipment, tools, cables, compatibility, and diagnostics.

## Architecture

The system follows a multi-agent orchestration pattern:

1. **Orchestrator Agent** - Routes queries to appropriate specialist assistants
2. **Specialist Assistants** - Handle domain-specific queries:
   - `equipment` - Equipment specifications and troubleshooting
   - `diagnostics` - Testing procedures and diagnostic methods
   - `compatibility` - Vehicle compatibility and OEM parts
   - `tools` - Hand tools and accessories
   - `cables` - Cable connections and wiring
   - `support` - General support and FAQs
3. **Combinator Agent** - Combines multiple specialist responses into final answer
4. **Telegram Bot** - Provides user interface

## Key Files

### Core System Files
- `config.py` - Assistant configurations and IDs
- `orchestrator_for_tg.py` - Main orchestration logic for Telegram integration
- `customer_support_orchestrator.py` - Query analysis and routing logic
- `response_processing_utils.py` - Text processing and image marker handling

### Bot Implementation
- `tg_bot.py` - Async Telegram bot implementation
- `main.py` - Legacy bot implementation (synchronous)

### Utilities
- `validators.py` - Response validation schemas
- `prompts/` - Contains orchestrator and combinator prompts

## Common Development Commands

### Running the Application
```bash
# Run the async Telegram bot (recommended)
python tg_bot.py

# Run the legacy bot
python main.py

# Run orchestrator tests
python orchestrator.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with:
```
OPENAI_TOKEN=your_openai_api_key
TELEGRAM_TOKEN=your_telegram_bot_token
```

## Data Structure

The system processes customer queries about automotive diagnostic equipment. Key data directories:

- `files_clean/` - Original documentation files (DOCX, Excel)
- `files_preproc/` - Processed text files and extracted images
- `files_raw/` - Raw source files

Each specialist assistant has access to specific file sets mapped in `response_processing_utils.py`.

## Development Notes

### Session Management
- Sessions are created per Telegram user: `tg-{user_id}`
- Context is maintained across conversations
- Thread management handles OpenAI Assistant API interactions

### Response Processing
- Image markers like `@D_N_IMG_NNN` are processed and replaced with figure references
- Sources are extracted from OpenAI file citations
- Responses support both single and multi-image attachments

### Error Handling
- All API calls include timeout and retry logic
- Failed specialist responses are handled gracefully
- Debug mode provides detailed execution flow information

### Testing Individual Components
```bash
# Test orchestrator routing
python customer_support_orchestrator.py

# Test preprocessing utilities
python preprocessing/preprocessing_pdf.py
```

## Assistant Configuration

Assistant IDs and configurations are centralized in `config.py`. Each assistant has:
- Unique OpenAI Assistant ID
- Purpose description
- Token limits and truncation strategy
- File access permissions

When adding new assistants, update both `config.py` and the routing logic in the orchestrator prompts.