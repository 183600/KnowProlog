# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KnowProlog is a Chinese knowledge graph system built in Rust that integrates natural language processing, RDF storage, and Prolog reasoning engine. The system extracts knowledge from Chinese text and performs logical reasoning.

## Development Commands

### Build and Run
```bash
# Build the project
cargo build --release

# Run the application
cargo run --release

# Run tests
cargo test

# Run specific test
cargo test test_name
```

### Data Setup
```bash
# Create Oxigraph data directory (required before first run)
mkdir -p oxigraph_data
```

## System Architecture

The system follows a modular pipeline architecture:

1. **NLP Processing Layer** (`src/nlp.rs`)
   - Chinese text segmentation using jieba-rs
   - Part-of-speech tagging with custom dictionary
   - Synonym normalization
   - Query intent analysis with pattern matching

2. **Knowledge Extraction Layer** (`src/lib.rs`)
   - Pattern-based entity and relationship extraction
   - Configurable extraction rules via TOML
   - Multi-pattern matching for complex knowledge structures

3. **Storage Layer** (`src/lib.rs` - OxigraphManager)
   - RDF triple storage using Oxigraph
   - SPARQL query interface
   - Automatic fact loading for Prolog engine

4. **Reasoning Layer** (`src/prolog_engine.rs`)
   - Scryer Prolog engine integration
   - Dynamic fact loading from RDF store
   - Query generation and result processing

5. **Main Application** (`src/main.rs`)
   - Interactive command-line interface
   - Input mode detection (import vs query)
   - Response formatting and user interaction

## Key Configuration

The system is configured via `config.toml`:

- **System settings**: Oxigraph store path, endpoints, language settings
- **NLP configuration**: Custom POS dictionary, name patterns, import keywords
- **Entity types**: Configurable entity schemas with properties
- **Extraction rules**: Pattern-based knowledge extraction rules
- **Prolog rules**: Query intent patterns and corresponding Prolog queries
- **Synonyms**: Word normalization mappings
- **Messages**: User interface text in Chinese

## Important Implementation Details

### Chinese Text Processing
- Uses jieba-rs for segmentation with custom POS dictionary
- Implements name recognition using configurable patterns
- Handles synonym normalization before processing

### RDF Storage Schema
- Namespace: `http://example.org/kg/`
- Entity URIs: `{BASE_NS}entity/{name}`
- Property URIs: `{BASE_NS}{property_name}`
- Type assertions use `rdf:type`

### Prolog Integration
- Facts loaded dynamically from RDF store
- Custom predicates: `oxigraph_entity/3`, `oxigraph_relation/3`
- Query generation based on intent patterns
- Result extraction currently has limitations (see warnings in code)

### Error Handling
- Comprehensive error handling in async operations
- Graceful degradation for unrecognized inputs
- Detailed error messages for debugging

## Testing

The test suite (`tests/integration_test.rs`) covers:
- Configuration loading and validation
- NLP preprocessing and synonym normalization
- Knowledge extraction from text
- Oxigraph storage and retrieval
- Prolog engine query execution
- End-to-end workflow testing
- Concurrent access patterns
- Error handling scenarios

Tests use temporary directories and create test configurations automatically.

## Known Limitations

1. **Prolog Result Extraction**: The current Scryer Prolog integration has limitations in extracting variable bindings programmatically (see warnings in `prolog_engine.rs:106-118`)

2. **Chinese Name Recognition**: Limited to configured patterns and may miss complex name variations

3. **Pattern Matching**: Extraction rules are rigid and may not handle all linguistic variations

## Development Notes

- The system uses async/await extensively for I/O operations
- All configuration is loaded from TOML files - avoid hardcoding
- The codebase is structured as a library (`src/lib.rs`) with a main application
- Error handling uses `Box<dyn std::error::Error>` for flexibility
- Chinese text processing requires careful handling of UTF-8 encoding