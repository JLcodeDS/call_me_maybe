*This project has been created as part of the 42 curriculum by jludolph.* 

### Description

This project implements a high-reliability **Function Calling Assistant** designed to work with small language models, specifically **Qwen3-0.6B**. While lightweight models natively fail at structure adherence, this solution leverages token-level **Constrained Decoding** coupled with an abstract finite state machine to guarantee **100% syntactically valid JSON output** that perfectly conforms to a dynamic function schema. Given an arbitrary natural language prompt, the tool intercepts logit distributions to choose a precise function target and safely build typed parameter arguments without relying on model instruction compliance alone. 

### Architecture & Project Structure

The repository is structured dynamically to support validation, cache optimization, and isolated dependency management: 

* src/main.py: Main entry point orchestrating structural parsing, setup pipelines, loop processing, and output synchronization.
* src/load_input.py: CLI arguments engine and schema-driven input sanitizer.
* src/data_models.py: Internal structural foundations, including character-branch trie representations and dynamic validation layouts.
* src/data_preparation.py: High-performance vocabulary categorization filters and functional data cleaners.
* src/fsms.py: Character state isolation tracker managing string, delimiter, validation boundary transitions.
* src/generate_output.py: Token logic coordinator managing logit filtering boundaries, formatting wraps, and functional output injection.
* llm_sdk/: The abstract LLM interface interacting with the fundamental neural layers.

### Instructions

### Installation

Ensure you have uv installed. Synchronize dependencies and establish the virtual development layout by running: 

```bash
uv sync
```

### Execution

Run the system using the structured module pattern. By default, inputs are sourced from data/input/ and written to data/output/: 

```bash

uv run python -m src
```

You can explicitly configure override paths using the standard parameter definitions: 

```bash

uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
```

### Formatting and Linting

Static types and style baselines are managed via flake8 and mypy rule groups: 

```bash

# Standard check
make lint

# Strict check
make lint-strict
```

### Algorithm Explanation

The system achieves robust validation guarantees through a dual-phase logit manipulation architecture: 

1. **Function Target Selection (Trie Transition)**: Available function names are compiled into a character/token-level prefix tree (TokenTrie). During the model's function declaration phase, logits are forcefully masked to -inf for any token ID that does not form a valid prefix path of an active function entry, mathematical certainty of target resolution.
2. **Schema Enforcement (Pydantic FSM & Cache Filtering)**: Parameters are dynamically evaluated using an abstract state tracker (PydanticFSM) that determines valid syntactic contexts (e.g., waiting for colons, tracking inner strings, parsing numerical ranges, or detecting structural terminal boundaries like closed brackets). Token representations are mapped using an accelerated prefix filter array (BuildTokenCache), stripping unaligned branches directly from the model's neural sampling output.

### Design Decisions

* **Dynamic Pydantic Generation**: Instead of rigid structural patterns, schemas inside functions_definition.json are dynamically converted into live validation classes at runtime using pydantic.create_model, facilitating arbitrary schema complexity changes without manual updates.
* **Vocab Token Pre-Caching**: Filtering raw vocab tables over 50,000+ entries token-by-token is highly CPU intensive. We isolate categorization filters into a structural array pre-cache (BuildTokenCache), dropping the active validation inference cost to a quick hash-map resolution boundary.
* **Encapsulated Masking**: Logit mapping array transformations avoid explicit loop structures, leveraging accelerated vector lookups via numpy matrices to select proper tokens efficiently within execution limits.

### Performance Analysis

* **Accuracy**: The combination of TokenTrie route constraints and token caching yields absolute deterministic compliance (100% syntactical validation parsing rate). No target crashes occur from loose structural strings or missing closing array braces.
* **Speed Optimization**: Pre-caching structural tokens minimizes state assessment overhead. Complete target text analysis executes well within grading constraints on standard execution limits.
* **Reliability**: Malformed JSON input configurations or empty file matrices are checked and explicitly rejected before inference initialization, ensuring error handlers prevent unexpected application crashes.

### Challenges Faced

* **Whitespace Token Idiosyncrasies**: Space representations fluctuate significantly across sub-tokenizers (e.g., explicit preceding character sequences like Ġ). Resolving this required developing customized alignment filters (filter_tokens) using strict prefix evaluations to match raw JSON components flawlessly.
* **State Tracking Instability**: Standard string evaluation patterns easily fall out of sync inside inner quotes. Implementing string parity contexts within get_str_context ensured escape characters and key designations are tracked stably across evaluation steps.

### Testing Strategy

* **Automated Lint Matrices**: Strict mypy checks guarantee function parameter annotations, variable hints, and validation layers match across internal module configurations.
* **Edge Prompts Parsing**: Handled verification using prompts containing nested escape expressions (\"), numerical variants with scientific indicators (e+), empty functional targets, and extreme value lengths.

### Resources

* Trie Structure: https://www.geeksforgeeks.org/dsa/trie-insert-and-search/
* Constrained Decoding: https://www.aidancooper.co.uk/constrained-decoding/
* FSMs: https://adacomputerscience.org/concepts/machines_fsm
* Along with those, multiple youtube videos of in depth python implementations for Tries, Constrained Decoding and FSMs. 
* Regex Cheatsheet: https://www.w3schools.com/python/python_regex.asp
* AI Usage Statement: AI was leveraged to find and understand resources on Tries, FSMs and tokenization techniques. Additionally, to cross-check static typing declarations across the data modules and help with the Readme file.