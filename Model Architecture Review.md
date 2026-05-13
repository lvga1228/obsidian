---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: model-architecture-review
---

# Model Architecture Review

**分类:** 📊 Data Science
**Skill ID:** `model-architecture-review`

> Systematically explore, understand, and explain a deep learning model codebase — trace data flow, map tensor shapes, extract architecture logic, and produce concrete- example explanations suitable for mobile reading.

---


# Model Architecture Review

Systematic workflow for exploring a deep learning model codebase, extracting its architecture logic, and producing clear explanations with dimension-tracked, concrete-number examples — formatted for readability on mobile devices.

## Workflow

### 1. Discover project structure
- Run `find . -not -path '*/.mypy_cache/*' -not -path '*/__pycache__/*' -type f -name '*.py' | sort` to list all Python source files
- Read `__init__.py` files to understand module exports
- Identify the directory hierarchy: config/, data_process/, model_module/, model_build/, model_interface/, model_applicat/

### 2. Locate the base/foundational class
- Search for `base_model`, `BaseModel`, or the parent class that defines shared architecture
- Read it first — it contains the core pipeline (`create_model()`, `encoder_part()`, `decoder_part_base()`)
- Identify abstract methods vs. concrete methods
- Note the "strategy pattern" registries (dictionaries mapping string names to function blocks)

### 3. Read the concrete model classes
- Subclasses (e.g., `fork_model`, `mw_model`) override abstract methods with specific logic
- Understand what each subclass changes: input shaping, encoder output processing, decoder routing

### 4. Read algorithm blocks one by one
- **Encoder blocks**: WaveNet, TSC/SciNet, Informer — understand the inductive bias
- **CA/CT module**: How encoder output is decomposed into amplitude + trend
- **Decoder blocks**: MLP layers, hidden state concatenation
- **Cannibalization / special modules**: Business-specific logic (competition, hierarchy)

### 5. Reconstruct the data flow
- Trace from `all_input` (flattened 1D) → slice → reshape → 3D tensors
- Follow through encoder → encoder_output_process → CA/CT → decoder → output
- Note every reshape, concatenation, and RepeatVector

### 6. Explain with concrete examples
- **Pick small, round hyperparameters** for the demo (not actual production values)
- Define a concrete scenario (e.g., "predict shampoo sales for 7 days")
- Walk through every step with:
  - Tensor name (e.g., `encoder_input`)
  - Exact shape (e.g., `(1, 8, 2)`)
  - What each dimension means (e.g., "8天 × [销量, 促销]")
  - Concrete numerical values where helpful

### 7. Format for mobile readability
- Use short, labeled sections (###) rather than dense paragraphs
- Avoid wide ASCII tables — prefer step-by-step decomposition with arrows (→)
- Shape annotations should be inline: `encoder_hidden (1, 8, 16) ← 每时刻16维`
- Use emoji sparingly as visual anchors for steps
- At the end, provide a **dimension change summary table** as a quick reference

## Pitfalls
- **Don't skip the dimension explanation**: Users on mobile cannot visually parse dense tensor notation. Always spell out what each axis means.
- **Don't use production-scale parameters** for examples — tiny encoder_len=8, forecast_len=3 makes the math traceable.
- **Don't explain the code lib line-by-line** — explain the architecture logic. Code references (e.g., "see `ca_ct_block_fork()`") are secondary.
- **Read config files too**: `dl_model_config.py` contains the block registry dicts that show all available architecture choices.

## References
- `references/u-forecast-case-study.md` — Full architecture breakdown of the U_forecast demand forecasting model (TensorFlow/Keras)

## Verification
- After explanation, the user should be able to sketch the data flow from input to output with shapes
- The dimension summary table at the end serves as a self-check: every tensor in the pipeline should appear in it

