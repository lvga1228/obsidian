---
tags: [skill, hermes]
category: ⚙️ MLOps
skill_id: ml-training-methods
---

# ML Training Methods

**分类:** ⚙️ MLOps
**Skill ID:** `ml-training-methods`

> Advanced ML training techniques — GRPO/RL fine-tuning with TRL for reasoning and task-specific training, and PyTorch FSDP for distributed large-scale model training with parameter sharding.

---


# ML Training Methods — Umbrella Skill

Covers advanced training techniques for LLMs: GRPO (Group Relative Policy Optimization) for reinforcement-learning-based fine-tuning, and FSDP (Fully Sharded Data Parallel) for distributed large-scale training. Each has a detailed reference file.

## When to Use

Load this skill when training or fine-tuning large models — whether you need RL-based alignment (enforce formats, teach reasoning, optimize for verifiable rewards) or distributed infrastructure (multi-GPU sharding, mixed precision, CPU offloading).

## Methods

### GRPO/RL Training (`references/grpo-rl.md`)
- **Use for**: Enforcing structured output formats, teaching verifiable tasks (math/code), improving reasoning, multi-objective optimization without labeled preference data
- **Key library**: TRL (Transformer Reinforcement Learning) `GRPOTrainer`
- **Critical insight**: Loss INCREASES during training (measures KL divergence) — monitor reward metrics instead
- **Reward functions**: Compose 3-5 functions (format + correctness + style), use incremental rewards for partial credit
- **Hyperparams**: `num_generations=8-16`, `learning_rate=5e-6`, group comparison within each prompt

### PyTorch FSDP (`references/fsdp.md`)
- **Use for**: Training models too large for single GPU via parameter sharding across devices
- **Key library**: PyTorch FSDP (torch.distributed.fsdp)
- **Features**: FSDP1 (parameter/gradient/optimizer sharding), FSDP2 (per-parameter sharding), mixed precision (bf16/fp16), CPU offloading, activation checkpointing
- **Critical insight**: Use `HYBRID_SHARD` for multi-node, `FULL_SHARD` for single-node multi-GPU

## Quick Reference

```bash
# GRPO
pip install trl>=0.14.0 transformers>=4.47.0
# See references/grpo-rl.md for full training workflow

# FSDP  
pip install torch>=2.0
# See references/fsdp.md for sharding strategies and config
```

## Related Skills

- `guidance` — Constrained LLM generation (inference-time, complementary to GRPO)
- `peft-fine-tuning` — Parameter-efficient fine-tuning (LoRA/QLoRA, alternatives to full FSDP)
- `unsloth` — 2-5x faster LoRA/QLoRA fine-tuning

