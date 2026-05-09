---
category: "mlops"
skill_id: "openai-models"
display_name: "OpenAI模型"
---

## Skill Content

# OpenAI Specialized Models — Umbrella Skill

Covers OpenAI's two most popular specialized models: CLIP (vision-language) and Whisper (speech recognition). Each has a detailed reference file with API usage, model sizes, best practices, and performance benchmarks.

## When to Use

Load this skill when the task involves image understanding (classification, search, moderation) or audio/speech processing (transcription, translation).

## Models

### CLIP — Contrastive Language-Image Pre-Training (`references/clip.md`)
- **Use for**: Zero-shot image classification, image-text similarity, semantic image search, content moderation
- **Model sizes**: RN50 (102M) → ViT-B/32 (151M) → ViT-L/14 (428M)
- **GitHub**: 25,300+ stars | **License**: MIT
- **Key insight**: Matches ResNet-50 on ImageNet with zero-shot — no training data needed

### Whisper — Robust Speech Recognition (`references/whisper.md`)
- **Use for**: Speech-to-text (99 languages), translation to English, podcast transcription
- **Model sizes**: tiny (39M) → turbo (809M) → large (1.55B)
- **GitHub**: 72,900+ stars | **License**: MIT
- **Key insight**: Trained on 680K hours of audio; `turbo` model gives best speed/quality balance

## Quick Reference

```bash
# CLIP — zero-shot classification
pip install git+https://github.com/openai/CLIP.git
# See references/clip.md for full usage

# Whisper — transcription
pip install -U openai-whisper
whisper audio.mp3 --model turbo
# See references/whisper.md for full usage
```

## Alternatives

- Instead of CLIP: BLIP-2 (better captioning), LLaVA (vision chat), SAM (segmentation)
- Instead of Whisper: AssemblyAI (managed API + diarization), Deepgram (real-time streaming)