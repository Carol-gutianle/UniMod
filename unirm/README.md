<p align="center">
  <a href="./README.md"><b>English</b></a> | <a href="./README_zh.md">中文</a>
</p>

# UniRM: Multi-Head Scalar Reward Model for Multimodal Moderation

**UniRM** is a **multi-head scalar reward model** that provides **interpretable, attribute-level scoring** for multimodal moderation.  
It is designed to support policy optimization for open-ended reasoning in **UniMod**, especially for the posterior response stage where deterministic labels are absent.

UniRM decouples reward attribution into multiple dimensions so the model can distinguish **stylistic quality** from **safety boundaries** (privacy, bias, toxicity, legality), enabling transparent diagnosis and stable optimization.

---

## Demo Video

> UniRM demo video:

<video controls preload="metadata" style="width:100%; max-width:900px; border-radius:12px;">
  <source src="static/videos/unirm.mp4" type="video/mp4">
</video>

---

## Quick Start (Gradio)

Below is a minimal Gradio demo that loads **UniRM** and returns **multi-head scores** for a *(prompt, response, optional image)* triple.

```bash
git clone https://github.com/TideDra/lmm-r1.git
cd lmm-r1
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
python unirm.py --model_path {PATH_TO_UNIRM}
