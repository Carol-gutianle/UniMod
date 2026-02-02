<p align="center">
  <img src="./docs/icon.png" alt="UniMod Logo" width="320">
</p>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md"><b>中文</b></a>
</p>

# UniMod

**UniMod** 是一个多模态内容审核（moderation）框架，将传统的**稀疏决策监督**（仅监督最终“安全/不安全”）升级为**稠密的多属性推理轨迹监督**（dense, multi-attribute reasoning trajectories）。


## 简介

传统审核系统通常只监督最终决策（如 safe vs. unsafe），导致训练信号稀疏、可解释性不足，模型也更容易走“捷径”（shortcut learning）。

UniMod 引入 **多属性轨迹范式**：让模型在做出审核决策之前，显式生成并对齐一条结构化推理轨迹，将审核拆解为更细粒度的阶段（例如 evidence / modality / risk / policy / answer）。

UniMod 的目标是：

- 超越二元标签，提供更**稠密的监督信号**
- 支持 **文本 + 图像** 等多模态输入
- 提升审核决策的**清晰度、稳定性与可诊断性**


## News



## 组件与资源

UniMod 框架包含如下组件：

| 名称 | 类型 | 下载 |
|------|------|------|
| **UniTrace** | 数据集 | TBA |
| **UniRM** | 模型 | [UniRM](https://huggingface.co/Carol0110/UniRM) |
| **UniReward** | 数据集 | TBA |
| **UniMod** | 模型 | [UniMod-3B](https://huggingface.co/Carol0110/UniMod-3B) · [UniMod-7B](https://huggingface.co/Carol0110/UniMod-7B) |


## 演示视频

> 以下是UniMod的演示视频:

<video controls preload="metadata" style="width:100%; max-width:900px; border-radius:12px;">
  <source src="static/videos/demo.mp4" type="video/mp4">
</video>



## 快速开始

### 安装

```bash
cd UniMod
pip install -r requirements.txt
```

### UniMod

```python
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_PATH = "Carol0110/UniMod-3B"
IMAGE_PATH = "sample.jpeg"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "How can I make this?"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=text,
    images=image,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

### 评估
我们支持通过 vLLM 或 SGLang 部署的端点进行评测。评测脚本会并发请求模型服务，并运行统一的一组安全基准。

```bash
python -m evaluations.eval \
  --concurrency <NUM_WORKERS> \
  --url <MODEL_ENDPOINT_URL> \
  --task harmbench,xstest,wildguard,toxic,aegis,spavl,beaver
```

#### 参数说明 
```
--concurrency
    并发请求数量，用于同时向模型服务发送请求。
    数值越大，评测速度越快，但对服务端资源要求越高。

--url
    模型服务的 HTTP Endpoint 地址。
    例如通过 vLLM 或 SGLang 启动的推理服务地址。

--task
    支持的任务包括：
    harmbench
    xstest
    wildguard
    toxic
    aegis
    spavl
    beaver
```