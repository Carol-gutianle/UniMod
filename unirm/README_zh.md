# UniRM：用于多模态内容审核的多头标量奖励模型

**UniRM** 是一个用于多模态内容审核的**多头标量奖励模型**，能够提供**可解释的、属性级别的评分信号**。  
该模型被设计用于支持 **UniMod** 中的开放式推理策略优化，尤其适用于**缺乏确定性标签的响应生成阶段**。

UniRM 将奖励信号解耦为多个维度，使模型能够区分**表达质量**与**安全边界**（隐私、偏见、有害性、合法性），从而实现更透明的诊断与更稳定的训练。

---

## 演示视频

> UniRM 演示视频：

<video controls preload="metadata" style="width:100%; max-width:900px; border-radius:12px;">
  <source src="static/unirm.mp4" type="video/mp4">
</video>

---

## 快速开始（Gradio）

下面给出一个最小化的 Gradio 示例，用于加载 **UniRM**，并对 *(输入指令、模型回复、可选图像)* 进行**多头奖励评分**。

```bash
git clone https://github.com/TideDra/lmm-r1.git
cd lmm-r1
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
python unirm.py --model_path {PATH_TO_UNIRM}
