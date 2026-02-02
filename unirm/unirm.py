import os
import tempfile
import argparse
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import json, os
from typing import Optional, Dict
from openrlhf.models.lmm_kits.qwen2_5_vl.patch import Qwen2_5_VLPatch
from openrlhf.models.lmm_kits.base.data_processor import MMInputs
from openrlhf.models.lmm_kits.qwen2_5_vl.data_processor import Qwen2_5_VLDataProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


class UniRM(nn.Module):
    def __init__(self, base_model, head_names, head_activation="sigmoid"):
        super().__init__()
        self.config = base_model.config
        self.base_model = base_model
        if hasattr(base_model, "lm_head"):
            try:
                del base_model.lm_head
                print("[UniRM] Removed lm_head from base_model.")
            except Exception:
                for p in base_model.lm_head.parameters():
                    p.requires_grad = False
                self.base_model.lm_head = None
                print("[UniRM] Froze lm_head parameters instead of deletion.")

        if hasattr(base_model, "model") and hasattr(base_model.model, "lm_head"):
            del base_model.model.lm_head
            print("[UniRM] Removed nested lm_head from base_model.model.")

        self.config.mgrm_heads = head_names
        self.config.mgrm_head_activation = head_activation
        self.config.model_type = getattr(self.config, "model_type", "mgrm_vlm")

        hidden_size = base_model.config.hidden_size
        dtype = next(base_model.parameters()).dtype

        if head_activation == "sigmoid":
            activation = nn.Sigmoid()
        elif head_activation == "tanh":
            activation = nn.Tanh()
        elif head_activation == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {head_activation}")

        self.value_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_size, 1, bias=False, dtype=dtype),
                activation
            )
            for name in head_names
        })

        print(f"[UniRM] ✅ Initialized Multi-Head Reward Model with heads: {head_names}")
        print(f"[UniRM] 🔧 Activation: {head_activation} | Hidden size: {hidden_size}")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_inputs: Optional[MMInputs] = None,
        return_output=False,
    ) -> Dict[str, torch.Tensor]:
        if visual_inputs is None:
            class _Empty:
                emb_inputs = {}
                forward_inputs = {}
            visual_inputs = _Empty()

        inputs_embeds = self.base_model.get_inputs_embeds(input_ids, **visual_inputs.emb_inputs)
        position_ids = self.base_model.get_position_ids(input_ids, attention_mask=attention_mask, **visual_inputs.emb_inputs)
        outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False, 
            **visual_inputs.forward_inputs,
        )

        hidden = outputs["hidden_states"][-1]
        eos_idx = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
        eos_hidden = hidden.gather(
            dim=1,
            index=eos_idx.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        ).squeeze(1)

        rewards = {name: head(eos_hidden).squeeze(-1) for name, head in self.value_heads.items()}
        return (rewards, outputs) if return_output else rewards


    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        base_model_dir = os.path.join(save_directory, "base_model")
        os.makedirs(base_model_dir, exist_ok=True)

        if hasattr(self._base_model, "save_pretrained"):
            self._base_model.save_pretrained(base_model_dir, **kwargs)
        else:
            torch.save(self._base_model.state_dict(), os.path.join(base_model_dir, "base_model.pt"))

        value_head_path = os.path.join(save_directory, "value_heads.pt")
        torch.save({k: v.cpu() for k, v in self.value_heads.state_dict().items()}, value_head_path)

        cfg = self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)
        cfg.update({
            "model_type": "mgrm_vlm",
            "head_names": list(self.value_heads.keys()),
            "attn_implementation": "eager"
        })
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"✅ UniRM saved to {save_directory}")
        
    @staticmethod
    def is_backend_compatible() -> bool:
        return True
    

    @classmethod
    def from_pretrained(cls, load_directory, torch_dtype=torch.bfloat16):
        from openrlhf.models.lmm_kits.qwen2_5_vl.patch import Qwen2_5_VLPatch
        Qwen2_5_VLPatch._load_all_patches()

        cfg_path = os.path.join(load_directory, "config.json")
        mgrm_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(load_directory, mgrm_cfg.get("base_model_path", "base_model")),
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )

        head_names = mgrm_cfg.get("mgrm_heads", mgrm_cfg.get("head_names", []))
        model = cls(base_model, head_names).to("cuda")

        vh_path = os.path.join(load_directory, "value_heads.pt")
        if os.path.exists(vh_path):
            print(f"💡 Loading structured value_heads from {vh_path} ...")
            vh_state = torch.load(vh_path, map_location="cuda")
            for name, head in model.value_heads.items():
                if name in vh_state:
                    try:
                        head.load_state_dict(vh_state[name], strict=False)
                        print(f"  ✅ Loaded head: {name}")
                    except Exception as e:
                        print(f"  ⚠️ Failed to load head {name}: {e}")
                else:
                    print(f"  ⚠️ Missing head in saved file: {name}")
        else:
            print("[Info] No value_heads.pt found.")

        print(f"✅ UniRM fully loaded from {load_directory}")
        return model

    @classmethod
    def _from_config(cls, config, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration
        config._attn_implementation_internal = "eager"
        base_model = Qwen2_5_VLForConditionalGeneration(config)
        head_names = getattr(config, "mgrm_heads", [])
        return cls(base_model, head_names)


EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for multimodal generation models.
Your task is to assess the quality of the response based on the given prompt.

### Dimension:
{dimension}

### Prompt:
{prompt}

### Response:
{response}
""".strip()



class UniRMProxy:
    def __init__(self, args):
        Qwen2_5_VLPatch._load_all_patches()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if args.bf16 else torch.float32
        self.head_names = args.head_names.split(",")

        try:
            self.model = UniRM.from_pretrained(
                args.model_path,
                torch_dtype=self.dtype,
            )
        except Exception as e:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            self.model = UniRM(base_model, self.head_names)

        self.model = self.model.to(self.device).eval()
        self.model.value_heads = self.model.value_heads.to(
            next(self.model.base_model.parameters()).dtype
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        base_processor = AutoProcessor.from_pretrained(
            os.path.join(args.model_path, "base_model"),
            trust_remote_code=True,
        )
        self.processor = Qwen2_5_VLDataProcessor(
            processor=base_processor,
            processor_kwargs=args.processor_kwargs,
        )

        self.max_length = args.max_len
        self.batch_size = args.batch_size


    @torch.no_grad()
    def score(self, prompt, response, image, dimension):
        if not prompt or not response:
            return "❌ Prompt and Response are required.", []

        messages = [{
            "role": "user",
            "content": []
        }]


        if image is not None:
            if isinstance(image, Image.Image):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image.save(tmp.name)
                img_payload = tmp.name
            else:
                img_payload = image
            messages[0]["content"].append({"type": "image", "image": img_payload})
            text = EVALUATION_PROMPT_TEMPLATE.format(
                dimension=dimension,
                prompt=prompt,
                response=response,
            )
            messages[0]["content"].append({"type": "text", "text": text})

        mm_inputs = self.processor(
            [json.dumps(messages, indent=2)],
            max_length=self.max_length,
            padding=True,
            device=self.device,
            return_tensors="pt",
        )

        outputs, _ = self.model(
            input_ids=mm_inputs.extra_info["input_ids"],
            attention_mask=mm_inputs.extra_info["attention_mask"],
            visual_inputs=mm_inputs,
            return_output=True,
        )

        rows = []
        pretty = []
        for h in self.head_names:
            v = outputs[h].detach().cpu().float().item()
            rows.append([h, v])
            pretty.append(f"{h}: {v:.6f}")

        return "\n".join(pretty), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--head_names", type=str, default="style,privacy,bias,toxicity,legality")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--processor_kwargs",
        type=json.loads,
        default={"min_pixels": 4 * 28 * 28, "max_pixels": 16384 * 28 * 28},
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    proxy = UniRMProxy(args)

    with gr.Blocks(title="UniRM Scoring") as demo:
        gr.Markdown("# 🧠 UniRM – Multimodal Reward Model")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=4)
                response = gr.Textbox(label="Response", lines=6)
                dimension = gr.Textbox(label="Dimension", value="general")
                image = gr.Image(type="pil", label="Image (optional)")
                btn = gr.Button("Score")

            with gr.Column():
                text_out = gr.Textbox(label="Scores", lines=6)
                table_out = gr.Dataframe(headers=["Head", "Score"], interactive=False)

        btn.click(
            proxy.score,
            inputs=[prompt, response, image, dimension],
            outputs=[text_out, table_out],
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
