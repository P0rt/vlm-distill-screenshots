---
license: apache-2.0
base_model: Qwen/Qwen2-VL-2B-Instruct
datasets:
  - rootsautomation/RICO-Screen2Words
language:
  - en
tags:
  - vision-language-model
  - screenshot-understanding
  - knowledge-distillation
  - lora
  - peft
  - qwen2-vl
library_name: peft
pipeline_tag: image-text-to-text
---

# qwen2-vl-2b-screenshots-distill

A compact **screenshot-understanding** model: a `Qwen2-VL-2B-Instruct` student
fine-tuned with **LoRA** to reproduce the screenshot descriptions of a larger
`Qwen2-VL-7B-Instruct` teacher (sequence-level knowledge distillation).

Given a UI screenshot, it produces a one-sentence summary followed by a list of
the key interface elements.

- **Teacher:** `Qwen/Qwen2-VL-7B-Instruct` (4-bit)
- **Student / base:** `Qwen/Qwen2-VL-2B-Instruct` (this repo ships a LoRA adapter)
- **Method:** response-based / sequence-level KD — the student is trained on the
  teacher's generated targets
- **Data:** [Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words)
  (RICO Android UI screenshots, CC-BY-4.0)
- **Code:** https://github.com/P0rt/vlm-distill-screenshots

> **Status: proof of concept.** Validated end-to-end on an Apple M4 Pro (MPS):
> train loss 0.80 → 0.39, and the reloaded adapter generates in the trained
> format. Full-scale training and quality/latency/memory benchmarks
> (CIDEr / ROUGE-L + LLM-as-judge, throughput, VRAM) are tracked in the repo.

## Usage

```python
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

BASE = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER = "p00rt/qwen2-vl-2b-screenshots-distill"

processor = AutoProcessor.from_pretrained(BASE, min_pixels=200704, max_pixels=401408)
model = Qwen2VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

image = Image.open("screenshot.png")
prompt = ("Describe this UI screenshot in one sentence, then list the key "
          "interface elements as a comma-separated list.")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

## Training

- LoRA (r=16, α=32, dropout=0.05) on the language-model attention projections.
- Visual tokens capped (`max_pixels = 512·28·28`) so large screenshots fit the
  context window.
- Backend: `transformers` + `peft` (`hf`), on CUDA or Apple MPS.

## Limitations

- Narrow domain: Android UI screenshots (RICO). Not a general VLM.
- Perception only — no grounding (bounding boxes) and no action/agent layer.
- Inherits teacher biases and any teacher hallucinations in the distilled targets.

## License & attribution

- Code & adapter: Apache-2.0.
- Base model `Qwen/Qwen2-VL-2B-Instruct`: see the base model's license.
- Screen2Words / RICO data: CC-BY-4.0.
