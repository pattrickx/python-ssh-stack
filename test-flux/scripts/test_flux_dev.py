import os, sys
from dotenv import load_dotenv
load_dotenv()

from diffusers import FluxPipeline
import torch

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO = os.environ.get("FLUX_DEV_REPO", "black-forest-labs/FLUX.1-dev")
OUT = os.environ.get("OUTPUT_DIR", "/root/test-flux/output")
PREC = os.environ.get("FLUX_PRECISION", "fp8")
STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "28"))
GUIDE = float(os.environ.get("GUIDANCE_SCALE", "3.5"))
SEED = int(os.environ.get("SEED", "42"))

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "A cinematic photo of a red fox wearing a leather jacket, neon city background, 35mm, shallow depth of field"

torch_dtype = torch.float8_e4m3fn if PREC == "fp8" else (torch.bfloat16 if PREC == "bf16" else torch.float16)
print(f"[flux-dev] loading {REPO} (dtype={torch_dtype}) ...")
pipe = FluxPipeline.from_pretrained(
    REPO,
    token=HF_TOKEN or None,
    torch_dtype=torch_dtype,
)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload() if False else None  # mantem tudo na GPU se couber
print(f"[flux-dev] warmup + generate ({STEPS} steps, cfg={GUIDE}) ...")
generator = torch.Generator("cuda").manual_seed(SEED)
img = pipe(
    prompt=PROMPT,
    height=1024, width=1024,
    num_inference_steps=STEPS,
    guidance_scale=GUIDE,
    generator=generator,
).images[0]
os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, f"flux_dev_seed{SEED}.png")
img.save(out_path)
print(f"[flux-dev] OK -> {out_path}")
