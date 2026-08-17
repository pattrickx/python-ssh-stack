import os, sys
from dotenv import load_dotenv
load_dotenv()

import torch
from diffusers import FluxPipeline

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO = os.environ.get("FLUX_DEV_REPO", "black-forest-labs/FLUX.1-dev")
OUT = os.environ.get("OUTPUT_DIR", "/root/test-flux/output")
PREC = os.environ.get("FLUX_PRECISION", "fp8")
STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "28"))
GUIDE = float(os.environ.get("GUIDANCE_SCALE", "3.5"))
SEED = int(os.environ.get("SEED", "42"))

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "A serene lake at sunset, volumetric light, ultra detailed, 35mm photography"

# fp8 so é suportado em torch>=2.6; se nao suportado, cai p/ bf16
if PREC == "fp8":
    try:
        torch_dtype = torch.float8_e4m3fn
        _ = torch_dtype  # valida existencia
    except AttributeError:
        print("[warn] torch sem suporte a fp8; usando bf16")
        torch_dtype = torch.bfloat16
elif PREC == "bf16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

print(f"[flux-dev] loading {REPO} (dtype={torch_dtype}) ...")
pipe = FluxPipeline.from_pretrained(
    REPO,
    token=HF_TOKEN or None,
    torch_dtype=torch_dtype,
).to("cuda")

if PREC != "fp8":
    # fp16/bf16 nao cabem em 12GB: manda componentes p/ CPU sob demanda
    pipe.enable_model_cpu_offload()

print(f"[flux-dev] generate ({STEPS} steps, cfg={GUIDE}) ...")
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
