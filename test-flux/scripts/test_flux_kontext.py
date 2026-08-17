import os, sys
from dotenv import load_dotenv
load_dotenv()

import torch
from diffusers import FluxKontextPipeline
from PIL import Image

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO = os.environ.get("FLUX_KONTEXT_REPO", "black-forest-labs/FLUX.1-Kontext-dev")
OUT = os.environ.get("OUTPUT_DIR", "/root/test-flux/output")
PREC = os.environ.get("FLUX_PRECISION", "fp8")
STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "28"))
SEED = int(os.environ.get("SEED", "42"))

if len(sys.argv) < 3:
    print("Uso: python scripts/test_flux_kontext.py <imagem_entrada> <prompt_edicao>")
    sys.exit(1)
IMG_PATH = sys.argv[1]
PROMPT = sys.argv[2]

if PREC == "fp8":
    try:
        torch_dtype = torch.float8_e4m3fn
    except AttributeError:
        torch_dtype = torch.bfloat16
elif PREC == "bf16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

print(f"[kontext] loading {REPO} (dtype={torch_dtype}) ...")
pipe = FluxKontextPipeline.from_pretrained(
    REPO,
    token=HF_TOKEN or None,
    torch_dtype=torch_dtype,
).to("cuda")
if PREC != "fp8":
    pipe.enable_model_cpu_offload()

init = Image.open(IMG_PATH).convert("RGB")
print(f"[kontext] edit: {PROMPT}")
generator = torch.Generator("cuda").manual_seed(SEED)
img = pipe(
    prompt=PROMPT,
    image=init,
    num_inference_steps=STEPS,
    generator=generator,
).images[0]
os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, f"flux_kontext_seed{SEED}.png")
img.save(out_path)
print(f"[kontext] OK -> {out_path}")
