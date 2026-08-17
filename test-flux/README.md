# test-flux

Ambiente para testar os modelos FLUX da Black Forest Labs usando a GPU (NVIDIA RTX 3060 12GB) dentro do container `python-ssh` (Portainer stack).

## Acesso

```bash
ssh -p 2222 root@192.168.0.11        # senha: python123
cd /root/test-flux
```

## Estrutura

```
test-flux/
├── requirements.txt   # dependencias (torch CUDA + diffusers)
├── .env               # tokens + config (HF_TOKEN obrigatorio)
├── venv/              # virtualenv Python 3.12
├── models/            # cache LOCAL dos pesos (HF_HOME aponta pra ca)
├── output/            # imagens geradas
├── scripts/
│   ├── test_flux_dev.py        # gera imagem texto->imagem (FLUX.1-dev)
│   └── test_flux_kontext.py    # edita imagem (FLUX.1-Kontext-dev)
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Rodar 100% LOCAL (pesos baixados, sem API do HuggingFace)

Os pesos são baixados **uma vez** e cacheados localmente em `models/` (via `HF_HOME`
e `HF_HUB_CACHE` no `.env`). Depois do download, a inferência roda inteiramente no
container, sem chamar nenhuma API do HF.

1. Aceite a licenca (gated) em https://huggingface.co/black-forest-labs/FLUX.1-dev
   e em https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
2. Crie um token HF (read) em https://huggingface.co/settings/tokens
3. Edite `.env`: `HF_TOKEN=...` e confirme `HF_HOME=/root/test-flux/models`
4. Na primeira execucao os pesos sao baixados e salvos em `models/hub/`.
   Execucoes seguintes usam o cache local (offline, `HF_HUB_OFFLINE=1` opcional).

```bash
source venv/bin/activate
python scripts/test_flux_dev.py        # baixa (1x) + gera output/flux_dev_<seed>.png
python scripts/test_flux_kontext.py img.png "troque o fundo para deserto"
```

## Notas de VRAM (RTX 3060 = 12GB)

- FLUX.1-dev em fp16 nao cabe em 12GB sozinho.
- Use `FLUX_PRECISION=fp8` no `.env` (carrega o transformer em fp8 via `transformer=FluxTransformer2DModel.from_pretrained(..., torch_dtype=torch.float8_e4m3fn)`).
- Se ainda estourar, rode com `pipe.enable_model_cpu_offload()` ou quantização NF4 (`bitsandbytes`).
- `shm_size: 8gb` ja esta configurado no docker-compose da stack.

## GPU passthrough

O container herda a GPU do host via `nvidia-container-toolkit` (runtime default nvidia no TrueNAS).
Verifique com: `nvidia-smi` dentro do container.
