# python-ssh-stack

Stack Python (3.12-slim) + OpenSSH, feito para ser implantado no Portainer
via **Git repository** (método nativo de Stack).

## Acesso

```bash
ssh -p 2222 root@<IP_DO_HOST>
# senha: python123  (TROQUE depois de subir!)
```

- Porta host: `2222` → porta container `22`
- Imagem base: `python:3.12-slim` + `openssh-server`
- Reinício automático: `unless-stopped`

## Arquivos

- `docker-compose.yml` — definição do serviço
- `Dockerfile` — imagem com Python + SSH

## Deploy no Portainer

1. Stacks → Add stack → método **Git repository**
2. Repository URL: `https://github.com/pattrickx/python-ssh-stack.git`
3. Branch: `main` | Compose path: `docker-compose.yml`
4. Authentication: nenhuma (repo público)
5. Deploy the stack
