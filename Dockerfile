FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

# Acesso SSH como root (senha padrão python123 — troque depois)
RUN echo 'root:python123' | chpasswd
RUN sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
