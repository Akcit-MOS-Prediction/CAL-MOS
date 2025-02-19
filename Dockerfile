# Escolhe uma imagem base da NVIDIA com CUDA e CUDNN
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu18.04

# Impedir que o apt-get fique perguntando por timezone ou afins
ENV DEBIAN_FRONTEND=noninteractive

# 1) Instalar algumas dependências (wget, bzip2) e Miniconda (opcional)
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2) Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# 3) Adicionar conda no PATH
ENV PATH="/opt/conda/bin:${PATH}"

# 4) Copiar o environment.yml (gerado no passo anterior) para dentro do container
COPY environment.yml /tmp/environment.yml

# 5) Criar o ambiente conda baseando-se no environment.yml
RUN conda env create -f /tmp/environment.yml

# 6) Ativar o ambiente default (ajuste se seu environment.yml tiver outro nome de env, ex: "name: meu_env")
# Se seu environment.yml definir "name: meu_env", então use:
ENV PATH="/opt/conda/envs/calmos/bin:${PATH}"

# 7) Copiar o código da sua aplicação para dentro do container
RUN mkdir /workspace
WORKDIR /workspace
COPY . /workspace

