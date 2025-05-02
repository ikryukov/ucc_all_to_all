FROM nvcr.io/nvidia/nvhpc:25.1-devel-cuda12.6-ubuntu24.04

ARG USERNAME=ikryukov
ARG USER_UID=1000
ARG USER_GID=1000

ENV CUDA_HOME=/usr/local/cuda

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends sudo locales \
    && if getent passwd $USER_UID > /dev/null; then \
        # If the UID already exists, rename the user to our desired username if it's not already taken \
        OLD_USER=$(getent passwd $USER_UID | cut -d: -f1); \
        if [ "$OLD_USER" != "$USERNAME" ]; then \
            if id $USERNAME > /dev/null 2>&1; then userdel $USERNAME; fi; \
            usermod -l $USERNAME -d /home/$USERNAME -m $OLD_USER; \
        fi; \
    else \
        # UID doesn't exist, create the user normally \
        groupadd --gid $USER_GID $USERNAME 2>/dev/null || groupmod -o -g $USER_GID $(getent group $USER_GID | cut -d: -f1 || echo nogroup); \
        useradd -l -u $USER_UID -g $USER_GID -m $USERNAME; \
    fi \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    libtool \
    numactl \
    libnuma-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install HPC-X
RUN rm -rf /opt/hpcx/ && cd /tmp && \
    wget -q https://content.mellanox.com/hpc/hpc-x/v2.22.1rc4/hpcx-v2.22.1-gcc-inbox-ubuntu22.04-cuda12-x86_64.tbz && \
    tar xf hpcx-v2.22.1-gcc-inbox-ubuntu22.04-cuda12-x86_64.tbz  && \
    mv hpcx-v2.22.1-gcc-inbox-ubuntu22.04-cuda12-x86_64  /opt/hpcx && \
    rm -f hpcx-v2.22.1-gcc-inbox-ubuntu22.04-cuda12-x86_64.tbz

# RUN sudo apt install -y libibverbs-dev rdma-core
USER $USERNAME
