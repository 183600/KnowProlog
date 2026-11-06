# syntax=docker/dockerfile:1

FROM rust:1.99-slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev \
        build-essential autoconf automake libtool clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Cargo.lock 被 .gitignore 排除、不入库（见 .github/workflows/tests.yml 中对
# 无锁文件场景的处理），因此用通配符容忍其缺失，避免全新克隆直接 docker build 失败。
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
# 存在 Cargo.lock 时用 --locked 严格校验依赖版本；否则现场解析依赖再构建
RUN if [ -f Cargo.lock ]; then cargo build --release --locked; else cargo build --release; fi

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin knowprolog

WORKDIR /app
COPY --from=builder /build/target/release/know_prolog /usr/local/bin/know_prolog
COPY config.toml ./
COPY basic_knowledge.txt additional_knowledge.txt knowledge_base.txt \
    knowledge_base_expanded.txt comprehensive_knowledge_expansion.txt ./
COPY examples ./examples
COPY plugins ./plugins

RUN mkdir -p /app/oxigraph_data \
    && chown -R knowprolog:knowprolog /app

USER knowprolog
VOLUME ["/app/oxigraph_data"]
EXPOSE 8080
ENTRYPOINT ["know_prolog"]
CMD ["interactive"]
