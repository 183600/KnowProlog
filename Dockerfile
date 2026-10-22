# syntax=docker/dockerfile:1

FROM rust:1.99-slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev \
        build-essential autoconf automake libtool clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

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
