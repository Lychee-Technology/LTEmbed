# Dockerfile — Reproducible Amazon Linux 2023 build environment
#
# For ARM64 Lambda: run on ARM64 host or add --platform linux/arm64
#   docker build --platform linux/arm64 -t ltembed-builder .
#   docker run --rm --platform linux/arm64 -v $(pwd):/out ltembed-builder cp /workspace/dist/ltembed-lambda.zip /out/

FROM public.ecr.aws/amazonlinux/amazonlinux:2023

RUN dnf install -y gcc tar gzip openssl-devel zip && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . .

RUN cargo build --release && \
    mkdir -p dist && \
    cp target/release/bootstrap dist/bootstrap && \
    cp -r assets dist/assets && \
    (cd dist && zip -r ltembed-lambda.zip bootstrap assets/)

CMD ["cp", "dist/ltembed-lambda.zip", "/out/ltembed-lambda.zip"]
