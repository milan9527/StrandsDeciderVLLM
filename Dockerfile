FROM vllm/vllm-openai:v0.31.0
WORKDIR /opt/decider-vllm
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=1
EXPOSE 8100
ENTRYPOINT ["decider-vllm"]
CMD ["serve", "--model", "/models/decider-vllm", "--host", "0.0.0.0", "--port", "8100"]
