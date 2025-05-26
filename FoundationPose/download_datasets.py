from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bop-benchmark/lm",
    repo_type="dataset",
    local_dir="./lm",
    allow_patterns=["lm_models.zip", "lm_base.zip"]
)
