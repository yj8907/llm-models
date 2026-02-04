import os
import subprocess
import shutil
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------

DATASET = "mlfoundations/dclm-baseline-1.0"
NUM_SHARDS = 400               # ~300 GB uncompressed
WORK_DIR = Path("./work/dclm")

RAW_DIR = WORK_DIR / "raw"
JSONL_DIR = WORK_DIR / "jsonl"
MEGATRON_OUT = WORK_DIR / "megatron"

MEGATRON_PATH = Path("/path/to/Megatron-LM")
PREPROCESS_SCRIPT = MEGATRON_PATH / "tools/preprocess_data.py"

# Tokenizer (example: GPT-2)
TOKENIZER_TYPE = "TikTokenizer"

OUTPUT_PREFIX = MEGATRON_OUT / "dclm_300gb"

NUM_WORKERS = 32

S3_BUCKET = "s3://my-bucket/datasets/dclm_300gb"

# -------------------------
# HELPERS
# -------------------------

def run(cmd, **kwargs):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)

# -------------------------
# STEP 1: Download shards
# -------------------------

def download_shards():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Listing dataset files...")
    files = subprocess.check_output(
        ["huggingface-cli", "repo", "ls", DATASET],
        text=True
    ).splitlines()

    shards = sorted(
        f for f in files
        if f.startswith("train-") and f.endswith(".jsonl.zst")
    )[:NUM_SHARDS]

    print(f"Downloading {len(shards)} shards")

    for shard in shards:
        url = (
            "https://huggingface.co/datasets/"
            f"{DATASET}/resolve/main/{shard}"
        )
        run(["wget", "-c", url, "-P", str(RAW_DIR)])

# -------------------------
# STEP 2: Decompress
# -------------------------

def decompress_shards():
    JSONL_DIR.mkdir(parents=True, exist_ok=True)

    for zst_file in RAW_DIR.glob("*.jsonl.zst"):
        out = JSONL_DIR / zst_file.name.replace(".zst", "")
        if out.exists():
            continue
        run(["zstd", "-d", str(zst_file), "-o", str(out)])

# -------------------------
# STEP 3: Megatron preprocessing
# -------------------------

def run_megatron_preprocess():
    MEGATRON_OUT.mkdir(parents=True, exist_ok=True)

    input_files = sorted(JSONL_DIR.glob("*.jsonl"))
    input_args = [str(f) for f in input_files]

    cmd = [
        "python", str(PREPROCESS_SCRIPT),
        "--input", *input_args,
        "--output-prefix", str(OUTPUT_PREFIX),
        "--tokenizer-type", TOKENIZER_TYPE,
        "--append-eod",
        "--workers", str(NUM_WORKERS),
    ]

    run(cmd)

# -------------------------
# STEP 4: Upload to S3
# -------------------------

def upload_to_s3():
    run([
        "aws", "s3", "sync",
        str(MEGATRON_OUT),
        S3_BUCKET,
        "--only-show-errors"
    ])

# -------------------------
# STEP 5: Cleanup
# -------------------------

def cleanup():
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(JSONL_DIR, ignore_errors=True)

# -------------------------
# MAIN
# -------------------------

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    download_shards()
    decompress_shards()
    run_megatron_preprocess()
    upload_to_s3()
    cleanup()

    print("✅ DCLM subset processed and uploaded to S3")

if __name__ == "__main__":
    main()
