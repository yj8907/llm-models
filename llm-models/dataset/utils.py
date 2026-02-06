import os
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import list_repo_files

# -------------------------
# CONFIG
# -------------------------

DATASET = "mlfoundations/dclm-baseline-1.0"
NUM_SHARDS = 400               # ~300 GB uncompressed
WORK_DIR = Path("./work/dclm")

RAW_DIR = WORK_DIR / "raw"
JSONL_DIR = WORK_DIR / "jsonl"
MEGATRON_OUT = WORK_DIR / "megatron"

MEGATRON_PATH = Path("/home/ubuntu/projects/Megatron-LM")
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

from concurrent.futures import ThreadPoolExecutor

def download_single_shard(shard):
    """Function to handle an individual download task with flattened filenames"""
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{shard}"
    
    # Replace slashes with underscores to create the new filename
    flattened_filename = shard.replace("/", "_")
    # Define the full local path
    output_path = os.path.join(str(RAW_DIR), flattened_filename)
    
    print(f"Starting download: {shard} -> {flattened_filename}")
    
    try:
        # Use -O to specify the exact output path and filename
        run(["wget", "-c", url, "-O", output_path])
    except Exception as e:
        print(f'{url} failed: {e}')

def download_shards():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Listing dataset files...")

    try:
        repo_id = "mlfoundations/dclm-baseline-1.0"
        repo_type = "dataset"
        files = list_repo_files(repo_id, repo_type=repo_type)
    except Exception as e:
        print(f"Error accessing Hugging Face repo: {e}")
        files = []

    shards = sorted(
        f for f in files
        if f.startswith("global-") and f.endswith(".jsonl.zst")
    )
    shards = shards[260:NUM_SHARDS]
    print(f"Downloading {len(shards)} shards using multi-threading...")

    # Max_workers determines how many downloads run at once. 
    # 4 to 8 is usually a sweet spot for bandwidth without getting rate-limited.
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(download_single_shard, shards)

    print("All downloads complete.")

# -------------------------
# STEP 2: Decompress
# -------------------------

def decompress_shards():
    JSONL_DIR.mkdir(parents=True, exist_ok=True)

    for zst_file in RAW_DIR.glob("*.jsonl.zst"):
        out = JSONL_DIR / zst_file.name.replace(".zst", "")
        if out.exists():
            continue
        try:
            run(["zstd", "-d", str(zst_file), "-o", str(out)])
        except:
            print(f'{zst_file} decompression failed')

# -------------------------
# STEP 3: Megatron preprocessing
# -------------------------

def run_megatron_preprocess():
    MEGATRON_OUT.mkdir(parents=True, exist_ok=True)

    input_files = sorted(JSONL_DIR.glob("*.jsonl"))
    input_args = [str(f) for f in input_files]
    preprocess_input = JSONL_DIR / "*.jsonl"

    cmd = [
    "python", str(PREPROCESS_SCRIPT),
    "--input", str(preprocess_input),
    "--output-prefix", str(OUTPUT_PREFIX),
    "--tokenizer-type", TOKENIZER_TYPE,
    "--append-eod",
    "--workers", str(NUM_WORKERS),
    "--partitions", str(16),
    "--tokenizer-type", "TikTokenizer",
    "--tokenizer-model", "/home/ubuntu/projects/llm-models/llm-models/dataset/work/dclm/megatron/gpt-4.json",
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

    # download_shards()
    decompress_shards()
    run_megatron_preprocess()
    # upload_to_s3()
    cleanup()

    print("✅ DCLM subset processed and uploaded to S3")

if __name__ == "__main__":
    main()
