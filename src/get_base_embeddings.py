import torch
import os, sys
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer

file_dir = os.path.dirname(os.path.realpath(__file__))
ARXIV_KAGGLE_DIR = os.path.join(file_dir, "arxiv_kaggle")
BATCH_SIZE = 1000

if not os.path.exists(ARXIV_KAGGLE_DIR):
    os.mkdir(ARXIV_KAGGLE_DIR)

metadata_path = os.path.join(ARXIV_KAGGLE_DIR, "arxiv-metadata-oai-snapshot.json")

# Download Kaggle snapshot if missing
if not os.path.exists(metadata_path):
    # requires authorized `kaggle` command
    res = os.system(f"kaggle datasets download -d Cornell-University/arxiv -f arxiv-metadata-oai-snapshot.json -p {ARXIV_KAGGLE_DIR} --unzip")

    if not res:
        print("Download failed")
        sys.exit(res)

    # Separate into batches for easier loading
    metadata_batch_path = os.path.join(ARXIV_KAGGLE_DIR, "meta_batches")
    os.mkdir(metadata_batch_path)

    with jsonlines.open(metadata_path, "r") as metadata:

        batch_number = -1
        end_of_file = False

        while True:
            batch_number += 1

            # Gather batch
            batch = []
            for i in range(BATCH_SIZE):
                try:
                    batch.append(metadata.read())
                except EOFError:
                    end_of_file = True
                    break

            # Write batch
            with jsonlines.open(os.path.join(metadata_batch_path, f"batch_{batch_number}_meta.jsonl"), "w") as batch_write:
                for obj in batch:
                    batch_write.write(obj)
                    
            del batch

            if end_of_file:
                break

# Get embeddings of abstracts

# Encoder
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B").to("cuda").half()

batch_number = 0

base_vectors_path = os.path.join(ARXIV_KAGGLE_DIR, "base_vectors")

while True:
    batch_path = os.path.join(base_vectors_path, f"batch_{batch_number}.npy")
    if os.path.exists(batch_path):
        batch_number += 1
        continue

    try:
        with jsonlines.open(os.path.join(base_vectors_path, f"batch_{batch_number}_meta.jsonl"), "r") as f:
            batch = [" ".join(obj["abstract"].strip().splitlines()) for obj in f]
    except FileNotFoundError as e:
        print(e)
        break

    vectors = model.encode(batch)
    torch.cuda.empty_cache()
    np.save(batch_path, vectors)
    batch_number += 1
