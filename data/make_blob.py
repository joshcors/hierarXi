"""
Script for making blob and index file for arXiv abstracts and titles
"""
import re
import os
import jsonlines
import numpy as np
from pathlib import Path



def build_blobs(meta_dir, blob_dir, fields, _id, encoding="utf-8", test_limit=None):
    """
    build BLOB objects from metadata (dir with batches) at `fields`
    """
    offsets = {field : [0, ] for field in fields}

    binary_paths = {field : os.path.join(blob_dir, f"{_id}_{field}.bin") for field in fields}
    index_paths  = {field : os.path.join(blob_dir, f"{_id}_{field}.idx") for field in fields}

    Path(blob_dir).mkdir(parents=True, exist_ok=True)

    # Extract and sort batch files
    meta_files = os.listdir(meta_dir)

    file_pattern = "batch_([0-9]+)_meta.jsonl"
    meta_files = list(filter(lambda x : re.match(file_pattern, x), meta_files))
    meta_files = sorted(meta_files, key=lambda x : int(re.match(file_pattern, x).group(1)))
    meta_files = [os.path.join(meta_dir, file) for file in meta_files]

    if test_limit is not None:
        meta_files = meta_files[:test_limit]

    bin_open_files = {field: open(path, "ab") for field, path in binary_paths.items()}

    try:
        # Only want to read each file once, thus outermost
        for meta_file in meta_files:
            with jsonlines.open(meta_file, "r") as metadata:
                for meta_obj in metadata:
                    for field in fields:
                        # Get bin, append, and store offset
                        b = meta_obj[field].encode(encoding)
                        bin_open_files[field].write(b)
                        offsets[field].append(offsets[field][-1] + len(b))
    except Exception as e:
        print(e)
    finally:
        for open_file in bin_open_files.values():
            open_file.close()

    # Save offsets
    for field in fields:
        np.array(offsets[field], dtype=np.uint64).tofile(index_paths[field])

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--metadir", type=str, required=True)
    parser.add_argument("--blobdir", type=str, required=True)
    parser.add_argument("--fields", type=str, nargs="+", required=True)

    parser.add_argument("--id", type=str, default="arxiv")
    parser.add_argument("--testlimit", type=int, default=None)

    args = parser.parse_args()
    
    meta_dir = args.metadir
    blob_dir = args.blobdir
    fields = args.fields
    _id = args.id
    test_limit = args.testlimit

    build_blobs(meta_dir, blob_dir, fields, _id, test_limit=test_limit)
