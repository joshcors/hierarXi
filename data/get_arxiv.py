import requests, sys, time, json, xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import jsonlines
import numpy as np
import os
import re
import tqdm

file_dir = os.path.dirname(os.path.realpath(__file__))
arxiv_dir = os.path.join(file_dir, "arxiv", "batches")

BASE_ATOM = "https://export.arxiv.org/api/query"
BASE = "https://export.arxiv.org/oai2"
NS   = {"oai": "http://www.openarchives.org/OAI/2.0/",
        "arxiv": "http://arxiv.org/OAI/arXiv/"}
EARLIEST = "2012-01-01"
BATCH_SIZE = 3500
SUBBATCH_SIZE = 350

pattern = "batch_([0-9]+)\.jsonl"
all_files = os.listdir(arxiv_dir)
all_files = list(filter(lambda x : re.match(pattern, x), all_files))

IDS = []
batch_number = 0
for file in tqdm.tqdm(all_files, desc="Loading downloaded IDs"):
    path = os.path.join(arxiv_dir, file)
    batch_number = max(batch_number, int(re.match(pattern, file)[1]))

    with jsonlines.open(path, "r") as r:
        _ids = [obj["id"] for obj in r]
        IDS.extend(_ids)

IDS = set(IDS)
batch_number += 1

def iter_new_ids(since_iso: str):
    """Yield arXiv identifiers (with version) newer than `since_iso` (YYYY-MM-DD)."""
    params = {"verb": "ListIdentifiers",
              "metadataPrefix": "arXiv",
              "from": since_iso}
    while True:
        root = oai_call(params)
        for h in root.findall(".//oai:header", NS):
            raw = h.findtext("oai:identifier", namespaces=NS)     # oai:arXiv:2306.01234v1
            _id = raw.split(":")[-1]
            
            if _id in IDS: 
                continue

            yield _id
            
        rt = root.find(".//oai:resumptionToken", NS)
        if rt is None or not rt.text.strip():
            break
        params = {"verb": "ListIdentifiers", "resumptionToken": rt.text}

def oai_call(params):
    """Return parsed ElementTree for one OAI request, retrying on 503."""
    while True:
        r = requests.get(BASE, params=params, timeout=60)
        if r.status_code == 503:              # arXiv uses 503 Retry-After for politeness
            time.sleep(int(r.headers.get("Retry-After", 20)))
            continue
        r.raise_for_status()
        return ET.fromstring(r.text)
    
def fetch_batch(ids):
    """Return list of dicts {id,title,abstract} via Atom batch request."""
    id_list = ",".join(ids)
    r = requests.get(BASE_ATOM, params={"id_list": id_list, "max_results": len(ids)}, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        arx_id = entry.findtext("{http://www.w3.org/2005/Atom}id", default="").split("/")[-1].split("v")[0]
        title  = entry.find("{http://www.w3.org/2005/Atom}title").text.strip().replace("\n", " ")
        abstr  = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip().replace("\n", " ")
        out.append({"id": arx_id, "title": title, "abstract": abstr})
    return out

def fast_download():
    global batch_number
    model = SentenceTransformer('all-mpnet-base-v2')
    gen = iter_new_ids(EARLIEST)

    batch = []

    counter = tqdm.tqdm(total=BATCH_SIZE, desc=f"Batch {batch_number}")

    for _id in gen:

        batch.append(_id)
        counter.update(1)

        if len(batch) == BATCH_SIZE:

            batch = np.reshape(batch, (BATCH_SIZE // SUBBATCH_SIZE, SUBBATCH_SIZE)).tolist()

            results = []

            for subbatch in batch:
                results.extend(fetch_batch(subbatch))

            titles = [res["title"] for res in results]
            abstracts = [res["abstract"] for res in results]

            batch_titles = model.encode(titles)
            batch_abstracts = model.encode(abstracts)

            np.save(os.path.join(arxiv_dir, f"batch_{batch_number}_titles.npy"), batch_titles)
            np.save(os.path.join(arxiv_dir, f"batch_{batch_number}_abstracts.npy"), batch_abstracts)
            with jsonlines.open(os.path.join(arxiv_dir, f"batch_{batch_number}.jsonl"), "w") as writer:
                for meta in results:
                    writer.write(meta)

            print(f"Finished batch {batch_number}")

            batch = []
            batch_number += 1
            counter = tqdm.tqdm(total=BATCH_SIZE, desc=f"Batch {batch_number}")
    
def list_records():
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv"}
    root   = oai_call(params)

    model = SentenceTransformer('all-mpnet-base-v2').to("cuda")

    batch_n = 0

    while True:
        print(f"Processing batch {batch_n}")
        abstracts = []
        titles = []
        metas = []
        for rec in root.findall(".//oai:record", NS):
            meta = rec.find(".//arxiv:arXiv", NS)
            if meta is None:  # deleted record
                continue

            def txt(tag):
                return meta.findtext(f"arxiv:{tag}", default="", namespaces=NS)

            _id = txt("id")
            title = txt("title")
            created = txt("created")
            abstract = txt("abstract")
            authors = meta.findall("arxiv:authors/arxiv:author", namespaces=NS)
            doi = txt("doi")
            journal_ref = txt("journal-ref")
            catagories = txt("catagories").split()

            titles.append(title)
            abstracts.append(abstract)
            metas.append(
                {
                    "id": _id,
                    "title": title,
                    "created": created,
                    "abstract": abstract,
                    "authors": [{"forenames": au.findtext("arxiv:forenames", default="", namespaces=NS),
                                 "keyname":   au.findtext("arxiv:keyname",   default="", namespaces=NS),
                                 "affiliation": au.findtext("arxiv:affiliation", default="", namespaces=NS)} for au in authors],
                    "doi": doi,
                    "journal-ref": journal_ref,
                    "catagories": catagories
                }
            )

        batch_titles = model.encode(titles)
        batch_abstracts = model.encode(abstracts)

        np.save(os.path.join(arxiv_dir, f"batch_{batch_n}_titles.npy"), batch_titles)
        np.save(os.path.join(arxiv_dir, f"batch_{batch_n}_abstracts.npy"), batch_abstracts)
        with jsonlines.open(os.path.join(arxiv_dir, f"batch_{batch_n}.jsonl"), "w") as writer:
            for meta in metas:
                writer.write(meta)
        
        batch_n += 1
        
        rt = root.find(".//oai:resumptionToken", NS)
        if rt is None or rt.text is None or rt.text.strip() == "":
            break                       # no more pages
        root = oai_call({"verb": "ListRecords", "resumptionToken": rt.text})

if __name__ == "__main__":
    fast_download()
    # for rec in list_records():
    #     sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")