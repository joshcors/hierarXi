from pathlib import Path

def find_project_root(start=None):
    sentinels = {"setup.py", "requirements.txt", "Makefile"}

    p = Path(__file__).resolve() if start is None else Path(start)
    for parent in [p, ] + list(p.parents):
        if all((parent / s).exists() for s in sentinels):
            return parent
    raise RuntimeError("No root found")