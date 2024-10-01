#RAG with ell

## Install
```
python -m venv .venv
./.venv/scripts/Activate.ps1
pip install -e ".[dev]"
pre-commit install
```

## Selected Files
1. [`rag_example_simple.py`](src/beginner/examples/rag_example_simple.py) attempts to use [ell](https://github.com/MadcowD/ell) for RAG with both remote and local LLMs.