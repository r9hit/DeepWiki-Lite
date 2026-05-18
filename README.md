# deepwiki-lite

Documentation generator for GitHub repos. Point it at any Python repository and it produces persona-specific technical docs grounded in the actual source code.

## what it does

- Clones the repo (shallow) and parses every Python file with the AST module to extract classes, functions, imports, and detected frameworks
- Builds a TF-IDF retriever over per-file summaries so the LLM can only cite files that actually exist
- For each of four personas (Sysadmin, App Developer, ML Researcher, Perf Engineer), runs a four-step agent: plan questions, retrieve relevant files, synthesize the doc, self-grade it

## stack

Python 3.11, Groq (Llama 3.3 70B free tier), scikit-learn, GitPython, python-dotenv. No GPU. No paid APIs.

## quickstart

```bash
git clone https://github.com/YOUR_USERNAME/deepwiki-lite.git
cd deepwiki-lite
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add your Groq API key (free at console.groq.com) to a `.env` file:

GROQ_API_KEY=gsk_your_key_here

Run it:

```bash
python pipeline.py https://github.com/karpathy/nanoGPT
```

First run takes about 30 seconds. Subsequent runs are instant — every LLM call is cached on disk by prompt hash.

## why this is different from filename-keyword tools

Most repo-doc tools match keywords against filenames. A file called `inference_engine.py` with 500 lines of CUDA code gets missed because the filename doesn't contain "cuda". This one parses the AST, so it sees the actual imports and class definitions.

## status

Core pipeline complete. Coming next: citation-required vendor doc extraction, ground-truth eval harness, and a Streamlit UI.