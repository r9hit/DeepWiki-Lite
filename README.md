# deepwiki-lite

Documentation generator for GitHub repos. Point it at any Python repository and it produces persona-specific technical docs grounded in the actual source code, plus side modules for citation-verified vendor doc analysis, ground-truth quality evaluation, and Mermaid architecture diagrams.

## what it does

- **pipeline** — clones the repo (shallow) and parses every Python file with the AST module to extract classes, functions, imports, and frameworks. Builds a TF-IDF retriever over per-file summaries so the LLM can only cite files that actually exist. For each of four personas (Sysadmin, App Developer, ML Researcher, Perf Engineer), runs a four-step agent: plan questions, retrieve relevant files, synthesize the doc, self-grade it.
- **citations** — given a doc URL (vendor pages, framework docs, anything), scrapes the page and extracts feature claims with mandatory verbatim quotes from the source. Each quote is verified against the scraped text so hallucinated citations are caught automatically.
- **evals** — runs the pipeline against a small set of repos with hand-picked ground-truth facts and reports a pass rate. Surfaces real failure modes, like TF-IDF missing foundational dependencies in 1000+ file repos.
- **diagram** — converts the extracted class and import graph into Mermaid syntax for visual architecture diagrams.

## stack

Python 3.11, Groq (Llama 3.3 70B free tier), scikit-learn, GitPython, BeautifulSoup, python-dotenv. No GPU. No paid APIs.

## quickstart

```bash
git clone https://github.com/YOUR_USERNAME/deepwiki-lite.git
cd deepwiki-lite
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add your Groq API key (free at console.groq.com) to a `.env` file:

```
GROQ_API_KEY=gsk_your_key_here
```

## usage

Generate persona docs for any repo:

```bash
python pipeline.py https://github.com/karpathy/nanoGPT
```

Extract feature claims from a documentation page:

```bash
python citations.py https://docs.vllm.ai/en/latest/ vLLM
```

Run the eval suite:

```bash
python evals.py
```

Generate an architecture diagram (paste output at mermaid.live to render):

```bash
python diagram.py https://github.com/karpathy/nanoGPT
```

First run on a new repo takes about 30 seconds. Every LLM call is cached on disk by prompt hash, so re-runs are instant.

## why this is different from filename-keyword tools

Most repo-doc tools match keywords against filenames. A file called `inference_engine.py` with 500 lines of CUDA code gets missed because the filename doesn't contain "cuda". This one parses the AST, so it sees the actual imports and class definitions. The retriever then constrains the LLM to cite only files that exist in the repo, which eliminates hallucinated file references.

## caching

LLM calls are keyed on `md5(prompt + temperature)` and stored as files in `.cache/`. Cache hit rate is 100% on repeated runs of the same repo. Getting there required sorting every collection that goes into a prompt — Python's set iteration order is nondeterministic, which silently breaks caches if you stringify sets directly.

## status

Pipeline, citations, evals, and diagram modules complete. Streamlit UI coming next.