import os
import re
import ast
import hashlib
import json
import time
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from git import Repo
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = Groq()
MODEL = "llama-3.3-70b-versatile"

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def llm(prompt, temperature=0.2):
    key = hashlib.md5(f"{prompt}|{temperature}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # free tier is 30 rpm
    time.sleep(2.1)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = response.choices[0].message.content
    cache_file.write_text(text, encoding="utf-8")
    return text


def clone_repo(url, base_dir="repos"):
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    path = Path(base_dir) / name

    if path.exists():
        print(f"  already cloned: {path}")
        return path

    print(f"  cloning {url}")
    Repo.clone_from(url, path, depth=1)
    return path


FRAMEWORK_HINTS = {
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "jax": "JAX",
    "transformers": "Hugging Face Transformers",
    "vllm": "vLLM",
    "langchain": "LangChain",
    "fastapi": "FastAPI",
    "streamlit": "Streamlit",
    "flask": "Flask",
    "django": "Django",
    "habana_frameworks": "Intel Gaudi",
    "deepspeed": "DeepSpeed",
    "accelerate": "HF Accelerate",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "sklearn": "scikit-learn",
}

SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git", "build", "dist"}


def extract_code_signals(repo_path):
    signals = {
        "files": [],
        "classes": Counter(),
        "imports": Counter(),
        "frameworks": set(),
    }

    for py_file in repo_path.rglob("*.py"):
        if any(skip in py_file.parts for skip in SKIP_DIRS):
            continue

        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            if len(source) > 200_000:
                continue
            tree = ast.parse(source)
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue

        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])

        signals["classes"].update(classes)
        signals["imports"].update(imports)

        for imp in imports:
            if imp in FRAMEWORK_HINTS:
                signals["frameworks"].add(FRAMEWORK_HINTS[imp])

        # sort everything so cache keys stay stable across runs
        rel = str(py_file.relative_to(repo_path))
        summary = (
            f"{rel}: classes={sorted(classes)[:5]}, "
            f"functions={sorted(funcs)[:8]}, "
            f"imports={sorted(set(imports))[:8]}"
        )
        signals["files"].append((rel, summary))

    signals["classes"] = dict(signals["classes"].most_common(20))
    signals["imports"] = dict(signals["imports"].most_common(20))
    signals["frameworks"] = sorted(signals["frameworks"])

    return signals


class CodeRetriever:
    def __init__(self, files):
        self.files = files
        self.docs = [s for _, s in files]

        if not self.docs:
            self.vectorizer = None
            return

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def retrieve(self, query, k=5):
        if not self.vectorizer:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        # stable sort matters for ties, otherwise cache keys vary
        top_idx = sims.argsort(kind="stable")[-k:][::-1]
        return [self.files[i] for i in top_idx if sims[i] > 0.05]


PERSONAS = {
    "Sysadmin": "Cares about deployment, drivers, infrastructure, observability, multi-GPU setup.",
    "App Developer": "Cares about APIs, SDKs, integration patterns, request/response formats.",
    "ML Researcher": "Cares about model architecture, training pipelines, evaluation metrics.",
    "Perf Engineer": "Cares about kernels, memory, throughput, latency, profiling.",
}


def generate_persona_doc(persona, persona_desc, signals, retriever):
    plan_prompt = f"""You are a {persona}. {persona_desc}

You are about to read documentation for a repository with these characteristics:
- Frameworks detected: {sorted(signals['frameworks'])}
- Top classes: {sorted(list(signals['classes'].keys()))[:10]}
- Top imports: {sorted(list(signals['imports'].keys()))[:10]}

List exactly 4 specific questions you'd want this documentation to answer.
Output as a plain numbered list, nothing else. Example:
1. How do I deploy this?
2. What are the runtime dependencies?
..."""

    plan = llm(plan_prompt, temperature=0)
    questions = [q.strip() for q in re.findall(r"\d+\.\s*(.+)", plan)][:4]

    relevant_files = []
    for q in questions:
        hits = retriever.retrieve(q, k=3)
        relevant_files.extend(hits)

    seen = set()
    relevant_files = [
        f for f in relevant_files if not (f[0] in seen or seen.add(f[0]))
    ][:8]

    relevant_files = sorted(relevant_files, key=lambda f: f[0])

    evidence = "\n".join(f"- {summary}" for _, summary in relevant_files)
    synth_prompt = f"""Write documentation for this repository, aimed at a {persona}.
{persona_desc}

The {persona} wants these questions answered:
{chr(10).join(f"- {q}" for q in questions)}

Evidence from the codebase (these are the ONLY files you may cite):
{evidence}

Repository-level facts:
- Frameworks: {sorted(signals['frameworks'])}
- Notable classes: {sorted(list(signals['classes'].keys()))[:8]}

Write a focused doc with one section per question. Cite specific files
using backticks like `path/to/file.py`. Do NOT invent files that are
not in the evidence above. Keep total length around 400 words."""

    draft = llm(synth_prompt, temperature=0.3)

    critique_prompt = f"""Grade this documentation on a scale of 1-10 for a {persona}.

Criteria:
- Does it answer the persona's questions specifically?
- Does it cite real files from the evidence (not generic)?
- Is it actionable rather than vague?

Documentation:
{draft}

Output ONLY a JSON object on a single line, like:
{{"score": 7, "issues": "could cite more files", "verdict": "ACCEPT"}}"""

    crit_raw = llm(critique_prompt, temperature=0)
    try:
        match = re.search(r"\{.*\}", crit_raw, re.S)
        critique = json.loads(match.group()) if match else {}
    except Exception:
        critique = {"score": 5, "issues": "could not parse critique", "verdict": "ACCEPT"}

    return {
        "persona": persona,
        "questions": questions,
        "evidence_files": [f[0] for f in relevant_files],
        "doc": draft,
        "critique": critique,
    }


def analyze(repo_url):
    print(f"\n[1/4] cloning {repo_url}")
    path = clone_repo(repo_url)

    print("[2/4] extracting code signals")
    signals = extract_code_signals(path)
    print(f"      {len(signals['files'])} python files")
    print(f"      frameworks: {signals['frameworks']}")

    print("[3/4] building retriever")
    retriever = CodeRetriever(signals["files"])

    print("[4/4] generating persona docs")
    persona_docs = {}
    for persona, desc in PERSONAS.items():
        print(f"      {persona}")
        persona_docs[persona] = generate_persona_doc(persona, desc, signals, retriever)

    return {"signals": signals, "personas": persona_docs}


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/karpathy/nanoGPT"
    result = analyze(url)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for persona, data in result["personas"].items():
        score = data["critique"].get("score", "?")
        print(f"\n{'-' * 70}")
        print(f"{persona}  (self-critique: {score}/10)")
        print(f"{'-' * 70}")
        print(data["doc"])