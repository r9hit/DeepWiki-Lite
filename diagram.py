import re
from pipeline import llm


def generate_mermaid(signals):
    classes = sorted(list(signals.get("classes", {}).keys()))[:15]
    imports = sorted(list(signals.get("imports", {}).keys()))[:15]
    frameworks = sorted(signals.get("frameworks", []))

    if not classes and not imports:
        return "graph TD\n    A[no structure found]"

    prompt = f"""Generate a Mermaid graph showing the architecture of a Python repository.

Classes detected: {classes}
Frameworks used: {frameworks}
Top imports: {imports}

Rules:
1. Use `graph TD` (top-down) syntax.
2. Group related classes into subgraphs if it makes sense.
3. Show frameworks as external dependencies pointing into the relevant classes.
4. Use only ASCII characters in node names. No parentheses, no quotes.
5. Maximum 20 nodes. Pick the most architecturally important ones.
6. Output ONLY the Mermaid code, no explanation, no markdown fences.

Example output format:
graph TD
    PyTorch --> Model
    Model --> Attention
    Model --> MLP
    Trainer --> Model
"""

    raw = llm(prompt, temperature=0)

    # strip markdown fences if the model added them
    cleaned = re.sub(r"^```(?:mermaid)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if not cleaned.lstrip().startswith("graph"):
        cleaned = "graph TD\n" + cleaned

    return cleaned


if __name__ == "__main__":
    import sys
    from pipeline import clone_repo, extract_code_signals

    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/karpathy/nanoGPT"

    print(f"\nanalyzing {url}")
    path = clone_repo(url)
    signals = extract_code_signals(path)

    print(f"classes: {len(signals['classes'])}, imports: {len(signals['imports'])}")
    print("\ngenerating diagram...\n")

    diagram = generate_mermaid(signals)
    print(diagram)
    print("\npaste this at https://mermaid.live to render")