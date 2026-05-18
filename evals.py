import sys
from pipeline import analyze


GROUND_TRUTH = {
    "https://github.com/karpathy/nanoGPT": {
        "name": "nanoGPT",
        "facts": ["pytorch", "transformer", "training"],
    },
    "https://github.com/psf/requests": {
        "name": "requests",
        "facts": ["http", "session"],
    },
    "https://github.com/tiangolo/fastapi": {
        "name": "fastapi",
        "facts": ["pydantic", "starlette", "async"],
    },
}


def check_facts(generated_text, expected_facts):
    text_lower = generated_text.lower()
    return [(fact, fact.lower() in text_lower) for fact in expected_facts]


def evaluate_repo(repo_url, expected_facts):
    result = analyze(repo_url)

    all_text = ""
    for persona_data in result["personas"].values():
        all_text += persona_data["doc"] + "\n"

    return check_facts(all_text, expected_facts)


def run_suite():
    print("\neval suite")
    print("-" * 50)

    summary = []

    for url, info in GROUND_TRUTH.items():
        name = info["name"]
        expected = info["facts"]

        print(f"\nrepo: {name}")
        print(f"url:  {url}")

        try:
            results = evaluate_repo(url, expected)
        except Exception as e:
            print(f"  error: {e}")
            summary.append((name, 0, len(expected)))
            continue

        passed = sum(1 for _, ok in results if ok)
        for fact, ok in results:
            mark = "pass" if ok else "FAIL"
            print(f"  [{mark}] {fact}")

        summary.append((name, passed, len(expected)))

    print("\nsummary")
    print("-" * 50)

    total_passed = 0
    total_facts = 0

    for name, passed, total in summary:
        pct = (passed / total * 100) if total else 0
        print(f"  {name:20s} {passed}/{total}  ({pct:.0f}%)")
        total_passed += passed
        total_facts += total

    overall_pct = (total_passed / total_facts * 100) if total_facts else 0
    print(f"  {'-' * 40}")
    print(f"  {'overall':20s} {total_passed}/{total_facts}  ({overall_pct:.0f}%)")


if __name__ == "__main__":
    run_suite()