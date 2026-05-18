import json
import re
import requests
from bs4 import BeautifulSoup

from pipeline import llm


def fetch_doc_text(url, max_chars=8000):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["nav", "footer", "aside", "script", "style", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 40]
    cleaned = "\n".join(lines)
    return cleaned[:max_chars]


def extract_features_with_citations(url, vendor_name="this product"):
    doc_text = fetch_doc_text(url)

    prompt = f"""You are a technical analyst. Read this documentation page for {vendor_name}
and extract claims about its features.

Rules:
1. For every feature you list, include a direct verbatim quote from the doc below.
2. The quote must appear EXACTLY in the doc text. Do not paraphrase, do not invent.
3. If you cannot find a supporting quote for a feature, do not include that feature.
4. Quotes should be 5-25 words.

Identify 5-7 features mentioned in the doc.

Output strict JSON only, no markdown:
{{
  "features": [
    {{
      "feature": "Feature name",
      "claim": "What the doc says about it in plain English",
      "quote": "Exact verbatim string from the doc",
      "category": "deployment | performance | api | hardware | observability | other"
    }}
  ]
}}

Documentation text:
---
{doc_text}
---
"""

    raw = llm(prompt, temperature=0)

    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"error": "could not parse json", "raw": raw, "features": []}

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        return {"error": f"json parse failed: {e}", "raw": raw, "features": []}

    # check each quote actually appears in the doc, catches hallucinated quotes
    for feature in data.get("features", []):
        quote = feature.get("quote", "")
        quote_norm = re.sub(r"\s+", " ", quote.strip())
        doc_norm = re.sub(r"\s+", " ", doc_text)
        feature["verified"] = quote_norm in doc_norm

    data["url"] = url
    data["vendor"] = vendor_name
    return data


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://docs.vllm.ai/en/latest/"
    vendor = sys.argv[2] if len(sys.argv) > 2 else "vLLM"

    print(f"\nextracting features from: {url}")
    print(f"vendor: {vendor}\n")

    result = extract_features_with_citations(url, vendor)

    if "error" in result:
        print(f"error: {result['error']}")
        print("\nraw output:")
        print(result.get("raw", ""))
    else:
        print(f"found {len(result['features'])} features\n")
        for i, f in enumerate(result["features"], 1):
            mark = "ok" if f.get("verified") else "HALLUCINATED"
            print(f"{i}. [{f.get('category', 'other')}] {f.get('feature', '?')}")
            print(f"   claim: {f.get('claim', '?')}")
            print(f"   quote: \"{f.get('quote', '?')}\"  [{mark}]")
            print()