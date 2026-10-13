"""Tool-calling agent: answers gene-therapy questions from PubMed abstracts."""
import re
import sys

from llm import MODEL, call_llm
from tools import SEARCH_PUBMED_TOOL, format_results, search_pubmed

SYSTEM_PROMPT = """You are a gene-therapy literature assistant. Answer questions \
using ONLY PubMed abstracts retrieved with the search_pubmed tool.

Rules:
- Always search before answering. Use focused keyword queries.
- Cite every factual claim as [PMID 12345678], one PMID per bracket.
- Only cite PMIDs that appeared in your search results. Never invent one.
- If the retrieved abstracts don't answer the question, say so plainly.
- Be concise: 1-3 short paragraphs."""

MAX_SEARCHES = 8   # cap tool calls per question (prevents runaway loops and saves cost)
MAX_STEPS = 4      # hard cap on model calls per question


def run_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    """Execute a tool call. Returns (text for the model, records retrieved)."""
    if name != "search_pubmed":
        return f"Unknown tool: {name}", []
    try:
        records = search_pubmed(**args)
        return format_results(records), records
    except Exception as e:  # network/HTTP errors go back to the model to prevent a crash
        return f"Search failed: {e}", []


def extract_cited_pmids(text: str) -> set[str]:
    """Find PMIDs inside [PMID ...] citations in the answer text."""
    pmids = set()
    for bracket in re.findall(r"\[([^\]]*PMID[^\]]*)\]", text):
        pmids.update(re.findall(r"\d{6,9}", bracket))
    return pmids


def answer(question: str, verbose: bool = True) -> dict:
    """Run the agent loop for one question and return answer + diagnostics."""
    messages = [{"role": "user", "content": question}]
    retrieved: dict[str, dict] = {}   # pmid, refer by an agent
    queries: list[str] = []
    n_searches = 0
    tokens = {"input": 0, "output": 0}

    for _ in range(MAX_STEPS):
        resp = call_llm(messages, tools=[SEARCH_PUBMED_TOOL], system=SYSTEM_PROMPT)
        tokens["input"] += resp.usage.input_tokens
        tokens["output"] += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})

        # The final answer
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            cited = extract_cited_pmids(text)
            return {
                "question": question,
                "answer": text,
                "queries": queries,
                "retrieved_pmids": sorted(retrieved),
                "retrieved_records": retrieved, 
                "cited_pmids": sorted(cited),
                "ungrounded_pmids": sorted(cited - set(retrieved)),  # cited but never retrieved
                "n_searches": n_searches,
                "tokens": tokens,
                "model": MODEL,
            }

        # Run every tool call in this turn and send all results back together.
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            n_searches += 1
            if n_searches > MAX_SEARCHES:
                content = "Search limit reached. Answer now using the results you already have."
            else:
                queries.append(block.input.get("query", ""))
                if verbose:
                    print(f"  [search {n_searches}] {block.input}")
                content, records = run_tool(block.name, block.input)
                retrieved.update({r["pmid"]: r for r in records})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )
        messages.append({"role": "user", "content": tool_results})

    return {"question": question, "answer": "Stopped: step limit reached.",
            "queries": queries, "retrieved_pmids": sorted(retrieved),
            "retrieved_records": retrieved,
            "cited_pmids": [], "ungrounded_pmids": [], "n_searches": n_searches,
            "tokens": tokens, "model": MODEL}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or (
        "What genetic and immunological factors influence the development of FVIII inhibitors in patients with hemophilia A?"
    )
    print(f"Q: {q}\n")
    result = answer(q)
    print(f"\n{result['answer']}\n")
    print("-" * 60)
    print(f"Searches:   {result['n_searches']}  {result['queries']}")
    print(f"Retrieved:  {len(result['retrieved_pmids'])} PMIDs")
    print(f"Cited:      {result['cited_pmids']}")
    print(f"Ungrounded: {result['ungrounded_pmids'] or 'none'}")
    print(f"Tokens:     {result['tokens']}")