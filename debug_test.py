"""Diagnostic: prints what happens at each step of the agent loop.

Run:  python debug_run.py
Shows per step: stop reason, which content blocks came back, how many
tokens went in, and how big the message history is. If the message list
stops growing, the tool results are not reaching the model.
"""
from llm import call_llm
from tools import SEARCH_PUBMED_TOOL, format_results, search_pubmed
from agent import SYSTEM_PROMPT, MAX_STEPS

QUESTION = ("How does the timing of gene therapy administration affect the "
            "reversal of hepatic copper accumulation and liver damage?")

messages = [{"role": "user", "content": QUESTION}]

for step in range(1, MAX_STEPS + 1):
    resp = call_llm(messages, tools=[SEARCH_PUBMED_TOOL], system=SYSTEM_PROMPT)

    block_types = [b.type for b in resp.content]
    print(f"\n--- step {step} ---")
    print(f"  messages in history : {len(messages)}")
    print(f"  input tokens        : {resp.usage.input_tokens}")
    print(f"  stop_reason         : {resp.stop_reason}")
    print(f"  blocks returned     : {block_types}")

    for b in resp.content:
        if b.type == "text":
            print(f"  text                : {b.text[:160]!r}")

    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason != "tool_use":
        print("\n  -> model finished (no tool request). Loop would end here.")
        break

    tool_results = []
    for b in resp.content:
        if b.type != "tool_use":
            continue
        print(f"  tool call           : {b.input}")
        records = search_pubmed(**b.input)
        content = format_results(records)
        print(f"  records retrieved   : {len(records)}")
        print(f"  result chars        : {len(content)}")
        print(f"  result preview      : {content[:160]!r}")
        tool_results.append(
            {"type": "tool_result", "tool_use_id": b.id, "content": content}
        )

    print(f"  tool_results built  : {len(tool_results)}")
    messages.append({"role": "user", "content": tool_results})
    print(f"  history after append: {len(messages)} messages")
else:
    print("\n  -> hit MAX_STEPS without a final answer.")