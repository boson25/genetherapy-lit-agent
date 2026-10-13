# Gene-Therapy Literature Assistant

A tool-calling agent that answers gene-therapy research questions from PubMed
abstracts and cites every claim, plus an evaluation harness that measures
whether it does so honestly.

The agent is small on purpose. The point of the project is the measurement:
what the agent gets wrong, how that was found, and which fixes can be shown
to have worked.

## What it does

```
$ python agent.py "Why does FVIII expression decline faster than FIX after AAV gene therapy?"

  [search 1] {'query': 'FVIII FIX expression decline AAV gene therapy', 'n': 10}
  [search 2] {'query': 'factor VIII cellular stress hepatocyte unfolded protein response', 'n': 10}

FVIII expression declines more rapidly than FIX in part because FVIII is a
large, poorly secreted protein whose overexpression induces ER stress in
transduced hepatocytes [PMID 33183859] ...

------------------------------------------------------------
Searches:   2
Retrieved:  18 PMIDs
Cited:      ['33183859', '36103998']
Ungrounded: none
```

## Architecture

```
question
   │
   ▼
Claude (Haiku) ──tool_use──► search_pubmed()  ──► NCBI E-utilities
   ▲                              │                esearch → PMIDs
   │                              ▼                efetch  → abstracts
   └────────── tool_result ◄── formatted records
   │
   │  (up to 4 search rounds; the final round withholds the tools
   │   so the model must answer with what it has)
   ▼
answer with [PMID ...] citations
   │
   ▼
diagnostics: queries issued, PMIDs retrieved, PMIDs cited,
             PMIDs cited but never retrieved
```

Each answer carries the PMIDs the agent retrieved and the PMIDs it cited.
Comparing the two sets is what makes grounding measurable rather than assumed.

## Files

| File | Purpose |
|---|---|
| `tools.py` | PubMed retrieval via NCBI E-utilities; the tool schema Claude sees |
| `llm.py` | The only module that talks to the model provider |
| `agent.py` | The tool-calling loop and citation extraction |
| `eval.py` | Runs the question set, scores it, writes the manual-review file |
| `eval_questions.json` | 32 questions: 30 answerable, 2 unanswerable |
| `debug_test.py` | Step-by-step trace of the loop, used to diagnose failures |

`llm.py` exists so the provider is a one-file change. Nothing else in the
codebase imports the Anthropic SDK.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python test_key.py               # verify the key works
```

```bash
python agent.py "your question"  # one question
python eval.py                   # full question set
python eval.py --limit 3         # smoke test
python eval.py --score-only      # re-score the cache, no API calls
```

Results are cached in `results.json` and saved after every question, so an
interrupted run costs nothing extra. A full 32-question run is roughly $0.45
on Claude Haiku (~1.4 cents per question, ~4.4 searches each).

## Evaluation

The question set covers four liver-directed AAV gene therapy programs —
Wilson disease, OTC deficiency, PKU, and hemophilia A/B — plus cross-disease
comparisons. Questions are tagged `factual`, `synthesis`, or `unanswerable`,
and were written from domain experience rather than generated from the
literature.

The two `unanswerable` questions are traps with no honest answer: one cites
guidelines that do not exist and are dated in the future, the other asks for
the approved dose of a therapy that has never been approved. An agent that
answers them confidently is confabulating.

**Measured automatically**

| Metric | What it catches |
|---|---|
| Grounding | PMIDs cited that were never retrieved — i.e. invented |
| Citation presence | Answers that assert without citing |
| Citation format | One PMID per bracket, as the prompt requires |
| Refusal on unanswerable | Declining instead of inventing |

**Not measured automatically.** Whether each claim is actually supported by
the abstract it cites. No script can judge that. `eval.py` writes
`review.csv` — one row per claim-citation pair, with the claim text, the
paper title, a PubMed link, and a blank verdict column — so the judgment is
made against the source rather than from memory. This remains the project's
largest open gap.

## Results

Two runs over the same 32 questions, same model, same prompt.

| Metric | v1 | v2 |
|---|---|---|
| Grounded (no invented PMIDs) | 100% | 100% |
| Cited at least one source | 90% | **100%** |
| Citation format clean | 83% | 73% |
| Unanswerable declined | 2 / 2 | 2 / 2 |

v2 differs from v1 by one change: the loop now reserves its final round for
answering (see below).

## What the evaluation found

**1. A loop defect that produced silent empty results.** In v1, three
questions (10%) returned no answer at all. The agent had spent every
available round searching and never got a turn to respond — in one case
holding eight retrieved papers. Withholding the tools on the final round
forces an answer from what is already in hand. Citation presence went
90% → 100%.

**2. A noise floor larger than most prompt tweaks.** Citation format moved
83% → 73% between runs with no prompt change. On a 32-question set, that
variance is the detection limit: an improvement smaller than roughly ten
points cannot be distinguished from run-to-run chance without repeated runs.
This is the main reason the citation-format problem has not been "fixed" —
doing so without repeated measurement would produce a number, not evidence.

**3. A false negative in the scorer itself.** One unanswerable question was
flagged as answered when the agent had in fact declined; the refusal
detector matched "was unable to find" but not "I'm unable to find." The
metric was wrong, not the agent. Cached results made re-scoring free.

## Limitations

- Claim-level support is unverified; only the sampling plan exists.
- Retrieval quality is unmeasured. Several agent queries combine five or six
  concepts and return zero results, since PubMed ANDs all terms. There is no
  gold-standard paper set to score retrieval against.
- Abstracts only, not full text.
- One model, one prompt. `llm.py` makes a cross-model comparison
  straightforward, but it has not been run.
- 32 questions is small, and the measured variance reflects that.

## Next

1. Verified claim support on a stratified sample of `review.csv`.
2. Repeated v2/v3 runs so a prompt change for citation format can be
   evaluated against the noise floor rather than a single number.
3. Retrieval scoring against known-relevant papers per question.