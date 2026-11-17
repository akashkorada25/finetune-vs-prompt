# Fine-tuned small model vs. prompted LLM

**When is a small fine-tuned model better than a large model with a good prompt?** This project answers that for a realistic production task: routing customer banking messages to one of **77 intents** (the [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) benchmark).

Five systems are compared on the same 770 test messages through [EvalForge](https://github.com/akashkorada25/evalforge), on accuracy, macro-F1, invalid outputs, latency and cost. A learning curve shows how much labelled data the small model needs to catch up with Claude.

| | System | What it needs |
|---|---|---|
| 1 | TF-IDF + logistic regression | Labelled data; trains in seconds |
| 2 | **Fine-tuned MiniLM** (33M parameters) | Labelled data; trains in minutes on a CPU |
| 3 | Claude Haiku 5.5, zero-shot | Only the list of labels |
| 4 | Claude Haiku 5.5, few-shot | The labels plus one example message per label |
| 5 | Claude Sonnet 5.5, few-shot | The same prompt on a larger model |

## Results

<!-- RESULTS:START -->
Results are produced by the Benchmark workflow (`.github/workflows/benchmark.yml`) and recorded here.
<!-- RESULTS:END -->

## How it works

- **Data** (`ftvp/data.py`): 10,003 training and 3,080 test messages. The head-to-head uses a fixed, balanced subset of 770 test messages (10 per intent), so Claude's cost stays low and every system sees exactly the same inputs. Few-shot examples come only from the training set, and a test checks they never overlap with the test set.
- **Baseline** (`ftvp/baseline.py`): TF-IDF on words and character n-grams, plus logistic regression.
- **Fine-tuning** (`ftvp/finetune.py`): `sentence-transformers/all-MiniLM-L12-v2` with a classification head, trained with AdamW, linear warm-up and decay, on GitHub's free CPU runner. Small datasets get more epochs, so every run takes at least 300 optimizer steps.
- **Claude** (`ftvp/llm.py`, `ftvp/prompts.py`): the label list (and, for few-shot, one real training message per label) goes in a cached system prompt. Replies are mapped to a valid label, tolerating quotes, capitals and near-miss spellings; anything else counts as an invalid output, and as wrong.
- **Evaluation**: `eval/build_suite.py` writes an EvalForge suite. Each system is called through EvalForge's `python` provider (`ftvp.systems:classify`), so latency, tokens and cost are measured the same way for all five. `ftvp report` computes accuracy, macro-F1 and invalid-output rate from the results.

## Run it

```bash
git clone https://github.com/akashkorada25/finetune-vs-prompt && cd finetune-vs-prompt
pip install -e ".[eval]"
ftvp baseline                                   # TF-IDF + LR, ~25 s

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train]"
ftvp train                                      # fine-tune MiniLM on all data (CPU)
ftvp curve --finetune                           # learning curve: 5, 10, 25, all examples per intent

export ANTHROPIC_API_KEY=...
python eval/build_suite.py
evalforge run eval/suite.yaml --concurrency 6   # all five systems on 770 messages
ftvp report                                     # accuracy, macro-F1, invalid outputs, latency, cost

ftvp predict "I still haven't received my new card" --system finetuned
```

On GitHub, the **Benchmark** workflow does all of this. Add an `ANTHROPIC_API_KEY` repository secret to include the Claude systems. The trained model is saved as a workflow artifact.

## Project structure

```
ftvp/
  data.py       Banking77 loading, balanced subsets, few-shot examples
  baseline.py   TF-IDF + logistic regression
  finetune.py   CPU fine-tuning and prediction with transformers
  llm.py        Anthropic client with prompt caching and cost tracking
  prompts.py    zero-shot and few-shot prompts, reply-to-label parsing
  systems.py    one function that EvalForge calls for every system
  metrics.py    accuracy, macro-F1, confusions
  report.py     head-to-head and learning-curve tables
  cli.py        ftvp command
eval/build_suite.py   generates the EvalForge suite (770 cases)
data/banking77/       dataset (CC-BY 4.0, PolyAI)
tests/                pytest
```

## Limitations

- One task and one dataset. Intent classification with a fixed label set is where fine-tuning should shine; open-ended tasks favour large models.
- The 770-message subset gives about ±2 percentage points of uncertainty on accuracy, so small differences are noise.
- Latency for the local models is measured one message at a time on a shared CPU runner; Claude's includes the network round trip.

## License and data

Code: MIT. Banking77 is by PolyAI and released under [CC BY 4.0](data/banking77/LICENSE-CC-BY-4.0.txt): Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, 2020. The data in `data/banking77/` is unmodified from [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) (commit 57ec275).
