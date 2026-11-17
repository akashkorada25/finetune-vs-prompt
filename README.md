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
### Head-to-head: the same 770 test messages (10 per intent)

| System | Accuracy | Macro-F1 | Invalid outputs | Median latency | API cost per 1,000 messages |
|---|---|---|---|---|---|
| **Fine-tuned MiniLM (33M)** | **94.3%** | **0.943** | 0% | 66 ms | **$0** (runs on CPU) |
| TF-IDF + logistic regression | 91.8% | 0.917 | 0% | 9 ms | $0 |
| Claude Sonnet 5.5, few-shot | 82.9% | 0.843 | 5.5% | 747 ms | $0.81 |
| Claude Haiku 5.5, few-shot | 79.4% | 0.792 | 1.6% | 539 ms | $0.04 |
| Claude Haiku 5.5, zero-shot | 68.4% | 0.714 | 13.0% | 655 ms | $0.02 |

### How much labelled data does the small model need?

Accuracy on the same 770 messages ([full results](results/curve.md)):

| Examples per intent | Training examples | TF-IDF + LR | Fine-tuned MiniLM | For comparison |
|---|---|---|---|---|
| 0 | 0 | - | - | Claude Haiku zero-shot: 68.4% |
| 5 | 385 | 67.0% | **80.4%** | Claude Haiku few-shot: 79.4% |
| 10 | 770 | 74.2% | **87.0%** | Claude Sonnet few-shot: 82.9% |
| 25 | 1,925 | 84.5% | **91.3%** | |
| all (~130) | 10,003 | 91.8% | **94.3%** | |

### What the experiment found

1. **On a fixed set of 77 intents, the small fine-tuned model won clearly**: 94.3% accuracy against 82.9% for Claude Sonnet 5.5 with a few-shot prompt, about 10× faster than the Sonnet setup (66 ms vs. 747 ms), and with no per-message API cost. Even the TF-IDF baseline beat both Claude models.
2. **It doesn't need much data.** With only **5 labelled examples per intent** (385 in total), the fine-tuned model matched Claude Haiku's few-shot prompt. With **10 per intent** it beat Claude Sonnet. Training took 6–37 minutes on a free CPU machine; no GPU was needed.
3. **The LLMs struggled with near-duplicate intents and the exact label set.** Their most common mistakes were between labels that differ only in Banking77's conventions (`card_delivery_estimate` vs. `card_arrival`, `beneficiary_not_allowed` vs. `failed_transfer`). They also sometimes replied with a label that doesn't exist: 13% of the time zero-shot, 1.6–5.5% with examples. A model trained on the labelled data learns those boundaries; a prompt has to describe them.
4. **Claude wins where there is no labelled data.** With zero examples it still reached 68%, which no trained model can do. A practical path: start with an LLM, use its predictions (corrected by people) as training data, then switch to a small fine-tuned model once there are a few examples per class.
5. **Cost at scale:** at 1 million messages a month, the Sonnet prompt would cost about $810 a month in API fees, Haiku about $40, and the fine-tuned model only the CPU it runs on.
6. **Getting fine-tuning right mattered.** The first attempt barely learned from small datasets (final loss ~2.8; 71.8% accuracy with 5 examples per intent). Using MiniLM's native mean pooling, starting the classifier from each intent's average embedding, and a separate learning rate for the new layer raised that to 80.4%, and full-data accuracy from 93.0% to 94.3%. The project's git history keeps both versions.

The 770-message subset gives roughly ±2 points of uncertainty, so the large gaps (fine-tuned vs. LLM, 0 vs. 5 examples) are the reliable findings. The Claude prompts list the labels and one example each, with no label definitions; richer prompts would narrow the gap, at higher cost per message.
<!-- RESULTS:END -->

## How it works

- **Data** (`ftvp/data.py`): 10,003 training and 3,080 test messages. The head-to-head uses a fixed, balanced subset of 770 test messages (10 per intent), so Claude's cost stays low and every system sees exactly the same inputs. Few-shot examples come only from the training set, and a test checks they never overlap with the test set.
- **Baseline** (`ftvp/baseline.py`): TF-IDF on words and character n-grams, plus logistic regression.
- **Fine-tuning** (`ftvp/finetune.py`): `sentence-transformers/all-MiniLM-L12-v2` with mean pooling and a linear head, trained with AdamW, linear warm-up and decay, on GitHub's free CPU runner. The head starts as a nearest-centroid classifier (each intent's average embedding) and learns at 1e-3 while the pretrained encoder learns at 5e-5. Small datasets get more epochs, so every run takes at least 300 optimizer steps.
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
