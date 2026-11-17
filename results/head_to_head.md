# Head-to-head: 770 test messages (10 per intent)

EvalForge run `20261117-200308-ef83af`.

| System | Accuracy | Macro-F1 | Invalid outputs | Median latency | API cost per 1,000 messages |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 91.8% | 0.917 | 0.0% | 9 ms | $0 (runs locally) |
| Fine-tuned MiniLM (33M) | 94.3% | 0.943 | 0.0% | 66 ms | $0 (runs locally) |
| Claude Haiku 5.5, zero-shot | 68.4% | 0.714 | 13.0% | 655 ms | $0.02 |
| Claude Haiku 5.5, few-shot | 79.4% | 0.792 | 1.6% | 539 ms | $0.04 |
| Claude Sonnet 5.5, few-shot | 82.9% | 0.843 | 5.5% | 747 ms | $0.81 |

- TF-IDF + logistic regression: most common mistakes: top_up_reverted -> top_up_failed (3); card_arrival -> card_delivery_estimate (2); pin_blocked -> get_physical_card (2)
- Fine-tuned MiniLM (33M): most common mistakes: top_up_reverted -> top_up_failed (3); declined_transfer -> failed_transfer (2); virtual_card_not_working -> card_not_working (2)
- Claude Haiku 5.5, zero-shot: most common mistakes: get_physical_card -> pin_blocked (8); declined_card_payment -> card_not_working (5); reverted_card_payment? -> __invalid__ (5)
- Claude Haiku 5.5, few-shot: most common mistakes: card_delivery_estimate -> card_arrival (4); top_up_by_bank_transfer_charge -> transfer_fee_charged (4); compromised_card -> card_payment_not_recognised (4)
- Claude Sonnet 5.5, few-shot: most common mistakes: get_physical_card -> __invalid__ (5); beneficiary_not_allowed -> failed_transfer (5); card_delivery_estimate -> card_arrival (4)

## Learning curve (accuracy on the same 770 messages)

| Examples per intent | Training examples | TF-IDF + LR | Fine-tuned MiniLM |
|---|---|---|---|
| 5 | 385 | 67.0% | 80.4% |
| 10 | 770 | 74.2% | 87.0% |
| 25 | 1,925 | 84.5% | 91.3% |
| all | 10,003 | 91.8% | 94.3% |

GitHub Actions run 44153739312 (Nov 17, 2026).
