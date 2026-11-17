# Learning curve

Accuracy with 5, 10, 25 and all training examples per intent. Subset = the 770-message head-to-head subset; full = all 3,080 test messages.

| Method | Examples per intent | Training examples | Subset accuracy | Full-test accuracy | Macro-F1 | Train time |
|---|---|---|---|---|---|---|
| tfidf | 5 | 385 | 67.0% | 63.9% | 0.635 | 2 s |
| finetuned | 5 | 385 | 80.4% | 80.1% | 0.799 | 344 s |
| tfidf | 10 | 770 | 74.2% | 75.4% | 0.753 | 3 s |
| finetuned | 10 | 770 | 87.0% | 87.9% | 0.879 | 317 s |
| tfidf | 25 | 1925 | 84.5% | 83.7% | 0.837 | 5 s |
| finetuned | 25 | 1925 | 91.3% | 91.6% | 0.916 | 423 s |
| tfidf | all | 10003 | 91.8% | 91.3% | 0.914 | 26 s |
| finetuned | all | 10003 | 94.3% | 94.2% | 0.942 | 2201 s |

GitHub Actions run 44153739312.
