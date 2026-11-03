**Table 5.** Size and inference cost on one Apple M4 workstation (CPU, one thread, unless stated). XGBoost values are ranges over the reference fold (first partition, fold 1) of the five cohorts, after truncation at the early-stopping iteration. Stored values are MLP weights and biases and, for XGBoost, one feature index and one threshold per split node and one value per leaf; these are different kinds of quantity, so serialized size is the more direct memory comparison. Latencies are for the classifier alone; filtering, beat detection and feature extraction are excluded. TabPFN timings include fitting (storing the context) and predicting one outer-test set of 1,000–3,032 records, and therefore measure a different operation from the prediction-only latencies of the students; CPU runs with more than 5,000 context samples are refused by the tabpfn package by default and were not timed. All timings were taken with no other analysis running on the machine.

| Model | Stored values | Serialized size | Latency, 1 record (ms) | Latency, 1,000 records (ms) |
|---|---|---|---|---|
| MLP student 50–128–64–5 | 15,109 | 62 kB | 0.011 | 0.08 |
| XGBoost, hard labels | 39,658–70,274 | 1.5–2.6 MB | 0.053–0.071 | 3.69–5.56 |
| XGBoost, distillation α = 0.5, T = 1 | 75,060–130,739 | 2.7–4.8 MB | 0.070–0.103 | 5.89–10.20 |
| TabPFN teacher, 8 estimators (Apple M4 GPU, MPS) | pre-trained transformer | 203 MB checkpoint | n/a (in-context) | 8,961–11,989 (fit + predict, scaled to 1,000 test records) |
| TabPFN teacher, 8 estimators (CPU) | pre-trained transformer | 203 MB checkpoint | n/a (in-context) | 19,879–20,437 (fit + predict, scaled to 1,000 test records) |
