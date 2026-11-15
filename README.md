# Kazakh LLM evaluation

Local Windows CPU evaluation of Llama-3.1-8B-Instruct and Llama-3.1-Sherkala-8B-Chat on 500 fixed KazQAD-derived questions.

The code generates paired answers, prepares evaluation data and calculates accuracy, response-level hallucination rates, Wilson intervals, paired bootstrap intervals, exact McNemar tests, Holm correction and Cohen's kappa.

## What is included

- `run_defense_demo.py`: generate fresh answers to the first five fixed questions.
- `run_full_500.py`: generate 1,000 fresh answers, 500 from each model.
- `run_analysis_500.py`: recalculate statistics from the local final annotation package.
- `infer_main.py`: shared inference runner for the 500-question benchmark.
- `kazqad_eval_pipeline.py`: data preparation and statistical analysis.
- `llama_native_b11739.py`: native llama.cpp CPU binding.
- `data/main_500.json`: the fixed 500-question benchmark.
- `templates/`: model-specific chat templates.
- `.vscode/`: launch configurations for the demo, full run and analysis.

- `annotation_final_500/`: the two independent rater files and the consensus file for all 1,000 responses (`rater_1_500.csv`, `rater_2_500.csv`, `consensus_500.csv`). Each row holds the question, the blinded response, the reference answers and interpretation, the evidence, five binary labels, error type, verification source and notes.

Model weights, tokenizer files, runtime binaries and credentials are not included in this repository. The private blinding key that links response identifiers to models is kept outside the repository.

## Setup

Use Python 3.11 on Windows. From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For inference, supply the following local assets from the working RMT3 installation, preserving these paths:

```text
models/baseline-Q4_K_M.gguf
models/sherkala-Q4_K_M.gguf
runtime/                 # llama.cpp b11739 Windows DLLs, including llama.dll
tokenizers/baseline_llama31/
tokenizers/adapted_sherkala/
```

`laptop_config.json` records the source model revisions and required model checksums. Access to Meta's model may require accepting its license on Hugging Face. Obtain and use each model under its applicable license. The native binding targets llama.cpp b11739; another runtime version may be incompatible.

## Run

Validate the question selection without starting inference:

```powershell
.\.venv\Scripts\python.exe run_defense_demo.py --prepare-only
.\.venv\Scripts\python.exe run_full_500.py --prepare-only
```

Generate new answers:

```powershell
.\.venv\Scripts\python.exe run_defense_demo.py
.\.venv\Scripts\python.exe run_full_500.py
```

Each execution writes to a new timestamped directory under `runs/`. The demo produces a readable `responses.html`. A new full run produces raw answers and a manifest; it does not automatically create human ratings or consensus labels.

For analysis of the existing study, copy the `annotation_final_500` folder into `outputs/Assignment3_500_Final/` and add the private blinding key `PRIVATE_combined_blinding_key.json` to the same folder, then run:

```powershell
.\.venv\Scripts\python.exe run_analysis_500.py
```

This recalculates statistics from saved labels; it does not run either model. Agreement uses the two original rating sets, while outcome rates use the final adjudicated labels.

Alternatively, open the repository in VS Code and select a configuration in **Run and Debug**.

## Reproduction Test on macOS (Apple Silicon)

The statistical analysis for release `assignment-4` (commit `2560faf892ffe64aad5062ed9238cf44ec07067e`) was independently reproduced on a MacBook Air with Apple M3, macOS 26.6.2, ARM64 architecture, and Python 3.13.2. This test recalculates statistics from existing human annotations; it does not perform model inference.

Place the existing annotation package in `annotation_final_500/` at the repository root. It must contain:

- `rater_1_500.csv`
- `rater_2_500.csv`
- `consensus_500.csv`
- `PRIVATE_combined_blinding_key.json`

The annotation files and private blinding key are not distributed through the public repository and must be obtained separately from the final project package.

From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip check
```

Run the analysis:

```bash
python3 kazqad_eval_pipeline.py analyze \
  --rater1 annotation_final_500/rater_1_500.csv \
  --rater2 annotation_final_500/rater_2_500.csv \
  --consensus annotation_final_500/consensus_500.csv \
  --key annotation_final_500/PRIVATE_combined_blinding_key.json \
  --out analysis_repeated_500.json
```

The original reproduction instructions used the incorrect path `code/kazqad_eval_pipeline.py`. The correct path in this release is `kazqad_eval_pipeline.py` at the repository root.

Dependencies installed successfully, `pip check` passed, and the analysis completed with exit code 0, producing `analysis_repeated_500.json` for 500 paired questions:

| Model | Correct | Accuracy |
| --- | --- | --- |
| Sherkala | 253/500 | 50.6% |
| Llama | 116/500 | 23.2% |

Both accuracies exactly match the original report. Source code and annotation files remained unchanged during the test.

## Headline results (500 paired questions, release `final`)

| Outcome | Llama-3.1-8B-Instruct | Sherkala-8B-Chat |
|---|---|---|
| Correct direct answer | 116 (23.2%) | 253 (50.6%) |
| Confirmed false claim | 237 (47.4%) | 173 (34.6%) |

Benchmark checksum: `data/main_500.json` SHA-256 `9f88cf7658182a127d25c469db104129de39ce51e84ddabc14baa61fcc216503`.

## Reproducing the manuscript tables and figures

| Manuscript item | How to reproduce |
|---|---|
| Tables 2 and 3, agreement (Section 4.2) | `python run_analysis_500.py` (see above) |
| Tables 2, 4, 5 and timing (Sections 4.3-4.5) | `python make_tables.py path/to/PRIVATE_combined_blinding_key.json` |
| Figures 1 and 2 | PNG files and the chart workbook with their source counts are in `figures/` |
| Raw generations with timings and finish reasons | `logs_500/raw_generations_500.jsonl`, `logs_500/combined_manifest.json` |

`make_tables.py` reads `data/main_500.json`, `annotation_final_500/consensus_500.csv` and `logs_500/` and prints the counts shown in the manuscript. The blinding key is distributed with the final project package, not in this repository.

## Interpretation

The study compares two local Q4_K_M configurations with a 2,048-token context, eight CPU threads and a 128-token output cap. It is exploratory: it was not pre-registered, uses one sample and one run, and the questions were screened by a single reviewer. Subject imbalance, related questions, annotation uncertainty, quantization and laptop conditions limit generalization. A difference between the models does not isolate a causal effect of Kazakh adaptation.

## Data and third-party attribution

The benchmark is derived from [KazQAD](https://github.com/IS2AI/KazQAD), pinned source revision `ab8684a667109f2a7345f4f89ab3b123e8d78e1f`. Questions were selected and adapted for closed-book generation; retained passages support evaluation. Dataset attribution and the CC BY-SA 4.0 license are in `data/`.

The upstream llama.cpp header in `code_provenance/` belongs to [llama.cpp](https://github.com/ggml-org/llama.cpp). No model weights or runtime binaries are distributed here. Project-authored code is released under the MIT licence (see LICENSE); the benchmark data keep their CC BY-SA 4.0 licence.
