# Kazakh LLM evaluation

Local Windows CPU evaluation of Llama-3.1-8B-Instruct and Llama-3.1-Sherkala-8B-Chat on 500 fixed KazQAD-derived questions.

The code generates paired answers, prepares evaluation data and calculates accuracy, response-level hallucination rates, Wilson intervals, paired bootstrap intervals, exact McNemar tests, Holm correction and Cohen's kappa.

## What is included

- `run_defense_demo.py`: generate fresh answers to the first five fixed questions.
- `run_full_500.py`: generate 1,000 fresh answers, 500 from each model.
- `run_analysis_500.py`: recalculate statistics from the local final annotation package.
- `infer_main_100.py`: shared inference runner; its historical name remains, but it supports 100, 400 and 500 questions.
- `kazqad_eval_pipeline.py`: data preparation and statistical analysis.
- `llama_native_b11739.py`: native llama.cpp CPU binding.
- `data_extension_400/main_500.json`: the fixed 500-question benchmark.
- `templates/`: model-specific chat templates.
- `.vscode/`: launch configurations for the demo, full run and analysis.

Model weights, tokenizer files, runtime binaries, credentials, raw responses, human rating files and reports are not included in this code repository. Existing research results are kept in the separate final project package.

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

For analysis of the existing study, copy the `annotation_final_500` folder from the final package into `outputs/Assignment3_500_Final/`, then run:

```powershell
.\.venv\Scripts\python.exe run_analysis_500.py
```

This recalculates statistics from saved labels; it does not run either model. Agreement uses the two original rating sets, while outcome rates use the final adjudicated labels.

Alternatively, open the repository in VS Code and select a configuration in **Run and Debug**.

## Interpretation

The study compares two local Q4_K_M configurations with a 2,048-token context, eight CPU threads and a 128-token output cap. It is exploratory: the sample was expanded after the first 100 results were inspected. Subject imbalance, related questions, annotation uncertainty, quantization and laptop conditions limit generalization. A difference between the models does not isolate a causal effect of Kazakh adaptation.

## Data and third-party attribution

The benchmark is derived from [KazQAD](https://github.com/IS2AI/KazQAD), pinned source revision `ab8684a667109f2a7345f4f89ab3b123e8d78e1f`. Questions were selected and adapted for closed-book generation; retained passages support evaluation. Dataset attribution and the CC BY-SA 4.0 license are in `data_extension_400/`.

The upstream llama.cpp header in `code_provenance/` belongs to [llama.cpp](https://github.com/ggml-org/llama.cpp). No model weights or runtime binaries are distributed here. No separate license is granted for the project-authored code at this stage.
