# Fine_Tuining

PDF-to-model fine-tuning pipeline for Qwen3 0.6B. It extracts educational text,
creates simple supervised examples, fine-tunes with PEFT/LoRA on CPU, merges
the adapter, and can package the result for Ollama.

> **Model:** This project uses `Qwen/Qwen3-0.6B`. The former `qwen3.5:0.8b`
> label was inaccurate; the trained checkpoint is Qwen3 0.6B, not Qwen3.5 0.8B.

## Pipeline overview

The pipeline has four data-preparation stages followed by CPU fine-tuning:

1. Extract text from PDFs into page-level JSON.
2. Clean extracted text and normalize paragraphs.
3. Turn paragraphs into user/assistant chat examples in JSONL.
4. Apply the selected model tokenizer's chat template and save token IDs.
5. Fine-tune the base model with a PEFT LoRA adapter, save the adapter, and
  merge it with the base model.

The generated examples are a basic reconstruction task: the user message asks
for an explanation of a paragraph, and the assistant target is the same
paragraph. This does not create independent question-answer pairs or verify
factual accuracy. Review the generated dataset before training if you need a
different learning objective.

All commands below are intended to run from the repository root.

## Project layout

| Path | Purpose |
| --- | --- |
| `Extract_Pdf.py` | Reads `data/pdfs/*.pdf`; writes page-level JSON under `data/raw/`. |
| `Clean_Text.py` | Normalizes the extracted JSON; writes cleaned documents under `data/clean/`. |
| `Training_Dataset.py` | Builds `data/processed/training.jsonl` from cleaned page text. |
| `Tokenize.py` | Applies the configured model tokenizer and writes `data/tokenized/`. |
| `fine_tuneing.py` | Validates the tokenized data, trains the CPU LoRA adapter, and saves adapter and merged outputs under `models/`. |
| `Evaluate_Model.py` | Scores the merged model on held-out JSONL examples and saves generated/reference samples. |
| `test.py` | Checks the Python/dependency setup and validates the generated dataset and token IDs. |
| `requirements.txt` | Python dependencies and compatible PyTorch/torchvision pins. |
| `Modelfile` | Imports the converted GGUF model into Ollama. |
| `data/pdfs/` | PDF inputs. These are source data, not disposable build output. |
| `data/raw/`, `data/clean/`, `data/processed/` | Regenerable intermediate datasets. |
| `data/tokenized/` | Regenerable Hugging Face dataset cache; excluded from Git. |
| `models/` | Generated training outputs; excluded from Git. |

## Requirements

- Windows 10/11 or Linux. Training is configured for CPU.
- Python 3.12 is recommended; the smoke-test script accepts Python 3.10 through
  3.13. Python 3.12.15 was used for development checks.
- `uv` is used below to create the environment and install packages.
- Enough system memory for the float32 base model, LoRA optimizer state, and
  training activations. CPU training can be terminated by the OS when memory is
  exhausted; gradient checkpointing and the short sequence limit reduce but do
  not eliminate that risk.
- Internet access on first model/tokenizer load, unless the Hugging Face files
  are already cached locally.

No CUDA, GPU, or running Ollama service is used by the training script. The
model identifier is a Hugging Face repository name; Ollama deployment is a
separate step described below.

## Installation

Choose the setup instructions for your operating system. Run them from the
repository root. `uv` installs Python 3.12 if it is not already available.

### Linux

From the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

To activate the environment in later terminals:

```bash
source .venv/bin/activate
```

### Windows PowerShell

From the repository root:

```powershell
winget install --id=astral-sh.uv -e
# Close and reopen PowerShell after installing uv, then continue:
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

To activate the environment in later PowerShell sessions:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
winget install --id=astral-sh.uv -e
REM Reopen Command Prompt after installing uv, then continue:
uv python install 3.12
uv venv --python 3.12 .venv
.venv\Scripts\activate.bat
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

To activate in later terminals:

```cmd
.venv\Scripts\activate.bat
```

The dependency file pins the PyTorch and torchvision versions together to avoid
the import-time operator mismatch encountered with incompatible installations.
If you already have a different environment, install into the project virtual
environment rather than mixing system packages with the project dependencies.

## Prepare data

Place PDF inputs in `data/pdfs/`. Then run the pipeline in order:

```bash
python Extract_Pdf.py
python Clean_Text.py
python Training_Dataset.py
python Tokenize.py
```

The stages report an error if their required input directory has no files.
`Training_Dataset.py` splits cleaned page text on blank lines, skips paragraphs
shorter than 20 words, and reports an error if it produces no examples.

Each JSONL record has this shape:

```json
{
  "messages": [
    {"role": "user", "content": "Explain the following educational content clearly.\n\n..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {"source": "book.pdf", "page": 1}
}
```

The tokenizer applies the Qwen3 chat template and truncates each example to 128
tokens. `Tokenize.py` stages the replacement dataset before swapping it into
`data/tokenized/`, so a failed rebuild leaves the previous cache available.

## Validate the data and environment

After preparing data, run:

```bash
python test.py
```

The full check verifies the Python version, required imports, model identifier,
JSONL structure, tokenized dataset shape, token ID range, and training sequence
limit. It loads the tokenizer, so the first run may need network access.

To validate the environment and model identifier before preparing data:

```bash
python test.py --skip-data
```

The normal check also prevents the earlier embedding-index failure: it checks
that every cached token ID fits the tokenizer and rejects stale or incompatible
tokenized data.

## Train

Start training from the repository root:

```bash
python fine_tuneing.py
```

The training configuration is deliberately CPU-only:

- Loads the base model in float32 and explicitly places it on the CPU.
- Uses LoRA with rank 16, alpha 32, dropout 0.05, and attention/MLP projection
  modules as targets.
- Trains for one epoch with batch size 1 and gradient accumulation of 8.
- Uses a 128-token maximum sequence length and non-reentrant gradient
  checkpointing to reduce activation memory.
- Disables fp16/bf16 and pinned-memory data loading.
- Aligns the model and generation configuration's BOS/EOS/PAD IDs to the
  tokenizer before training.

CPU fine-tuning is slow. If the operating system prints `Killed`, it terminated
the process, usually because memory was exhausted. The code already applies
gradient checkpointing and a short sequence limit; close other memory-heavy
applications before trying again. Do not increase the sequence length or batch
size on a memory-constrained machine.

## Evaluate

Evaluation must use examples that were not included in the model's training
data. The current codebase does not generate a held-out split automatically.
Create `data/evaluation.jsonl` with one JSON object per line, using the same
`messages` format as the training data and an assistant answer as the final
message. Keep this file separate from `data/processed/training.jsonl`.

For example, use a prompt and reference answer from source material that was
not used to train the model:

```json
{"messages":[{"role":"user","content":"Explain this educational content clearly.\n\n[held-out passage]"},{"role":"assistant","content":"[reference answer]"}],"metadata":{"source":"held-out-source.pdf","page":1}}
```

Run the evaluator from the repository root after the merged model exists:

```bash
python Evaluate_Model.py
```

By default it loads `models/qwen3-0.6b-merged/`, reads
`data/evaluation.jsonl`, and writes `models/evaluation/summary.json` and
`models/evaluation/predictions.jsonl`. The summary reports mean negative
log-likelihood and perplexity for reference answer tokens, plus normalized
exact-match rate for greedy generated answers. The predictions file includes
each reference and generated answer for qualitative review.

To compare the fine-tuned model with its base checkpoint, evaluate both against
the same held-out file and use separate output directories:

```bash
python Evaluate_Model.py --output-dir models/evaluation-finetuned
python Evaluate_Model.py --model Qwen/Qwen3-0.6B --output-dir models/evaluation-base
```

The first base-model run downloads the checkpoint if it is not cached. Exact
match is a strict copying metric, not a measure of semantic equivalence. Review
the saved predictions and interpret perplexity in light of this project's
paragraph-copy training objective.

## Outputs and cleanup

Successful training writes:

- `models/qwen3-0.6b-lora/` — PEFT LoRA adapter plus tokenizer files.
- `models/qwen3-0.6b-merged/` — merged Hugging Face model and tokenizer, ready
  to convert to GGUF for Ollama.

The model files are saved in Hugging Face format. Convert the merged model
directory to GGUF for Ollama; the adapter directory alone is not a complete
model.

## Deploy with Ollama

Install Ollama for Windows or Linux from the [official download page](https://ollama.com/download),
and make sure its service is running. On Windows, importing the Safetensors
directory directly may fail because Ollama reports that the MLX runtime is
unavailable. Convert the merged model to GGUF first with llama.cpp. Skip the
conversion if `models/qwen3-0.6b-f16.gguf` already exists.

### Convert the model

If llama.cpp is not already available under `tools/llama.cpp`, clone it and
install the converter dependencies in an isolated uv environment.

Linux:

```bash
git clone https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
uv venv --python 3.12 tools/llama.cpp/.venv
cd tools/llama.cpp
uv pip install --python .venv/bin/python -r requirements/requirements-convert_hf_to_gguf.txt
cd ../..
tools/llama.cpp/.venv/bin/python tools/llama.cpp/convert_hf_to_gguf.py models/qwen3-0.6b-merged --outfile models/qwen3-0.6b-f16.gguf --outtype f16
```

Windows PowerShell:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
uv venv --python 3.12 tools/llama.cpp/.venv
Set-Location tools/llama.cpp
uv pip install --python .venv\Scripts\python.exe -r requirements\requirements-convert_hf_to_gguf.txt
Set-Location ..\..
& .\tools\llama.cpp\.venv\Scripts\python.exe .\tools\llama.cpp\convert_hf_to_gguf.py .\models\qwen3-0.6b-merged --outfile .\models\qwen3-0.6b-f16.gguf --outtype f16
```

The conversion creates an F16 GGUF. It is not quantized, so it uses more disk
space and memory than a quantized version. Ollama imports GGUF as provided.

### Create and run locally

The included `Modelfile` points to `models/qwen3-0.6b-f16.gguf`.

```bash
ollama create edulora -f Modelfile
ollama run edulora
```

### Publish or pull from your Ollama account

To publish under your Ollama account, add your Ollama public key in the account
settings, then tag and push the local model. Replace `<username>` with your
Ollama username:

```bash
ollama cp edulora <username>/edulora
ollama push <username>/edulora
```

Other users can install and run the published model with:

```bash
ollama run <username>/edulora
```

See Ollama's [model import and sharing guide](https://docs.ollama.com/import)
for account key and publishing details. Ollama's local API is available at
`http://localhost:11434` while its service is running.

Generated intermediate data and model outputs are ignored by Git. PDFs in
`data/pdfs/` are source inputs and are not regenerated by the pipeline. Raw,
cleaned, processed, tokenized, adapter, merged, and GGUF artifacts can be
rebuilt from their upstream inputs, but may take time, disk space, or network
access; do not delete them just to clean the repository unless you are ready to
rebuild them. The virtual environment can be recreated from `requirements.txt`.

## Troubleshooting

### `IndexError: index out of range in self`

The tokenized cache was created with token IDs that do not fit the active
model/tokenizer. Rebuild it and validate:

```bash
python Tokenize.py
python test.py
```

### The training process ends with `Killed`

This is an operating-system process termination rather than a Python exception;
on a small-memory machine it is commonly an out-of-memory kill. Keep the
configured batch size and 128-token sequence limit, close other large
applications, and ensure there is adequate available memory and swap. Swap may
avoid an immediate kill but can make CPU training substantially slower.

### `MLX runtime is not available` during Ollama import

Do not import the Safetensors directory directly on Windows. Convert
`models/qwen3-0.6b-merged/` to GGUF using the deployment steps above and point the
`Modelfile` at the resulting `.gguf` file. The LoRA directory contains only the
adapter and is not the model to convert.

### Missing package/import error

Reinstall the declared dependencies into the project environment:

Linux:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Windows PowerShell or Command Prompt:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

The PyTorch `KernelPreference` / `ScaleCalculationMode` Enum deprecation
warnings are emitted by the installed PyTorch stack and are not, by themselves,
training failures.
