# Fine-Tuning Qwen3.5-4B with Unsloth Studio → Deploy to vLLM

> **Goal:** End-to-end guide — finetune Qwen3.5-4B with **QLoRA** (quick demo) or **LoRA 16-bit** (production) using **Unsloth Studio** (no-code, local), then serve it with **vLLM in Docker**.
> **Hardware:** 16GB VRAM / 64GB RAM, native **Windows 11**.
> **Updated:** 2026-06-01

---

## 0. The 30-second mental model

> Full finetuning updates all weights — too expensive for one GPU. **LoRA** freezes the base model and trains tiny low-rank adapter matrices (A·B) injected into attention/MLP layers, so you train ~0.1–1% of params. **QLoRA** goes further: it loads the *base* model in 4-bit (NF4) to slash VRAM, then trains the LoRA adapters in 16-bit on top. **Unsloth** makes this ~2x faster with ~70% less memory via custom Triton kernels. After training I either keep the adapter separate or **merge** it back into the base weights for serving with vLLM.

---

## 1. Model & VRAM decision (why Qwen3.5-4B)

Estimated VRAM for **bf16 LoRA** training (params × 2 bytes + training overhead):

| Model        | bf16 LoRA VRAM | Fits 16GB? |
|--------------|----------------|------------|
| 0.8B         | 3 GB           | ✅          |
| 2B           | 5 GB           | ✅          |
| **4B**       | **10 GB**      | ✅ **← use this** |
| 9B           | 22 GB          | ❌          |
| 27B / 35B-A3B| 56–74 GB       | ❌          |

**Decisions:**
- Use **`Qwen/Qwen3.5-4B`** (or `unsloth/Qwen3.5-4B`).
- Official docs recommend bf16 LoRA for Qwen3.5, but **QLoRA (4-bit)** is a good starting point — lower VRAM, quicker training. Trade-off: marginally lower quality than bf16 LoRA, acceptable for a demo or first run.
- Qwen3.6's smallest size is 27B → too big to finetune on 16GB. That's why we use 3.5-4B.

**LoRA vs QLoRA — when to pick which:**

| | LoRA (16-bit) | QLoRA (4-bit) |
|---|---|---|
| Base model precision | bf16 / fp16 | 4-bit NF4 quantized |
| VRAM for 4B | ~10 GB | ~5–6 GB |
| Training quality | Higher | Slightly lower |
| Speed | Slower (more VRAM pressure) | Faster (less data to move) |
| **When to use** | Production runs, final model | Quick experiments, limited VRAM, demos |

*~10GB estimate: 4B params × 2 bytes (bf16) ≈ 8GB weights + LoRA adapters + overhead.*

> **Real-world observation:** The figures above are model weights only — what Unsloth's docs quote as the baseline. In practice, a full LoRA run on Qwen3.5-4B (batch=2, seq=1024, rank=32, gradient checkpointing on) used **15.44 / 15.93 GB** on an RTX 5060 Ti. Training also loads activations, optimizer state, and gradient buffers on top of the model weights. If you have exactly 16GB, expect it to be nearly full.

---

## 2. Install Unsloth Studio (Windows PowerShell)

> ⚠️ Do **NOT** use `irm https://unsloth.ai/install.ps1 | iex` — that URL 301-redirects and `irm` misparses it as XML (`System.Xml.XmlDocument` error). Use the **direct raw URL** below.

**(Optional) Install to E: drive instead of default `%USERPROFILE%\.unsloth\studio`:**
```powershell
$env:UNSLOTH_STUDIO_HOME = "<your-preferred-path>\unsloth-studio"
```

**Install (correct command):**
```powershell
irm https://raw.githubusercontent.com/unslothai/unsloth/main/install.ps1 | iex
```

**If blocked by execution policy:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
irm https://raw.githubusercontent.com/unslothai/unsloth/main/install.ps1 | iex
```

**What the installer does:** detects/installs Python 3.11–3.13 (via winget), creates a `uv` venv, detects CUDA from `nvidia-smi` and installs matching PyTorch, installs `unsloth`, sets up Studio, creates shortcuts, and adds an `unsloth` command to PATH. Downloads several GB — give it time.

---

## 3. Launch Studio

```powershell
unsloth studio -H 0.0.0.0 -p 8888
```
Then open **http://localhost:8888**

> If `unsloth` is "not recognized" right after install → **close & reopen PowerShell** (PATH refresh), then retry.

---

## 4. Configure training in Studio (UI)

1. **Model:** `Qwen/Qwen3.5-4B`
2. **Training mode:** **QLoRA (4-bit)** for a quick first run, or **LoRA (16-bit)** for production quality. See settings tables below.
3. **Dataset:** Upload your `finetuning_dataset.jsonl` (drag-and-drop onto the upload box, or click Upload).

### Quick demo settings (QLoRA, small dataset)

> Use this for a first run to verify your setup works end-to-end. Trains in ~10 minutes.

| Parameter | Value |
|---|---|
| Method | QLoRA (4-bit) |
| Max Steps | 60 |
| Context Length | 2048 |
| Learning Rate | 2e-4 |
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0 |
| Target Modules | all 7 (default) |
| Batch Size | 2 |
| Grad Accum | 4 |
| Optimizer | AdamW 8-bit |
| LR Scheduler | Linear |
| Weight Decay | 0 |

---

### Production settings (LoRA 16-bit, 500-row dataset, 16GB VRAM)

> Full-precision LoRA run. Use this for a real training run on a ~500-entry dataset.

| Parameter | Value | Reason |
|---|---|---|
| Model | `Qwen/Qwen3.5-4B` | Base model; load via Unsloth for optimized kernels |
| Method | LoRA (16-bit, bf16) | No quantization noise; cleaner gradients for domain adaptation |
| Load in 4-bit | ❌ Off | This is what QLoRA uses — disabled here so the base model stays in full bf16 |
| Load in 8-bit | ❌ Off | A separate quantization mode (LLM.int8()) — also disabled; we want full bf16, not a halfway quantization |
| Context Length | 1024 | Base model takes ~8GB; shorter seq keeps activations within remaining ~8GB |
| Rank (r) | 32 | More adapter capacity now that VRAM isn't the bottleneck |
| Alpha | 32 | Equals r → neutral scaling (effective update scale = 1.0) |
| Dropout | 0.05 | Light regularization for 500-entry dataset; prevents memorization |
| Target Modules | all 7 (default) | MLP layers hold factual knowledge — skip them and domain adaptation suffers |
| Batch Size | 2 | VRAM constraint: ~8GB left after base model leaves no room for larger batch |
| Grad Accum | 16 | Effective batch = 2 × 16 = 32 — production target without extra VRAM |
| Packing | ❌ Off | Packing concatenates multiple short samples into one sequence to reduce padding waste. Useful when sequences are short and variable. Here, avg ~480 tokens fits ~2 samples per 1024-length sequence → 500 examples become ~250 packed sequences → half the gradient steps. Since our sequences are already consistent in length, packing saves no padding but costs us steps — disabled. |
| Epochs | 10 | Without packing: 500 / 2 / 16 × 10 = ~150 optimizer steps — solid for this dataset size |
| Learning Rate | 2e-4 | Standard for LoRA adapters |
| LR Scheduler | Cosine | Smoother decay than linear; standard for production SFT |
| Warmup Steps | 10 | Prevents large updates on step 1 when optimizer momentum is cold |
| Optimizer | AdamW 8-bit | Saves ~4GB vs 32-bit; negligible quality difference |
| Weight Decay | 0.01 | L2 regularization; standard for runs over 100 steps |
| max_grad_norm | 1.0 | Clips gradient norm if it spikes — QLoRA run had spiky grad norms without this; prevents destabilizing updates |
| Gradient Checkpointing | ✅ On | Unsloth enables by default; recomputes activations on the backward pass instead of storing them, trading compute for memory — what makes 16-bit viable on 16GB |
| Precision | bf16 | RTX 5060 Ti supports it; matches base model dtype |
| Eval Split | 90/10 | 450 train / 50 eval |
| Save Steps | 30 | 5 checkpoints across ~150 steps — recovery points if training crashes |
| Eval Steps | 15 | 1 eval per epoch (10 evals total) — aligned to epoch rhythm; gives early warning if eval loss diverges from train loss |
| Seed | 42 | Fixed seed for reproducibility |
| **Effective batch** | **32** | 2 × 16 |
| **Est. total steps** | **~150** | 500 × 0.9 / 2 / 16 × 10 epochs |
| **Est. time** | ~45–60 min | Slightly longer than QLoRA due to full-precision ops |

---

### Parameters explained — what each one does

---

#### Max Steps vs Epochs — what's the difference?

**Max Steps** counts the number of times the optimizer updates the model weights.
Each step processes one batch of data and performs one gradient update.

**Epoch** is one full pass through your entire dataset.
- If you have 1000 samples and batch size 2 with grad accum 4 → effective batch = 8 → 1 epoch = 125 steps.

**Which to use:**
- Use **Steps** when you want tight control and fast iteration (smoke tests, hyperparameter search). Easy to stop early.
- Use **Epochs** in production — it's the standard because it's dataset-size-agnostic. "Train for 3 epochs" means the model sees every sample 3 times regardless of dataset size.

| Scenario | Recommended control | Value |
|---|---|---|
| Quick smoke test / first run | Steps | 30–100 |
| Small dataset (<1k rows) demo | Epochs | 3–5 |
| **Production SFT (1k–100k rows)** | **Epochs** | **2–5** |
| Production with very large dataset | Steps with eval checkpoints | 500–5000+ |

---

#### Learning Rate

**What it is:** Controls how large each weight update is. Too high → loss spikes and model breaks. Too low → model barely learns.

**What it does:** Multiplied against the gradient at each step. The LR scheduler then decays this value over time so early steps learn fast and later steps fine-tune carefully.

| Mode | Recommended LR | Why |
|---|---|---|
| QLoRA / LoRA | 2e-4 (0.0002) | Adapters are small — can tolerate higher LR than full FT |
| Full fine-tune | 2e-5 | All weights at risk; must be conservative |
| Continued pretraining | 5e-5 | Mid-range — adjusting existing knowledge |

**Production standard:** Always pair with a warmup (5–10% of total steps) so the model doesn't get a large update on the very first step when the optimizer's momentum is cold.

---

#### Context Length

**What it is:** The maximum number of tokens in a single training sample. Samples longer than this are truncated; shorter ones are padded (or packed).

**What it does:** Directly determines VRAM usage — a longer context means a larger KV cache during the forward pass. Doubling context length roughly quadruples attention memory.

| Value | Use when | VRAM impact |
|---|---|---|
| 512 | Short Q&A, classification | Very low |
| 2048 | General instruction tuning (default) | Moderate |
| 4096 | Code, reasoning, long documents | High |
| 8192+ | Multi-turn conversations, RAG outputs | Very high — production multi-GPU only |

**Production standard:** Set to the 95th-percentile length of your actual data, not arbitrarily high. Padding waste and memory cost add up.

---

#### LoRA Rank (r)

**What it is:** The inner dimension of the two low-rank matrices A (d × r) and B (r × d) that replace the full weight update. Instead of updating d × d parameters, you only update 2 × d × r.

**What it does:** Higher rank = more expressive adapters = more parameters trained = more capacity to learn. But also more VRAM and slower training.

| Rank | Trainable params (rough) | Use when |
|---|---|---|
| 4–8 | ~0.05% of model | Style/tone only, very fast |
| **16** | **~0.1–0.2%** | **General default — good balance** |
| 32–64 | ~0.3–0.5% | Complex domain adaptation, large datasets |
| 128+ | ~1%+ | Near full-finetune territory; rarely needed with LoRA |

**Production standard:** Start at r=16. If eval loss plateaus early, bump to 32. If you're compute-rich, run a sweep.

---

#### LoRA Alpha

**What it is:** A scaling constant. The actual weight update applied to the model is scaled by `alpha / r`.

**What it does:** Controls the magnitude of the LoRA update relative to the original weights. If alpha > r, the adapter update is amplified; if alpha < r, it's dampened.

**Rule of thumb:**
- `alpha = r` → neutral scaling (most common, safe default)
- `alpha = 2r` → amplified updates, useful when you want the adapter to have stronger influence
- Never set alpha < r/2 — the adapter becomes too weak to meaningfully shift behaviour

**Production standard:** Keep alpha = r unless you have a specific reason. Don't overthink this one.

---

#### Dropout

**What it is:** Randomly zeroes a fraction of LoRA activations during each training step, forcing the model to not rely on any single pathway.

**What it does:** Prevents overfitting — the adapter can't memorise the training data if random neurons are switched off each step.

| Value | Use when |
|---|---|
| 0.00 | Small datasets (<5k rows), short runs — default and fine here |
| 0.05 | Medium datasets, if you see train loss dropping but eval loss rising |
| 0.10 | Larger datasets with noisy labels |

**Production standard:** Monitor train vs eval loss gap. If eval loss diverges from train loss, add dropout 0.05.

---

#### Target Modules

**What it is:** The list of weight matrices inside the transformer that get LoRA adapters injected.

**What it does:** More modules = more params trained = more expressive = more VRAM.

| Module set | What it covers | When to use |
|---|---|---|
| `q_proj, v_proj` | Attention query + value only | Minimal — style/tone changes only |
| `q_proj, k_proj, v_proj, o_proj` | Full attention | Good for instruction following |
| **All 7 (default)** | **Attention + MLP (gate, up, down)** | **Best for domain adaptation and knowledge** |

**Production standard:** Use all 7 for any serious finetuning. The MLP layers hold factual knowledge — skipping them limits how much the model can learn.

---

#### Batch Size & Gradient Accumulation

**Batch Size:** Number of samples processed in parallel per GPU per step. Higher = better gradient estimates, more stable training. Limited by VRAM.

**Gradient Accumulation:** Instead of updating weights after every batch, accumulate gradients over N batches and update once. Mathematically equivalent to a larger batch, without the VRAM cost.

```
Effective batch size = Batch Size × Gradient Accumulation
                     = 2 × 16 = 32  (production setting)
```

**Why effective batch size matters:** Too small (< 4) → noisy gradients, unstable training. Too large → model updates less frequently, can get stuck.

| Effective batch size | Use when |
|---|---|
| 4–8 | Single GPU, limited VRAM, small dataset |
| **16–32** | **Production standard for SFT** |
| 64–128 | Multi-GPU, large dataset, need stability |

**Production standard:** Target effective batch of 16–32. If your GPU only allows batch=1, set grad_accum=16 to compensate.

---

#### Optimizer — AdamW 8-bit

**What it is:** Adam with weight decay, where optimizer state (momentum + variance) is stored in 8-bit instead of 32-bit.

**What it does:** Reduces optimizer memory by ~4x with negligible quality loss. On a 4B model, this saves ~4GB VRAM.

| Optimizer | VRAM | Quality | Use when |
|---|---|---|---|
| AdamW 32-bit | High | Best | Multi-GPU production with headroom |
| **AdamW 8-bit** | **Low** | **Near-identical** | **Single GPU — always use this** |
| SGD | Very low | Worse | Rare; only for very constrained setups |

---

#### LR Scheduler

**What it is:** Controls how the learning rate changes over the course of training.

**What it does:** Prevents the model from taking large destructive steps late in training when it's close to a good solution.

| Scheduler | Behaviour | Use when |
|---|---|---|
| **Linear** | LR decays steadily to 0 | Short runs, default, fine for demos |
| **Cosine** | LR decays following a cosine curve — slower at first, fast in middle, slow at end | **Production standard** — smoother, better final quality |
| Constant | No decay | Rarely useful; good for debugging only |

**Production standard:** Use **cosine** with warmup. Linear is fine for quick demo runs.

---

#### Weight Decay

**What it is:** L2 regularisation applied directly to the model weights at each step, pulling them toward zero.

**What it does:** Prevents any single weight from becoming too large, which is a form of overfitting.

| Value | Use when |
|---|---|
| 0 | Short runs, small datasets — default |
| 0.01 | Standard production setting |
| 0.1 | Aggressive regularisation for noisy or very small datasets |

**Production standard:** Set to 0.01 for any run over 500 steps.

---

4. **Train** — watch live loss in the observability panel. Healthy training looks like: loss drops steeply in the first 10–20 steps, then gradually levels off. If it flatlines immediately → LR too low. If it spikes → LR too high.
5. **Eval** in the UI to sanity-check before exporting.

**What the dashboard looks like mid-run:**

![Training dashboard — train loss dropping, eval loss visible](screenshots/LoRa-Training.png)

> ⚠️ Watch both curves. If train loss keeps falling but eval loss starts rising, the model is overfitting — the best checkpoint is the one where eval loss was lowest, not the final one.

---

## 5. Export the finetuned model

**For vLLM → merged 16-bit safetensors (use this):**
```python
model.save_pretrained_merged("qwen3.5-finetuned", tokenizer, save_method="merged_16bit")
```

**Other options:**
```python
# GGUF (for llama.cpp / Ollama / CPU)
model.save_pretrained_gguf("directory", tokenizer, quantization_method="q4_k_m")

# LoRA adapters only (smallest; serve with vLLM --enable-lora)
model.save_pretrained("finetuned_lora")
```

Studio also has an Export option for these. Note the output folder path shown in the UI after export completes.

---

## 6. Deploy to vLLM in Docker

> ⚠️ **vLLM v0.16.0 does NOT support Qwen3.5.** Use **≥ 0.17.0 or nightly**.

**Prereqs (one-time):**
- Docker Desktop with **WSL2 backend** (Settings → General → "Use WSL 2 based engine")
- Recent NVIDIA driver. Verify GPU passthrough:
  ```powershell
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```

**Serve:**
```powershell
docker run --gpus all `
  -v <your-model-path>:/model `
  -p 8000:8000 `
  vllm/vllm-openai:v0.17.0 `
  --model /model `
  --max-model-len 4096 `
  --gpu-memory-utilization 0.90
```
> If `v0.17.0` won't pull, try `:latest` or `:nightly`.

**Test (OpenAI-compatible API):**
```powershell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\":\"/model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'
```

---

## 7. Comparing base vs fine-tuned output

After exporting, use **Compare in Chat** in Unsloth Studio to test both models side by side with the same prompt.

![Base model vs fine-tuned model — same question, different style](screenshots/Output-Comparison.jpg)

> The base model gives a generic textbook-style answer. The fine-tuned model reflects the style and focus of your training data. If they look identical, the fine-tuning didn't take — check your dataset format and eval loss curve.

---

## 8. Troubleshooting cheat sheet

| Symptom | Fix |
|---|---|
| `System.Xml.XmlDocument` on install | Use the raw GitHub URL (Step 2), not `unsloth.ai/install.ps1`. |
| Execution policy error | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then retry. |
| `unsloth` not recognized | Reopen PowerShell (PATH refresh). |
| OOM during training | max seq len → 1024, batch=1. QLoRA uses less VRAM than LoRA — switch to QLoRA if still OOM. |
| Docker can't see GPU | Enable WSL2 backend; run the `nvidia-smi` Docker test. |
| vLLM "unknown architecture qwen3.5" | vLLM too old → bump tag to ≥ 0.17.0 / nightly. |
| Dataset format rejected | Ensure a single `text` column; reshape rows accordingly. |

---

## 9. Concept quick reference

| Question | Crisp answer |
|---|---|
| LoRA vs QLoRA? | LoRA = low-rank adapters on a frozen full-precision base. QLoRA = same but base is 4-bit quantized → less VRAM. |
| What is rank `r`? | Dimension of the low-rank decomposition; higher r = more capacity + more params. 8–32 typical. |
| What's `lora_alpha`? | Scaling factor; effective update ≈ (alpha/r)·BA. |
| Why merge for vLLM? | Merged weights serve with no adapter overhead + full kernel optimization. (Can also serve adapters live via `--enable-lora`.) |
| Why does Unsloth save memory/time? | Custom Triton kernels, fused ops, smart gradient checkpointing — no accuracy loss. |
| When NOT to finetune? | If RAG/prompting solves it. Finetune for behavior/format/style/domain, not for injecting fresh facts. |
| SFT vs RLHF/DPO? | SFT = supervised imitation of good answers. DPO/RLHF = align to ranked preferences. |
| Why vLLM over plain HF? | PagedAttention + continuous batching → far higher throughput; OpenAI-compatible server. |
| Catastrophic forgetting? | Risk with full FT; LoRA mitigates it since base is frozen. |

---

## Sources
- Qwen3.5 fine-tune guide: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Qwen3.5 model docs: https://unsloth.ai/docs/models/qwen3.5
- Windows install: https://unsloth.ai/docs/get-started/install/windows-installation
- Installer script: https://raw.githubusercontent.com/unslothai/unsloth/main/install.ps1
- Unsloth Studio (GitHub): https://github.com/unslothai/unsloth
