# NL-to-MQL: Natural Language to MongoDB Query Generator

A state-of-the-art small language model (SLM) that converts natural language requests into MongoDB MQL queries using LoRA fine-tuning, optimized for **Apple Silicon (MPS)**.

## 📋 Overview

This project fine-tunes **SmolLM3-3B** — HuggingFace's state-of-the-art 3-billion parameter SLM (2025) — using LoRA adapters to learn the mapping between natural language descriptions and MongoDB queries. SmolLM3-3B outperforms both Llama-3.2-3B and Qwen2.5-3B on reasoning benchmarks.


## Data Acquisition

The data was acquired from MongoDB Atlas Sample Dataset Benchmark available on Hugging Face: https://huggingface.co/datasets/mongodb-eai/natural-language-to-mongosh by [Ben Perlmutter](https://github.com/mongodben) from MongoDB Education Team.

### Key Features

- 🧠 **State-of-the-art**: SmolLM3-3B — best-in-class 3B parameter SLM
- 🍎 **Apple Silicon native**: Full MPS acceleration on M1/M2/M3/M4 chips
- ✨ **Lightweight adapters**: Only ~5-10MB of trainable LoRA parameters
- ⚡ **Multi-accelerator**: Auto-detects MPS → CUDA → CPU
- 📊 **Data-driven**: Trained on 766 real MongoDB query examples
- 🔧 **Simple CLI**: Easy train and inference commands
- 🎯 **Production-ready**: Generates valid MongoDB aggregation and find queries

## 🏗️ Project Structure

```
NLtoMQL/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── atlas_sample_data_benchmark.csv        # Training dataset (766 NL-MQL pairs)
├── atlas_sample_data_benchmark.flat.csv   # Original flat CSV format
├── DataProcessing.py                      # Full-featured trainer (HF Trainer class)
├── NLtoMQL_SLM.py                        # Lightweight trainer & inference (recommended)
├── models/
│   └── nl2mql-lora/                      # Trained adapter (after running train)
│       ├── adapter_model/                # LoRA weights
│       ├── config.json                   # Training metadata
│       └── tokenizer/                    # Tokenizer files
└── .venv/                                 # Python virtual environment
```

## 📦 Prerequisites

- **Python**: 3.10+
- **macOS**: Apple Silicon (M1/M2/M3/M4/M5) recommended
- **RAM**: 8GB minimum (16GB+ recommended for SmolLM3-3B)
- **GPU** (alternative): NVIDIA CUDA 11.8+
- **Storage**: ~6GB for model files (cached after first download)

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd /Users/rishi/Workspace/NLtoMQL
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

**Core packages:**
- `torch`: 2.1+ (with MPS backend for Apple Silicon)
- `transformers`: 4.45+
- `peft`: 0.7+
- `datasets`: 2.14+
- `pandas`: 2.0+
- `accelerate`: 0.25+

### 3. Verify Apple Silicon MPS

```bash
.venv/bin/python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"
```

Expected output on Apple Silicon:
```
PyTorch: 2.x.x
MPS available: True
MPS built: True
```

## 📚 Usage Guide

### Option A: Quick Training & Inference (Recommended)

Use `NLtoMQL_SLM.py` for the simplest workflow:

#### Step 1: Train the Adapter

```bash
.venv/bin/python NLtoMQL_SLM.py train \
  --csv atlas_sample_data_benchmark.csv \
  --epochs 3 \
  --lr 3e-4
```

**What happens:**
1. Auto-detects Apple Silicon MPS (or CUDA/CPU)
2. Downloads SmolLM3-3B base model (~6GB on first run, cached after)
3. Fine-tunes LoRA adapters on 766 training pairs
4. Saves weights to `models/nl2mql-lora/`

**Output:**
```
✓ Training complete! Adapter saved to models/nl2mql-lora
```

#### Step 2: Generate MQL from Natural Language

```bash
.venv/bin/python NLtoMQL_SLM.py infer \
  --nl "Find all users with age greater than 25 and sort by registration date"
```

**Example output:**
```
db.users.find({ "age": { $gt: 25 } }).sort({ "registration_date": -1 })
```

#### Step 3: Try More Examples

```bash
# Complex aggregation query
.venv/bin/python NLtoMQL_SLM.py infer \
  --nl "Calculate total revenue by product category for the last 30 days"

# Multi-stage pipeline
.venv/bin/python NLtoMQL_SLM.py infer \
  --nl "Show me the top 10 customers by total spend"

# Deterministic output (greedy decoding)
.venv/bin/python NLtoMQL_SLM.py infer \
  --nl "Get distinct product names where price < 100" \
  --temp 0.0
```

### Option B: Interactive Web UI (New!)

We've added a premium, dark-mode web application to easily interact with the trained model. It features syntax highlighting, history, and parameter tuning.

To start the UI server:

```bash
.venv/bin/python app.py
```

Then open `http://localhost:8000` in your web browser.

### Option C: Advanced Training (Full HF Trainer)

Use `DataProcessing.py` for production-grade training:

```bash
.venv/bin/python DataProcessing.py train \
  --csv atlas_sample_data_benchmark.csv \
  --output-dir models/nl2mql-lora \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-length 512 \
  --base-model HuggingFaceTB/SmolLM3-3B
```

Then infer:

```bash
.venv/bin/python DataProcessing.py infer \
  --adapter-dir models/nl2mql-lora \
  --nl-query "List all documents modified in the last week" \
  --max-new-tokens 220 \
  --temperature 0.0
```

## 🍎 Apple Silicon Optimization

This project is specifically optimized for Apple Silicon Macs:

| Feature | Implementation |
|---------|---------------|
| **Device detection** | Auto-detects MPS → CUDA → CPU |
| **Memory** | Uses unified memory architecture efficiently |
| **Dtype** | float32 on MPS (float16 not fully supported for training ops) |
| **Gradient clipping** | Enabled (max_norm=1.0) for MPS training stability |
| **Pin memory** | Disabled on MPS (only useful for CUDA) |

**Tips for Apple Silicon:**
- Close memory-heavy apps (browsers, Docker) during training
- 8GB Macs: use `--batch-size 1` and `--max-length 256`
- 16GB+ Macs: use `--batch-size 2` and `--max-length 512`
- Monitor memory: `Activity Monitor → Memory` tab

## 🔧 Configuration & Tuning

### Training Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--epochs` | 3 | 1-5 | More epochs = longer training, possibly better accuracy |
| `--lr` (learning rate) | 3e-4 | 1e-5 to 1e-3 | LoRA typically uses higher LR than full fine-tune |
| `--batch-size` | 1 | 1-4 | Larger = faster but needs more memory |
| `--max-length` | 512 | 256-1024 | Token limit; most queries fit in 256-512 |

### Inference Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--temp` | 0.2 | 0 = deterministic, 0.5-1.0 = more creative |
| `--max-tokens` | 200 | Length of generated MQL; 200 handles most queries |

## 📊 Dataset Overview

**File:** `atlas_sample_data_benchmark.csv`

**Columns:**
- `input.nlQuery`: Natural language question
- `expected.dbQuery`: Expected MongoDB query (MQL)
- Additional metadata: complexity, execution time, language, query operators

**Statistics:**
- **766 training pairs**
- Databases: sample_analytics, sample_mflix, sample_supplies
- Query types: find(), aggregate(), with various operators ($match, $group, $sort, $limit, etc.)

## ⚙️ Technical Details

### Model Architecture

```
Base Model: SmolLM3-3B (HuggingFaceTB/SmolLM3-3B)
├── Parameters: 3B (state-of-the-art SLM, 2025)
├── Context: 8192 tokens
├── Beats: Llama-3.2-3B, Qwen2.5-3B on benchmarks
└── Fine-tuning: LoRA (r=16, alpha=32)
    └── Trainable params: ~5-10MB (0.15% of base)
```

### LoRA Configuration

- **Rank (r)**: 16 (higher than TinyLlama's r=8 — better for 3B model)
- **Alpha (α)**: 32 (scaling factor = 2× rank)
- **Dropout**: 0.05 (regularization)
- **Target modules**: Attention layers (q_proj, v_proj)

### Device Selection Logic

```python
# Automatic accelerator detection:
MPS (Apple Silicon)  → float32  # Best for M1/M2/M3/M4
CUDA (NVIDIA GPU)    → float16  # Standard mixed-precision
CPU (fallback)       → float32  # Slowest but universal
```

## 📈 Training Duration & Resources

| Device | Time (3 epochs) | Memory | Notes |
|--------|-----------------|--------|-------|
| Apple M1/M2/M3/M4 (MPS) | 15-25 min | 8-12GB | **Recommended for Mac** |
| NVIDIA RTX 3060+ (CUDA) | 5-8 min | 10GB+ | Fastest option |
| Intel/ARM CPU | 60-90 min | 8GB+ | Slowest but works |

## 🔍 Troubleshooting

### MPS: RuntimeError or NaN values
```bash
# Use float32 (already the default for MPS in this project)
# Reduce batch size if memory issues:
.venv/bin/python NLtoMQL_SLM.py train --epochs 1
```

### Out of memory on 8GB Mac
```bash
.venv/bin/python NLtoMQL_SLM.py train --epochs 2
# Close browsers and other memory-heavy apps
```

### Model download fails
```bash
.venv/bin/python -c "from transformers import AutoModel; AutoModel.from_pretrained('HuggingFaceTB/SmolLM3-3B')"
```

### Poor quality MQL output
- Too few epochs → increase to 3-5
- Try simpler, clearer phrasing
- Use `--temp 0.0` for deterministic output

## Limitations
- The Model is trained on the dataset that has simple queires over the sample datasets of MongoDB. 
- The model is not trained on the complex queries and the queries that are not present in the dataset. 
- If asked a difficut query that involves complex aggregation, model will halucinate as it has not seen that before.

## 📋 Checklist: Getting Started

- [ ] Activate virtual environment: `source .venv/bin/activate`
- [ ] Install deps: `.venv/bin/pip install -r requirements.txt`
- [ ] Verify MPS: `.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"`
- [ ] Train adapter: `.venv/bin/python NLtoMQL_SLM.py train --epochs 3`
- [ ] Test inference: `.venv/bin/python NLtoMQL_SLM.py infer --nl "test query"`
- [ ] Verify `models/nl2mql-lora/` directory was created

## 📚 Resources

- [SmolLM3-3B on Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [MongoDB Query Language Docs](https://www.mongodb.com/docs/manual/reference/method/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Apple MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## 📄 License

This project uses:
- SmolLM3-3B (Apache 2.0)
- Hugging Face Transformers (Apache 2.0)
- PEFT (Apache 2.0)

## ❓ FAQ

**Q: Can I use a different base model?** </br>
A: Yes! Replace `--model` with any HuggingFace model ID:
```bash
.venv/bin/python NLtoMQL_SLM.py train --model microsoft/phi-4-mini-instruct --epochs 3
```

**Q: Why SmolLM3-3B over TinyLlama?** </br>
A: SmolLM3-3B is a 2025 model that significantly outperforms TinyLlama-1.1B on reasoning, code generation, and instruction following. It's 3× larger but still fits in 8GB unified memory on Apple Silicon.

**Q: How accurate is the generated MQL?** </br>
A: After 3 epochs with SmolLM3-3B, expect 80-90%+ valid MQL queries — a significant improvement over TinyLlama.

**Q: Can I add more training data?** </br>
A: Yes! Append rows to your CSV and retrain. The model benefits from more diverse examples.

**Q: What's the difference between the two Python scripts?** </br>
A: Both produce the same result. `NLtoMQL_SLM.py` is simpler (~250 lines, manual training loop), `DataProcessing.py` uses HF Trainer (gradient accumulation, checkpointing).

**Q: Can I deploy this on a web server?** </br>
A: Yes! Use FastAPI + the `generate_mql()` function from either script.

---

**Last updated:** March 10, 2026  
**Model:** SmolLM3-3B (HuggingFaceTB/SmolLM3-3B)  
