# 🎓 Model Training Guide

This guide covers how to fine-tune your own Stable Diffusion model using LoRA (Low-Rank Adaptation) for creating custom character models like our Pepe generator.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Running the Training](#running-the-training)
- [Model Upload](#model-upload)


---

## 🎯 Overview

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:
- ✅ Trains only a small fraction of parameters (~0.5% of full model)
- ✅ Requires significantly less VRAM (~10GB vs 40GB+)
- ✅ Maintains base model quality while adding custom styles
- ✅ Produces small, portable adapter files (~100MB vs 4GB+)
- ✅ Can be combined with other LoRAs

### Our Training Setup

**Model**: Pepe the Frog LoRA  
**Base**: Stable Diffusion v1.5  
**Dataset**: [iresidentevil/pepe_the_frog](https://huggingface.co/datasets/iresidentevil/pepe_the_frog)  
**Result**: [MJaheen/Pepe_The_Frog_model_v1_lora](https://huggingface.co/MJaheen/Pepe_The_Frog_model_v1_lora)  
**Training Time**: ~2-3 hours on T4 GPU (Google Colab)

---

## 🛠️ Prerequisites

### Hardware Requirements

**Minimum**:
- GPU: NVIDIA GPU with 10GB+ VRAM (e.g., RTX 3080, T4)
- RAM: 16GB system RAM
- Storage: 20GB free space

**Recommended**:
- GPU: NVIDIA A100, V100, or RTX 4090
- RAM: 32GB system RAM
- Storage: 50GB+ SSD


### Software Requirements

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.31.0
pip install transformers==4.45.1
pip install accelerate==0.34.2
pip install peft>=0.11.0
pip install safetensors==0.4.4
pip install datasets
pip install bitsandbytes  # For 8-bit Adam optimizer (optional)
```

---

## 📂 Dataset Preparation

### Dataset Structure

Your dataset should follow this structure:

```
dataset/
├── image_1.png
├── image_2.png
├── image_3.png
└── metadata.jsonl  # or metadata.csv
```

### Metadata Format

**Option 1: JSONL (Recommended)**

```jsonl
{"file_name": "image_1.png", "prompt": "pepe_style_frog, happy pepe smiling"}
{"file_name": "image_2.png", "prompt": "pepe_style_frog, sad pepe crying"}
{"file_name": "image_3.png", "prompt": "pepe_style_frog, pepe drinking coffee"}
```

**Option 2: CSV**

```csv
file_name,prompt
image_1.png,"pepe_style_frog, happy pepe smiling"
image_2.png,"pepe_style_frog, sad pepe crying"
image_3.png,"pepe_style_frog, pepe drinking coffee"
```

### Dataset Best Practices

1. **Image Quality**
   - Resolution: 512x512 or higher
   - Format: PNG or JPG
   - Clear, well-lit images
   - Varied poses and expressions

2. **Caption Quality**
   - Include trigger word (e.g., `pepe_style_frog`)
   - Describe key features and actions
   - Be consistent in naming conventions
   - 5-15 words per caption optimal

3. **Dataset Size**
   - Minimum: 20-50 images
   - Optimal: 100-500 images
   - More images = better generalization

4. **Diversity**
   - Various angles and poses
   - Different expressions
   - Multiple backgrounds
   - Different lighting conditions

### Our Pepe Dataset

We used **[iresidentevil/pepe_the_frog](https://huggingface.co/datasets/iresidentevil/pepe_the_frog)** which contains:
- ~200 high-quality Pepe images
- Consistent 512x512 resolution
- Varied expressions and styles
- Pre-captioned with trigger word

---

## ⚙️ Training Configuration

### Training Hyperparameters

Here's the exact configuration we used for the Pepe model:

```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/path/to/pepe-data" \
  --caption_column="prompt" \
  --image_column="image" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="./output" \
  --rank=16 \
  --validation_prompt="pepe_style_frog, a high-quality, detailed image of pepe the frog smiling and holding a cup of coffee at sunrise" \
  --validation_epochs=5 \
  --seed=42 \
  --mixed_precision="fp16" \
  --checkpointing_steps=150
```

### Parameter Explanation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pretrained_model_name_or_path` | `runwayml/stable-diffusion-v1-5` | Base model to fine-tune |
| `train_data_dir` | `/path/to/data` | Path to your dataset |
| `resolution` | `512` | Image resolution (512x512) |
| `train_batch_size` | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | `4` | Effective batch size = 1 * 4 = 4 |
| `max_train_steps` | `2000` | Total training steps |
| `learning_rate` | `1e-4` | Initial learning rate |
| `lr_scheduler` | `cosine` | Learning rate schedule |
| `rank` | `16` | LoRA rank (higher = more parameters) |
| `mixed_precision` | `fp16` | Use 16-bit precision for speed |
| `checkpointing_steps` | `150` | Save checkpoint every N steps |

### Hyperparameter Tuning Tips

**Learning Rate**:
- Too high: Training unstable, poor quality
- Too low: Slow convergence, underfitting
- Recommended: `1e-4` to `1e-5`

**LoRA Rank**:
- Lower (4-8): Faster training, smaller files, less expressive
- Medium (16-32): Balanced (recommended)
- Higher (64-128): More expressive, larger files, risk of overfitting

**Training Steps**:
- Small dataset (20-50 images): 500-1000 steps
- Medium dataset (50-200 images): 1000-2000 steps
- Large dataset (200+ images): 2000-5000 steps

**Batch Size**:
- Depends on VRAM availability
- Effective batch size = `batch_size × gradient_accumulation_steps`
- Recommended effective batch size: 4-8

---

## 🚀 Running the Training

### Option 1: Google Colab (Recommended for Beginners)

1. **Open the Notebook**:
   - Use our provided notebook: `diffusion_model_finetuning.ipynb`
   - Or create new Colab notebook

2. **Setup GPU**:
   ```
   Runtime → Change runtime type → GPU (T4)
   ```

3. **Mount Google Drive** (optional):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Install Dependencies**:
   ```python
   !pip install -q diffusers transformers accelerate peft
   ```

5. **Upload Dataset**:
   - Upload to Google Drive
   - Or download from Hugging Face

6. **Run Training**:
   ```python
   !accelerate launch train_text_to_image_lora.py \
     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
     --train_data_dir="/content/drive/MyDrive/pepe-data" \
     --max_train_steps=2000 \
     --learning_rate=1e-4 \
     --output_dir="./output"
   ```

7. **Monitor Progress**:
   - Watch loss decrease
   - Check validation images
   - Save checkpoints to Drive


### Generate test image
image = pipe("pepe_style_frog, wizard casting spells").images[0]
image.save("validation.png")
```


## 📤 Model Upload

### Prepare for Upload

1. **Test Locally**:
   ```python
   from diffusers import StableDiffusionPipeline
   
   pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
   pipe.load_lora_weights("./output")
   
   # Test
   image = pipe("pepe_style_frog, happy pepe").images[0]
   image.save("test.png")
   ```

2. **Prepare Files**:
   ```
   output/
   ├── pytorch_lora_weights.safetensors  # Main file
   ├── README.md  # Model card
   └── sample_images/  # Example outputs
   ```

### Upload to Hugging Face

1. **Install Hub CLI**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Create Model Card** (`README.md`):
   ```markdown
   ---
   license: creativeml-openrail-m
   base_model: runwayml/stable-diffusion-v1-5
   tags:
   - stable-diffusion
   - lora
   - text-to-image
   ---
   
   # Pepe LoRA Model
   
   Fine-tuned LoRA for generating Pepe the Frog images.
   
   ## Usage
   ```python
   from diffusers import StableDiffusionPipeline
   
   pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
   pipe.load_lora_weights("YOUR_USERNAME/your-model-name")
   
   image = pipe("pepe_style_frog, happy pepe").images[0]
   ```
   ```

3. **Upload**:
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   api.create_repo("YOUR_USERNAME/pepe-lora", repo_type="model")
   api.upload_folder(
       folder_path="./output",
       repo_id="YOUR_USERNAME/pepe-lora",
       repo_type="model"
   )
   ```


### Common Issues

**Out of Memory**:
- Reduce `train_batch_size` to 1
- Enable `--gradient_checkpointing`
- Use `--mixed_precision="fp16"`
- Reduce image resolution
