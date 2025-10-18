---
title: Pepe Meme Generator
emoji: 🐸
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: src/app.py
python_version: "3.11"
---

<div align="center">

# 🐸 Pepe the Frog AI Meme Generator

### Create custom Pepe memes using AI-powered Stable Diffusion with LoRA fine-tuning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/MJaheen/Pepe_The_Frog_model_v1_lora)

[Demo](https://huggingface.co/spaces/MJaheen/Pepe-Meme-Generator) • [Documentation](./docs/) • [Training Guide](./docs/TRAINING.md) • [Report Bug](https://github.com/YOUR_USERNAME/pepe-meme-generator/issues)

</div>

---

## 📖 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Information](#-model-information)
- [Performance Optimization](#-performance-optimization)
- [Project Structure](#-project-structure)
- [Training](#-training-your-own-model)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🎨 **Multiple AI Models**
- **Pepe Fine-tuned LoRA** - Custom trained on Pepe dataset (1600 steps)
- **Pepe + LCM (FAST)** - 8x faster generation with LCM technology
- **Tiny SD** - Lightweight model for faster CPU generation
- **Small SD** - Balanced speed and quality
- **Base SD 1.5** - Standard Stable Diffusion
- **Dreamlike Photoreal 2.0** - Photorealistic style
- **Openjourney v4** - Artistic Midjourney-inspired style

### ⚡ **Performance Features**
- **LCM Support**: Generate images in 6 steps (~30 seconds on CPU)
- **GPU Acceleration**: Automatic CUDA detection with xformers support
- **Memory Efficient**: Attention slicing and VAE slicing enabled

### 🎭 **Generation Features**
- **Style Presets**: Happy, sad, smug, angry, crying, and more
- **Raw Prompt Mode**: Use exact prompts without automatic enhancements
- **Text Overlays**: Add meme text with Impact font
- **Batch Generation**: Create multiple variations
- **Progress Tracking**: Real-time generation progress bar
- **Seed Control**: Reproducible generations with fixed seeds
- **Gallery System**: View and manage all generated memes

### 🎯 **User Experience**
- **Model Hot-Swapping**: Switch models without restart
- **Interactive UI**: Clean Streamlit interface
- **Example Prompts**: Built-in inspiration gallery
- **Download Support**: Save images with one click
- **Responsive Design**: Works on desktop and mobile

---

## 🚀 Quick Start

### Try Online (No Installation)

🌐 **[Open in Hugging Face Spaces](https://huggingface.co/spaces/MJaheen/Pepe-Meme-Generator)** - Run instantly in your browser!

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/pepe-meme-generator.git
cd pepe-meme-generator

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📦 Installation

### System Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (NVIDIA with CUDA for faster generation)
- **Storage**: ~5GB for models and dependencies

### Dependencies

```bash
# Core dependencies
pip install torch torchvision  # PyTorch
pip install diffusers transformers accelerate  # Diffusion models
pip install streamlit  # Web interface
pip install pillow numpy scipy  # Image processing
pip install peft safetensors  # LoRA support
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### GPU Setup (Optional but Recommended)

For NVIDIA GPUs with CUDA:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install xformers for memory-efficient attention
pip install xformers
```

---

## 🎮 Usage

### Basic Usage

1. **Select a Model**: Choose from the dropdown (try "Pepe + LCM (FAST)" for speed)
2. **Enter a Prompt**: e.g., "pepe the frog as a wizard casting spells"
3. **Adjust Settings**: Steps (6 for LCM, 25 for normal), guidance scale, etc.
4. **Generate**: Click "Generate Meme" and wait
5. **Download**: Save your creation!

### Example Prompts

```
pepe_style_frog, wizard casting magical spells, detailed
pepe_style_frog, programmer coding on laptop, cyberpunk style
pepe_style_frog, drinking coffee at sunrise, peaceful
pepe_style_frog, wearing sunglasses, smug expression
pepe_style_frog, crying with rain, emotional, dramatic lighting
```

### Advanced Features

#### **Using LCM for Fast Generation**
1. Select "Pepe + LCM (FAST)" model
2. Use 6 steps (optimal for LCM)
3. Set guidance scale to 1.5
4. Generate in ~30 seconds!

#### **Adding Text Overlays**
1. Expand "Add Text" section
2. Enter top and bottom text
3. Text automatically styled in Impact font
4. Signature "MJ" added to corner

#### **Reproducible Generations**
1. Enable "Fixed Seed" in Advanced Settings
2. Set a seed number (e.g., 42)
3. Same seed + prompt = same image

---

## 🤖 Model Information

### Fine-Tuned LoRA Model

**Model ID**: `MJaheen/Pepe_The_Frog_model_v1_lora`

**Training Details**:
- **Base Model**: Stable Diffusion v1.5
- **Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: [iresidentevil/pepe_the_frog](https://huggingface.co/datasets/iresidentevil/pepe_the_frog)
- **Training Steps**: 2000
- **Resolution**: 512x512
- **Batch Size**: 1 (4 gradient accumulation)
- **Learning Rate**: 1e-4 (cosine schedule)
- **LoRA Rank**: 16
- **Precision**: Mixed FP16
- **Trigger Word**: `pepe_style_frog`

**Performance**:
- Quality: ⭐⭐⭐ (Good)
- Speed (CPU): ~4 minutes (25 steps)
- Speed (GPU): ~15 seconds (25 steps)

---

## 📁 Project Structure

```
pepe-meme-generator/
├── src/                          # Source code
│   ├── app.py                    # Main Streamlit application
│   ├── model/                    # Model management
│   │   ├── __init__.py
│   │   ├── config.py             # Model configurations
│   │   └── generator.py          # Image generation logic
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── image_processor.py    # Image processing utilities
├── docs/                         # Documentation
│   └──TRAINING.md               # Model training guide
├── models/                       # Downloaded models (gitignored)
├── outputs/                      # Generated images (gitignored)
├── scripts/                      # Utility scripts
├── tests/                        # Test files
├── diffusion_model_finetuning.ipynb  # Training notebook
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── .dockerignore                 # Docker ignore rules
├── Dockerfile                    # Docker configuration
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 🎓 Training Your Own Model

Want to fine-tune your own Pepe model or create a different character?

### Quick Training Overview

```bash
# 1. Prepare your dataset (images + captions)
# 2. Run the training script
accelerator launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="./your-data" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --output_dir="./output" \
  --rank=16
```

### Complete Training Guide

See **[docs/TRAINING.md] for:
- Dataset preparation
- Training configuration
- Hyperparameter tuning
- Validation and testing
- Model upload to Hugging Face

Or check out the **[diffusion_model_finetuning.ipynb](./diffusion_model_finetuning.ipynb)** notebook!

---

## 🛠️ Technology Stack

### Core Technologies
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Diffusers](https://github.com/huggingface/diffusers)** - Diffusion models library
- **[Transformers](https://github.com/huggingface/transformers)** - NLP models
- **[PEFT](https://github.com/huggingface/peft)** - Parameter-efficient fine-tuning (LoRA)
- **[Streamlit](https://streamlit.io/)** - Web UI framework

### AI/ML Components
- **Stable Diffusion 1.5** - Base diffusion model
- **LoRA** - Low-Rank Adaptation for efficient fine-tuning
- **LCM** - Latent Consistency Model for fast inference
- **DPM Solver** - Fast diffusion sampling

### Image Processing
- **Pillow (PIL)** - Image manipulation
- **NumPy** - Numerical operations
- **SciPy** - Scientific computing

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🎨 Add new style presets
- ⚡ Optimize performance
- 🧪 Add tests

---

### Development Setup

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/pepe-meme-generator.git
cd pepe-meme-generator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make your changes
# Test locally
streamlit run src/app.py
```

# Submit pull request

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Out of memory error  
**Solution**: Reduce resolution to 512x512, use CPU mode, or enable memory optimizations

**Issue**: Slow generation on CPU  
**Solution**: Use "Pepe + LCM (FAST)" model with 6 steps

**Issue**: Model not loading  
**Solution**: Clear Streamlit cache with "Clear Cache & Reload" button

**Issue**: Import errors  
**Solution**: Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`


---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Model Licenses
- **Stable Diffusion 1.5**: CreativeML Open RAIL-M License
- **Pepe LoRA**: MIT License
- **Training Dataset**: Check [iresidentevil/pepe_the_frog](https://huggingface.co/datasets/iresidentevil/pepe_the_frog)

---

## 🙏 Acknowledgments

### Special Thanks
- **[WorldQuant University](https://www.wqu.edu/ai-lab-computer-vision)** - AI/ML education and resources
- **[Hugging Face](https://huggingface.co/)** - Model hosting and diffusers library
- **[Stability AI](https://stability.ai/)** - Stable Diffusion model
- **[Microsoft](https://github.com/microsoft/LoRA)** - LoRA technique
- **[iresidentevil](https://huggingface.co/iresidentevil)** - Pepe dataset

---

## 📞 Contact & Support

- **Issues**: Mohamed.a.jaheen@gmail.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ star on GitHub!

---

<div align="center">

**Made with ❤️ by MJaheen**

*Generate Pepes responsibly! 🐸*

</div>