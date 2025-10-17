---
title: Pepe Meme Generator
emoji: 🐸
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: src/app.py
python_version: "3.10"
---

# 🐸 Pepe the Frog Meme Generator

AI-powered meme generator using Stable Diffusion and LoRA fine-tuning.

---

## 🎮 Try It Online

🚀 **[Open in Hugging Face Spaces](https://huggingface.co/spaces/MJaheen/Pepe-Meme-Generator)**

---

## 🌟 Features

- Generate **custom Pepe memes** from text prompts  
- Multiple **style presets** (happy, sad, smug, angry, etc.)  
- **Add meme text overlays** and download results  
- Adjustable generation parameters (CFG, steps, seed, etc.)  
- Batch generation and meme gallery system  

---

## 💡 Example Prompts

- "pepe the frog as a wizard"  
- "pepe coding on a laptop"  
- "pepe drinking coffee"  
- "smug pepe wearing sunglasses"

---

## 🚀 Quick Start (GitHub)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/pepe-meme-generator.git
cd pepe-meme-generator

# Install
pip install -r requirements.txt

# Run
streamlit run src/app.py
```

---

## 📚 Project Structure

pepe-meme-generator/
├── src/
│   ├── app.py              # Main Streamlit app
│   ├── model/
│   │   ├── generator.py    # Generation logic
│   │   └── config.py       # Model configuration
│   └── utils/
│       └── image_processor.py
├── models/                 # Model weights (not committed)
├── outputs/                # Generated memes (not committed)
├── requirements.txt
├── .gitignore
└── README.md

---

## 🛠️ Tech Stack

Model: Stable Diffusion 1.5 + LoRA

Framework: PyTorch, Diffusers

UI: Streamlit

Processing: PIL, OpenCV

---



This project demonstrates:
- Diffusion model architecture
- Transfer learning with LoRA
- Text-to-image synthesis

---
## 🎓 🙏 Acknowledgments

- [WorldQuant](https://www.wqu.edu/ai-lab-computer-vision)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [LoRA](https://github.com/microsoft/LoRA)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Streamlit](https://github.com/streamlit/streamlit)


## 📜 License

MIT License — see LICENSE file.