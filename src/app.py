"""Pepe the Frog Meme Generator - Main Application"""

import streamlit as st
from PIL import Image
import io
from datetime import datetime

# Import our modules
from model.generator import PepeGenerator
from model.config import ModelConfig
from utils.image_processor import ImageProcessor

# Page config
st.set_page_config(
    page_title="🐸 Pepe Meme Generator",
    page_icon="🐸",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0


@st.cache_resource
def load_generator():
    """Load and cache the generator"""
    return PepeGenerator()


def get_example_prompts():
    """Return example prompts"""
    return [
        "pepe the frog as a wizard casting spells",
        "pepe the frog coding on a laptop",
        "pepe the frog drinking coffee",
        "pepe the frog as a superhero",
        "pepe the frog reading a book",
    ]


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.title("🐸 Pepe the Frog Meme Generator")
    st.markdown("Create custom Pepe memes using AI! Powered by Stable Diffusion.")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Style selection
    style_options = {
        "Default": "default",
        "😊 Happy": "happy",
        "😢 Sad": "sad",
        "😏 Smug": "smug",
        "😠 Angry": "angry",
        "🤔 Thinking": "thinking",
        "😲 Surprised": "surprised",
    }
    
    selected_style = st.sidebar.selectbox(
        "Choose Style",
        list(style_options.keys())
    )
    style = style_options[selected_style]
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        steps = st.slider("Steps", 20, 100, 50, 5)
        guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        use_seed = st.checkbox("Fixed Seed")
        seed = st.number_input("Seed", 0, 999999, 42) if use_seed else None
    
    # Text overlay settings
    with st.sidebar.expander("💬 Add Text"):
        add_text = st.checkbox("Add Meme Text")
        top_text = st.text_input("Top Text") if add_text else ""
        bottom_text = st.text_input("Bottom Text") if add_text else ""
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✏️ Create Your Meme")
        
        # Prompt input
        prompt = st.text_area(
            "Describe your meme",
            height=100,
            placeholder="e.g., pepe the frog celebrating victory"
        )
        
        # Examples
        with st.expander("💡 Example Prompts"):
            for example in get_example_prompts():
                st.write(f"• {example}")
        
        # Generate button
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            generate = st.button("🎨 Generate Meme", type="primary")
        with col_btn2:
            num_vars = st.number_input("Variations", 1, 4, 1)
    
    with col2:
        st.subheader("🖼️ Generated Meme")
        placeholder = st.empty()
        
        if st.session_state.generated_images:
            placeholder.image(
                st.session_state.generated_images[-1],
                use_container_width=True
            )
        else:
            placeholder.info("Your meme will appear here...")
    
    # Generate
    if generate and prompt:
        try:
            generator = load_generator()
            
            progress = st.progress(0)
            status = st.empty()
            
            for i in range(num_vars):
                status.text(f"Generating {i+1}/{num_vars}...")
                progress.progress((i + 1) / num_vars)
                
                # Generate
                image = generator.generate(
                    prompt=prompt,
                    style=style,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed
                )
                
                # Add text if requested
                if add_text and (top_text or bottom_text):
                    processor = ImageProcessor()
                    image = processor.add_meme_text(image, top_text, bottom_text)
                
                st.session_state.generated_images.append(image)
                st.session_state.generation_count += 1
            
            progress.empty()
            status.empty()
            
            st.success("✅ Meme generated!")
            
            # Show result
            if num_vars == 1:
                placeholder.image(image, use_container_width=True)
                
                # Download
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download",
                    buf.getvalue(),
                    f"pepe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    "image/png"
                )
            else:
                st.subheader("All Variations")
                cols = st.columns(min(num_vars, 2))
                for idx, img in enumerate(st.session_state.generated_images[-num_vars:]):
                    with cols[idx % 2]:
                        st.image(img, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    elif generate and not prompt:
        st.error("Please enter a prompt!")
    
    # Gallery
    if st.session_state.generated_images:
        st.divider()
        with st.expander(f"🖼️ Gallery ({len(st.session_state.generated_images)} images)"):
            cols = st.columns(4)
            for idx, img in enumerate(reversed(st.session_state.generated_images[-8:])):
                with cols[idx % 4]:
                    st.image(img, use_container_width=True)
    
    # Footer
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Generated", st.session_state.generation_count)
    with col_b:
        st.metric("In Gallery", len(st.session_state.generated_images))
    with col_c:
        if st.button("🗑️ Clear"):
            st.session_state.generated_images = []
            st.session_state.generation_count = 0
            st.rerun()


if __name__ == "__main__":
    main()