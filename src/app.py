"""Pepe the Frog Meme Generator - Main Streamlit Application.

This is the main entry point for the web application. It provides a user-friendly
interface for generating Pepe memes using AI-powered Stable Diffusion models.

The application features:
- Model selection (multiple LoRA variants, LCM support)
- Style presets and raw prompt mode
- Advanced generation settings (steps, guidance, seed)
- Text overlay capability for meme creation
- Gallery system for viewing generated images
- Download functionality
- Progress tracking during generation

Application Structure:
    1. Page configuration and styling
    2. Session state initialization
    3. Model loading and caching
    4. Sidebar UI (model selection, settings)
    5. Main content area (prompt input, generation)
    6. Results display and download
    7. Gallery view

Usage:
    Run with: streamlit run src/app.py
    Access at: http://localhost:8501

Author: MJaheen
License: MIT
"""

import streamlit as st
from PIL import Image
import io
from datetime import datetime

# Import our modules
from model.generator import PepeGenerator
from model.config import ModelConfig
from utils.image_processor import ImageProcessor

def supports_use_container_width():
    """Check if the current Streamlit version supports use_container_width parameter."""
    try:
        import pkg_resources
        streamlit_version = pkg_resources.get_distribution("streamlit").version
        # use_container_width was introduced in Streamlit 1.16.0
        return tuple(map(int, streamlit_version.split('.')[:2])) >= (1, 16)
    except:
        return False

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
    """
    Initialize Streamlit session state variables.
    
    This function sets up persistent state across app reruns:
    - generated_images: List of all generated images in current session
    - generation_count: Counter for tracking total generations
    - current_model: Currently selected model name for cache invalidation
    
    Session state persists across reruns but is reset when the page is refreshed.
    """
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None


@st.cache_resource
def load_generator(model_name: str = "Pepe Fine-tuned (LoRA)"):
    """
    Load and cache the Stable Diffusion generator.
    
    This function loads a PepeGenerator instance configured with the selected
    model. It's cached using @st.cache_resource to avoid reloading the model
    on every interaction, which would be very slow.
    
    The cache is automatically invalidated when:
    - The model_name parameter changes
    - The user manually clears cache
    
    Args:
        model_name: Name of the model from AVAILABLE_MODELS dict.
            Examples: "Pepe Fine-tuned (LoRA)", "Pepe + LCM (FAST)"
    
    Returns:
        PepeGenerator: Configured generator instance with loaded model.
    
    Note:
        Model loading can take 30-60 seconds on first load as it downloads
        weights from Hugging Face (~4GB for base model + LoRA).
    """
    config = ModelConfig()
    model_config = config.AVAILABLE_MODELS[model_name]
    
    # Update config with selected model settings
    config.BASE_MODEL = model_config['base']
    config.LORA_PATH = model_config.get('lora')
    config.USE_LORA = model_config.get('use_lora', False)
    config.TRIGGER_WORD = model_config.get('trigger_word', 'pepe the frog')
    
    # LCM settings
    config.USE_LCM = model_config.get('use_lcm', False)
    config.LCM_LORA_PATH = model_config.get('lcm_lora')
    
    # Log which model is being loaded
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Base: {config.BASE_MODEL}, LoRA: {config.USE_LORA}, LCM: {config.USE_LCM}")
    
    return PepeGenerator(config)


def get_example_prompts():
    """
    Return a list of example prompts for inspiration.
    
    These prompts are designed to work well with the fine-tuned Pepe model
    and demonstrate various styles, activities, and scenarios.
    
    Returns:
        list: List of example prompt strings with trigger word and descriptions.
    """
    return [
        "pepe the frog as a wizard casting spells",
        "pepe the frog coding on a laptop",
        "pepe the frog drinking coffee",
        "pepe the frog as a superhero",
        "pepe the frog reading a book",
    ]


def main():
    """
    Main application function that builds and runs the Streamlit UI.
    
    This function orchestrates the entire application flow:
    1. Initializes session state
    2. Loads configuration and sets up sidebar controls
    3. Handles model selection and switching
    4. Processes user input (prompts, settings)
    5. Generates images when requested
    6. Displays results with download options
    7. Shows gallery of previous generations
    
    The UI is organized into:
    - Sidebar: Model selection, style presets, advanced settings
    - Main area: Prompt input, generation button, results
    - Bottom: Gallery view (expandable)
    
    Flow:
        User selects model → Enters prompt → Adjusts settings → 
        Clicks generate → Shows progress → Displays result → 
        Offers download → Adds to gallery
    """
    # Initialize session state for persistent data across reruns
    init_session_state()
    
    # Sidebar (needs to be first to define selected_model)
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    config = ModelConfig()
    available_models = list(config.AVAILABLE_MODELS.keys())
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        available_models,
        index=0,
        help="Select which model to use for generation"
    )
    
    # Detect model change and auto-clear cache
    if st.session_state.current_model is not None and st.session_state.current_model != selected_model:
        st.cache_resource.clear()
        st.sidebar.success(f"✅ Switched to: {selected_model}")
    
    # Update current model in session state
    st.session_state.current_model = selected_model
    
    # Show LCM mode indicator if enabled
    model_config = config.AVAILABLE_MODELS[selected_model]
    if model_config.get('use_lcm', False):
        st.sidebar.success("⚡ LCM Mode: 8x Faster! (6-8 steps optimal)")
    
    # Header
    st.title("🐸 Pepe the Frog Meme Generator")
    st.markdown("Create custom Pepe memes using AI! Powered by Stable Diffusion.")
    
    st.sidebar.divider()
    
    # Style selection
    st.sidebar.subheader("🎨 Style & Prompt")
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
    
    # Raw prompt mode
    use_raw_prompt = st.sidebar.checkbox(
        "Raw Prompt Mode",
        help="Use your exact prompt without trigger words or style modifiers"
    )
    
    # Advanced settings - adjust defaults based on LCM mode
    is_lcm_mode = model_config.get('use_lcm', False)
    
    with st.sidebar.expander("🔧 Advanced Settings"):
        if is_lcm_mode:
            # LCM needs fewer steps and lower guidance
            steps = st.slider("Steps", 4, 12, 6, 1, 
                            help="⚡ LCM Mode: 4-8 steps optimal. Recommended: 6")
            guidance = st.slider("Guidance Scale", 1.0, 2.5, 1.5, 0.1,
                               help="⚡ LCM Mode: Lower guidance (1.0-2.0). Recommended: 1.5")
        else:
            # Normal mode settings
            steps = st.slider("Steps", 15, 50, 25, 5, 
                            help="Fewer steps = faster generation. 20-25 recommended for CPU")
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
            # Use use_container_width only if supported by Streamlit version
            image_kwargs = {}
            if supports_use_container_width():
                image_kwargs['use_container_width'] = True

            placeholder.image(
                st.session_state.generated_images[-1],
                **image_kwargs
            )
        else:
            placeholder.info("Your meme will appear here...")
    
    # Generate
    if generate and prompt:
        try:
            generator = load_generator(selected_model)
            processor = ImageProcessor()
            
            # Overall progress for multiple images
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # Progress for current image generation steps
            step_progress = st.progress(0)
            step_status = st.empty()
            
            for i in range(num_vars):
                overall_status.text(f"Generating image {i+1}/{num_vars}...")
                
                # Define callback for step-by-step progress
                def progress_callback(current_step: int, total_steps: int):
                    step_progress.progress(current_step / total_steps)
                    step_status.text(f"Step {current_step}/{total_steps}")
                
                # Generate with progress callback
                image = generator.generate(
                    prompt=prompt,
                    style=style,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    progress_callback=progress_callback,
                    raw_prompt=use_raw_prompt
                )
                
                # Add text if requested
                if add_text and (top_text or bottom_text):
                    image = processor.add_meme_text(image, top_text, bottom_text)
                
                # Always add MJ signature
                image = processor.add_signature(image, signature="MJaheen", font_size=10, opacity=200)
                
                st.session_state.generated_images.append(image)
                st.session_state.generation_count += 1
                
                # Update overall progress
                overall_progress.progress((i + 1) / num_vars)
            
            # Clear progress indicators
            overall_progress.empty()
            overall_status.empty()
            step_progress.empty()
            step_status.empty()
            
            # Show result
            if num_vars == 1:
                # Use use_container_width only if supported by Streamlit version
                image_kwargs = {}
                if supports_use_container_width():
                    image_kwargs['use_container_width'] = True

                placeholder.image(image, **image_kwargs)
                
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
                        # Use use_container_width only if supported by Streamlit version
                        image_kwargs = {}
                        if supports_use_container_width():
                            image_kwargs['use_container_width'] = True

                        st.image(img, **image_kwargs)
        
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
                    # Use use_container_width only if supported by Streamlit version
                    image_kwargs = {}
                    if supports_use_container_width():
                        image_kwargs['use_container_width'] = True

                    st.image(img, **image_kwargs)
    
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
    
    # Personal Information
    st.divider()
    st.markdown("### 👨‍💻 About the Engineer")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Contact Information:**
        - 📧 Email: [Mohamed.a.jaheen@gmail.com](mailto:Mohamed.a.jaheen@gmail.com)
        - 🔗 LinkedIn: [Mohamed Jaheen](https://www.linkedin.com/in/mohamedjaheen/)
        """)
    
    with info_col2:
        st.markdown("""
        **About this App:**
        - supported by worldquant university
        - Built with Streamlit & Stable Diffusion
        - Fine-tuned Pepe model available
        - Open source and customizable
        - MIT licences
        """)
    
    st.caption("© 2025 - AI Meme Generator (Pepe the Frog) | Made with ❤️ using Python and MJ")


if __name__ == "__main__":
    main()