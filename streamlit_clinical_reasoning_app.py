import streamlit as st
import torch
import pandas as pd
import time
import os
import warnings
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
import gc
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MedGemma Clinical Reasoning AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .clinical-case {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 10px 0;
    }
    
    .ai-response {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    
    .demo-case {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 5px 0;
        cursor: pointer;
    }
    
    .demo-case:hover {
        background-color: #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Demo clinical cases for users to try
DEMO_CASES = [
    {
        "title": "Acute Abdominal Pain",
        "case": "A 28-year-old female presents to the emergency department with sudden onset of severe right lower quadrant abdominal pain that started 6 hours ago. The pain is sharp, constant, and has worsened progressively. She reports nausea and has vomited twice. No fever. Last menstrual period was 3 weeks ago. No significant past medical history.",
        "county": "Nairobi",
        "health_level": "Level 5 Hospital",
        "clinical_panel": "Emergency Medicine"
    },
    {
        "title": "Pediatric Fever and Cough",
        "case": "A 3-year-old boy brought by mother with 4-day history of fever (up to 39°C), cough, and difficulty breathing. Mother reports the child has been less active, eating poorly, and breathing faster than usual. No recent travel. Fully vaccinated for age. Lives in a rural area with recent seasonal rains.",
        "county": "Kisumu",
        "health_level": "Level 4 Hospital",
        "clinical_panel": "Pediatrics"
    },
    {
        "title": "Hypertensive Crisis",
        "case": "A 55-year-old male presents with severe headache, blurred vision, and blood pressure reading of 210/120 mmHg. He has a known history of hypertension but admits to poor medication compliance. He also reports chest tightness and shortness of breath that started this morning.",
        "county": "Mombasa",
        "health_level": "Level 6 Hospital",
        "clinical_panel": "Internal Medicine"
    },
    {
        "title": "Malaria Suspicion",
        "case": "A 25-year-old pregnant woman (28 weeks gestation) presents with 3-day history of fever, chills, and general body weakness. She lives in a malaria-endemic area and reports sleeping without a mosquito net. She has been attending antenatal clinic regularly with no previous complications.",
        "county": "Kakamega",
        "health_level": "Level 3 Health Centre",
        "clinical_panel": "Obstetrics & Gynecology"
    },
    {
        "title": "Diabetic Emergency",
        "case": "A 42-year-old male with type 2 diabetes presents with excessive thirst, frequent urination, and fatigue for the past week. He reports running out of diabetes medication 2 weeks ago. Blood glucose on arrival is 24 mmol/L. He appears dehydrated and has a fruity smell on his breath.",
        "county": "Eldoret",
        "health_level": "Level 5 Hospital",
        "clinical_panel": "Endocrinology"
    }
]

@st.cache_resource
def load_medgemma_model():
    """Load and cache the MedGemma model"""
    MODEL_NAME = "google/medgemma-4b-it"
    
    try:
        # Authentication - check for HF_TOKEN in environment or .env file
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            st.error("🔑 Hugging Face Token Required")
            st.markdown("""
            **To use this app, you need a Hugging Face token:**
            
            1. **Get a token:** Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
            2. **Create `.env` file:** In the same folder as this app, create a file named `.env`
            3. **Add your token:** Put this line in the `.env` file:
               ```
               HF_TOKEN=your_actual_token_here
               ```
            4. **Restart the app:** Stop and restart the Streamlit app
            
            The `.env` file will be automatically loaded and your token kept secure.
            """)
            st.stop()
        
        # Login to HuggingFace with the token (required for gated models like MedGemma)
        login(token=hf_token)
        
        with st.spinner("Loading MedGemma model... This may take a few minutes on first run."):
            # Load processor
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            
            # Load model
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available() and not hasattr(model, 'device_map'):
                model = model.cuda()
            
            model.eval()
            
        st.success("✅ MedGemma model loaded successfully!")
        return model, processor
        
    except Exception as e:
        st.error(f"Failed to load MedGemma model: {str(e)}")
        st.error("Please ensure you have a valid Hugging Face token and sufficient GPU memory.")
        st.stop()

def create_clinical_prompt(case_text, county="Kenya", health_level="healthcare facility", 
                         clinical_panel="General medicine", experience="experienced"):
    """Create structured prompts for clinical reasoning - from notebook"""
    
    structured_prompt = f"""You are an experienced clinician working in Kenya providing clinical reasoning and medical guidance.

Context:
- Location: {county}, Kenya
- Healthcare Level: {health_level}
- Clinical Expertise: {clinical_panel}
- Years of Experience: {experience}

Clinical Case:
{case_text}

Please provide a comprehensive clinical assessment including:
1. Clinical summary
2. Differential diagnosis considerations
3. Immediate management steps
4. Treatment recommendations
5. Follow-up care if needed

Clinical Response:"""
    
    return structured_prompt

def generate_clinical_response_fixed(prompt, model, processor, max_new_tokens=512):
    """Generate clinical response using MedGemma - from notebook with fixes"""
    
    try:
        # Simplify the prompt format
        simple_prompt = f"""As a medical expert, analyze this clinical case:

{prompt}

Provide your clinical assessment:"""
        
        # Direct tokenization
        inputs = processor.tokenizer(
            simple_prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        # Move to device
        device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Set model to eval mode
        model.eval()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=device_inputs["input_ids"],
                attention_mask=device_inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=False
            )
            
            # Handle generator object if returned
            if hasattr(outputs, '__iter__') and not torch.is_tensor(outputs):
                outputs = list(outputs)[0] if hasattr(outputs, '__iter__') else outputs
            
            # Extract new tokens
            input_length = device_inputs["input_ids"].shape[1]
            
            if torch.is_tensor(outputs):
                if len(outputs.shape) > 1:
                    generated_tokens = outputs[0][input_length:]
                else:
                    generated_tokens = outputs[input_length:]
            else:
                raise ValueError("Model output is not a tensor")
            
            # Decode response
            response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return f"""Clinical Assessment:

Based on the clinical presentation, this case requires:
1. Comprehensive history taking and physical examination
2. Appropriate diagnostic investigations
3. Evidence-based treatment planning
4. Regular follow-up and monitoring

Recommendation: Please consult with senior medical staff for detailed evaluation and management plan.

Note: This is a fallback response due to generation error: {str(e)}"""

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MedGemma Clinical Reasoning AI</h1>
        <p>Advanced AI-powered clinical decision support for Kenya healthcare professionals</p>
        <p><em>Powered by Google's MedGemma-4B medical AI model</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, processor = load_medgemma_model()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Clinical Settings")
        
        # Healthcare context settings
        county = st.selectbox(
            "County/Location",
            ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Kakamega", "Meru", "Kisii", "Garissa", "Other"],
            index=0
        )
        
        health_level = st.selectbox(
            "Healthcare Level",
            ["Level 1 - Community Unit", "Level 2 - Dispensary", "Level 3 - Health Centre", 
             "Level 4 - Sub-County Hospital", "Level 5 - County Hospital", "Level 6 - National Hospital"],
            index=4
        )
        
        clinical_panel = st.selectbox(
            "Clinical Specialty",
            ["General Medicine", "Emergency Medicine", "Pediatrics", "Internal Medicine", 
             "Obstetrics & Gynecology", "Surgery", "Orthopedics", "Psychiatry", "Dermatology", "Ophthalmology"],
            index=0
        )
        
        experience_years = st.slider("Years of Experience", 1, 40, 10)
        
        # Model parameters
        st.header("🤖 AI Parameters")
        max_tokens = st.slider("Max Response Length", 256, 1024, 512)
        
        # System information
        st.header("💻 System Info")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            st.info(f"GPU: {gpu_name[:30]}...")
            st.info(f"GPU Memory: {gpu_memory:.1f} GB")
        else:
            st.warning("Running on CPU")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Clinical Case Input")
        
        # Demo cases section
        st.subheader("🎯 Demo Cases (Click to Use)")
        for i, demo in enumerate(DEMO_CASES):
            if st.button(f"📋 {demo['title']}", key=f"demo_{i}", use_container_width=True):
                st.session_state.demo_case = demo
        
        # Clinical case input
        if 'demo_case' in st.session_state:
            demo = st.session_state.demo_case
            st.markdown(f"**Using Demo Case: {demo['title']}**")
            clinical_case = st.text_area(
                "Clinical Case Description",
                value=demo['case'],
                height=200,
                help="Describe the patient presentation, symptoms, history, and any relevant clinical findings"
            )
            
            # Update settings from demo
            county = demo.get('county', county)
            health_level = demo.get('health_level', health_level)
            clinical_panel = demo.get('clinical_panel', clinical_panel)
            
        else:
            clinical_case = st.text_area(
                "Clinical Case Description",
                placeholder="Enter the clinical case here...\n\nExample:\nA 45-year-old male presents with chest pain that started 2 hours ago. The pain is crushing in nature, radiates to the left arm, and is associated with sweating and nausea. He has a history of hypertension and smoking.",
                height=200,
                help="Describe the patient presentation, symptoms, history, and any relevant clinical findings"
            )
        
        # Additional clinical details
        st.subheader("📊 Additional Clinical Context")
        
        col_a, col_b = st.columns(2)
        with col_a:
            patient_age = st.number_input("Patient Age", 0, 120, 35)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Not specified"])
        
        with col_b:
            urgency = st.selectbox("Case Urgency", ["Routine", "Urgent", "Emergency"])
            setting = st.selectbox("Clinical Setting", ["Outpatient", "Emergency Dept", "Inpatient Ward", "ICU"])
        
        # Generate button
        generate_btn = st.button("🔬 Analyze Clinical Case", type="primary", use_container_width=True)
    
    with col2:
        st.header("🤖 AI Clinical Assessment")
        
        if generate_btn and clinical_case.strip():
            # Create enhanced clinical prompt
            enhanced_case = f"""Patient: {patient_age}-year-old {patient_gender.lower()}
Setting: {setting}
Urgency: {urgency}

Clinical Presentation:
{clinical_case}"""
            
            structured_prompt = create_clinical_prompt(
                enhanced_case, county, health_level, clinical_panel, f"{experience_years} years"
            )
            
            # Show processing
            with st.spinner("🧠 MedGemma is analyzing the clinical case..."):
                start_time = time.time()
                
                try:
                    # Generate response
                    response = generate_clinical_response_fixed(
                        structured_prompt, model, processor, max_tokens
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Display response
                    st.markdown(f"""
                    <div class="ai-response">
                        <h4>🔍 Clinical Assessment</h4>
                        <p><strong>Processing Time:</strong> {processing_time:.2f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(response)
                    
                    # Additional metrics
                    st.markdown("---")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Response Length", f"{len(response)} chars")
                    with col_m2:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    with col_m3:
                        word_count = len(response.split())
                        st.metric("Word Count", word_count)
                    
                    # Save to session state for download
                    st.session_state.last_assessment = {
                        'case': clinical_case,
                        'response': response,
                        'context': {
                            'county': county,
                            'health_level': health_level,
                            'specialty': clinical_panel,
                            'patient': f"{patient_age}-year-old {patient_gender}",
                            'setting': setting,
                            'urgency': urgency
                        }
                    }
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    
                finally:
                    # Memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        
        elif generate_btn:
            st.warning("Please enter a clinical case description before analyzing.")
        
        else:
            st.info("👆 Enter a clinical case and click 'Analyze Clinical Case' to get AI-powered clinical reasoning.")
            
            # Show example
            st.markdown("""
            **Example Clinical Case:**
            
            *A 65-year-old female with diabetes presents with a 3-day history of fever, dysuria, and suprapubic pain. She has been increasingly confused over the past 24 hours. Vital signs show fever 38.5°C, BP 90/60, HR 110. Urinalysis shows nitrites positive, leukocyte esterase positive.*
            """)
    
    # Download section
    if 'last_assessment' in st.session_state:
        st.markdown("---")
        st.header("💾 Download Assessment")
        
        assessment = st.session_state.last_assessment
        
        # Create downloadable report
        report = f"""CLINICAL ASSESSMENT REPORT
Generated by MedGemma Clinical Reasoning AI
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

CLINICAL CONTEXT:
- Location: {assessment['context']['county']}, Kenya
- Healthcare Level: {assessment['context']['health_level']}
- Clinical Specialty: {assessment['context']['specialty']}
- Patient: {assessment['context']['patient']}
- Setting: {assessment['context']['setting']}
- Urgency: {assessment['context']['urgency']}

CLINICAL CASE:
{assessment['case']}

AI CLINICAL ASSESSMENT:
{assessment['response']}

---
DISCLAIMER: This assessment is generated by AI and should be used as a clinical decision support tool only. 
Always consult with qualified healthcare professionals for patient care decisions.
"""
        
        st.download_button(
            label="📄 Download Clinical Assessment Report",
            data=report,
            file_name=f"clinical_assessment_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>MedGemma Clinical Reasoning AI</strong></p>
        <p>Built for Kenya Healthcare Professionals | Powered by Google's MedGemma-4B</p>
        <p><em>⚠️ For educational and clinical decision support only. Always consult qualified healthcare professionals.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
