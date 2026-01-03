import streamlit as st
import requests
import pandas as pd
import time
import os
import warnings
from dotenv import load_dotenv
import json

# Try to import gradio_client for MedGemma Space API
try:
    from gradio_client import Client as GradioClient
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

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
    
    .hosted-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
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

def create_clinical_prompt(case_text, county="Kenya", health_level="healthcare facility", 
                         clinical_panel="General medicine", experience="experienced"):
    """Create structured prompts for clinical reasoning"""
    
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

def query_huggingface_inference(prompt, hf_token, max_new_tokens=2048, model="google/medgemma-4b-it"):
    """Query MedGemma via HuggingFace Space Gradio API
    
    Since MedGemma is not available through HF Inference Providers (it's a gated model),
    we use a public Space that hosts MedGemma with ZeroGPU.
    """
    
    if not GRADIO_AVAILABLE:
        return "Error: gradio_client not installed. Run: pip install gradio_client"
    
    # Medical system prompt for clinical reasoning
    system_prompt = """You are an expert medical clinician with extensive experience in clinical reasoning and diagnosis. 
You are working in Kenya and are familiar with local healthcare contexts, common diseases in the region (including malaria, TB, HIV), 
and resource-appropriate medicine. Provide comprehensive, evidence-based clinical assessments.

When analyzing cases, consider:
1. Patient presentation and vital signs
2. Differential diagnoses ranked by likelihood
3. Recommended investigations appropriate to the healthcare level
4. Evidence-based treatment recommendations
5. Red flags and when to refer
6. Follow-up care recommendations"""

    try:
        # Connect to a public MedGemma Space (uses ZeroGPU)
        client = GradioClient('warshanks/medgemma-4b-it')
        
        # Call the MedGemma model via Gradio API
        result = client.predict(
            message={'text': prompt, 'files': []},
            param_2=system_prompt,
            param_3=max_new_tokens,
            api_name='/chat'
        )
        
        if isinstance(result, str):
            return result.strip()
        elif isinstance(result, dict):
            return str(result)
        else:
            return str(result)
            
    except Exception as e:
        error_msg = str(e)
        if "exceeded your GPU quota" in error_msg.lower():
            return "MedGemma Space GPU quota exceeded. Please try again in a few minutes."
        elif "is currently loading" in error_msg.lower() or "loading" in error_msg.lower():
            return "MedGemma model is loading. Please wait a moment and try again."
        else:
            return f"MedGemma API Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "Request timed out. The model may be busy. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def query_google_cloud_api(prompt, api_key, max_new_tokens=512):
    """Query Google Cloud Model Garden API (placeholder implementation)"""
    
    # This is a placeholder - you'd need to implement actual Google Cloud API calls
    # Based on Google Cloud Model Garden documentation
    
    st.warning("Google Cloud Model Garden integration not yet implemented. Using fallback response.")
    
    return """Clinical Assessment (Google Cloud API - Placeholder):

Based on the clinical presentation, this case requires:
1. Comprehensive history taking and physical examination
2. Appropriate diagnostic investigations
3. Evidence-based treatment planning
4. Regular follow-up and monitoring

Recommendation: Please consult with senior medical staff for detailed evaluation and management plan.

Note: This is a placeholder response. Google Cloud Model Garden API integration coming soon."""

def generate_fallback_response(prompt):
    """Generate a structured fallback response when APIs are unavailable"""
    
    return f"""Clinical Assessment (Fallback Response):

Thank you for providing this clinical case. While I cannot process this case with the MedGemma model at the moment, here is a general clinical approach:

**Clinical Summary:**
The case requires careful evaluation and systematic assessment.

**Recommended Approach:**
1. **Immediate Assessment:** Conduct thorough history taking and physical examination
2. **Vital Signs:** Monitor all vital signs and assess for any emergency indicators
3. **Diagnostic Workup:** Order appropriate investigations based on clinical presentation
4. **Treatment Planning:** Develop evidence-based treatment plan
5. **Follow-up Care:** Establish appropriate monitoring and follow-up schedule

**Important Notes:**
- This is a general framework for clinical reasoning
- All cases require individualized assessment by qualified healthcare professionals
- Consider local Kenya clinical guidelines and protocols
- Ensure appropriate resource utilization based on healthcare facility level

**Disclaimer:** This response is generated as a fallback when the AI model is unavailable. Please consult with senior clinical staff for actual patient care decisions.
"""

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MedGemma Clinical Reasoning AI</h1>
        <p>Advanced AI-powered clinical decision support for Kenya healthcare professionals</p>
        <p><em>Powered by Google's MedGemma-4B medical AI model</em></p>
        <p><span class="hosted-badge">🌐 HOSTED VERSION</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check authentication
    hf_token = os.getenv("HF_TOKEN")
    google_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
    
    if not hf_token and not google_api_key:
        st.error("🔑 **Authentication Required**")
        st.markdown("""
        **To use this hosted app, you need authentication:**
        
        **For Hugging Face Spaces Deployment:**
        1. Go to your Space's **Settings** page
        2. Scroll to **Repository secrets**
        3. Add a secret named `HF_TOKEN` with your [Hugging Face token](https://huggingface.co/settings/tokens)
        4. The Space will restart automatically
        
        **For Local Development:**
        1. Create a `.env` file in the project directory
        2. Add `HF_TOKEN=your_token_here`
        3. Restart the app
        
        **Alternative: Google Cloud Model Garden**
        - Add `GOOGLE_CLOUD_API_KEY` secret/environment variable
        """)
        st.stop()
    
    # Sidebar for settings and API selection
    with st.sidebar:
        st.header("⚙️ Clinical Settings")
        
        # MedGemma Status
        st.header("🩺 MedGemma Status")
        st.success("✅ Using Google MedGemma-4B-IT")
        st.caption("Specialized medical AI via Gradio API")
        st.caption("Model: google/medgemma-4b-it")
        
        # Set defaults for internal use
        api_choice = "Hugging Face Inference API"
        model_choice = "google/medgemma-4b-it"
        
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
        max_tokens = st.slider("Max Response Length", 512, 4096, 2048)
        
        # About MedGemma
        st.header("ℹ️ About MedGemma")
        st.info("""
        **MedGemma** is Google's medical AI model 
        fine-tuned for clinical reasoning tasks.
        
        - Built on Gemma 2
        - Specialized medical training
        - Optimized for clinical Q&A
        """)
    
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
            with st.spinner(f"🧠 {api_choice} is analyzing the clinical case..."):
                start_time = time.time()
                
                try:
                    # Generate response based on selected API
                    if api_choice == "Hugging Face Inference API" and hf_token:
                        response = query_huggingface_inference(structured_prompt, hf_token, max_tokens, model_choice)
                    elif api_choice == "Google Cloud Model Garden" and google_api_key:
                        response = query_google_cloud_api(structured_prompt, google_api_key, max_tokens)
                    else:
                        response = generate_fallback_response(structured_prompt)
                    
                    processing_time = time.time() - start_time
                    
                    # Display response
                    model_display = model_choice.split("/")[-1] if api_choice == "Hugging Face Inference API" else api_choice
                    st.markdown(f"""
                    <div class="ai-response">
                        <h4>🔍 Clinical Assessment</h4>
                        <p><strong>Model:</strong> {model_display}</p>
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
                        'api_used': api_choice,
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
        
        elif generate_btn:
            st.warning("Please enter a clinical case description before analyzing.")
        
        else:
            st.info("👆 Enter a clinical case and click 'Analyze Clinical Case' to get AI-powered clinical reasoning.")
            
            # Show API info
            if api_choice == "Hugging Face Inference API":
                st.markdown("""
                **🤗 Hugging Face Inference API:**
                - Serverless MedGemma inference
                - No local GPU required
                - May have cold start delays
                - Free tier with rate limits
                """)
            elif api_choice == "Google Cloud Model Garden":
                st.markdown("""
                **☁️ Google Cloud Model Garden:**
                - Professional hosted inference
                - Consistent performance
                - Pay-per-use pricing
                - Enterprise-grade reliability
                """)
            else:
                st.markdown("""
                **🔄 Fallback Mode:**
                - Structured clinical framework
                - No AI model required
                - Always available
                - Educational guidance only
                """)
    
    # Download section
    if 'last_assessment' in st.session_state:
        st.markdown("---")
        st.header("💾 Download Assessment")
        
        assessment = st.session_state.last_assessment
        
        # Create downloadable report
        report = f"""CLINICAL ASSESSMENT REPORT
Generated by MedGemma Clinical Reasoning AI (Hosted Version)
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
API Used: {assessment.get('api_used', 'Unknown')}

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
            file_name=f"clinical_assessment_hosted_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>MedGemma Clinical Reasoning AI - Hosted Version</strong></p>
        <p>Built for Kenya Healthcare Professionals | Powered by Google's MedGemma-4B</p>
        <p><em>⚠️ For educational and clinical decision support only. Always consult qualified healthcare professionals.</em></p>
        <p><small>🌐 This version uses hosted inference APIs - no local GPU required</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
