# MedGemma Clinical Reasoning AI

A Streamlit web application that uses Google's MedGemma-4B model to provide AI-powered clinical reasoning and decision support for Kenya healthcare professionals.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Authentication
1. Get a Hugging Face token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
3. Edit `.env` and replace `your_token_here` with your actual token:
   ```
   HF_TOKEN=hf_your_actual_token_here
   ```

### 3. Run the Application
```bash
streamlit run streamlit_clinical_reasoning_app.py
```

## 🏥 Features

- **Real MedGemma AI**: Uses Google's MedGemma-4B model for medical reasoning
- **Kenya Healthcare Context**: Optimized for Kenyan medical scenarios
- **Demo Cases**: Pre-built clinical scenarios to try
- **Interactive Interface**: Input your own clinical cases
- **Clinical Settings**: Customize location, healthcare level, and specialty
- **Download Reports**: Save clinical assessments as text files

## 📋 Demo Clinical Cases

The app includes 5 pre-built demo cases:
- Acute Abdominal Pain
- Pediatric Fever and Cough  
- Hypertensive Crisis
- Malaria Suspicion
- Diabetic Emergency

## 🔧 System Requirements

- **GPU**: 8GB+ VRAM recommended (for MedGemma-4B)
- **RAM**: 16GB+ system memory
- **Python**: 3.8+
- **CUDA**: Compatible GPU for best performance

## 🔒 Security

- Environment variables are loaded from `.env` file
- `.env` file is ignored by git to keep tokens secure
- Use `.env.example` as a template

## ⚠️ Disclaimer

This application is for educational and clinical decision support purposes only. Always consult qualified healthcare professionals for patient care decisions.

## 🇰🇪 Kenya Healthcare Integration

The app is specifically designed for the Kenya healthcare system with:
- All 47 Kenya counties
- Kenya healthcare levels (1-6)
- Local medical terminology
- Regional health considerations
