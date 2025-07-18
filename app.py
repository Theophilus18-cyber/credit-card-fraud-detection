import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb 
from geopy.distance import geodesic

# Custom CSS for Standard Bank styling
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Standard Bank blue colors and speech functionality
st.markdown("""
<style>
    /* Standard Bank Blue Color Palette */
    :root {
        --sb-primary: #003DA5;
        --sb-secondary: #0052CC;
        --sb-accent: #0066FF;
        --sb-light: #E6F0FF;
        --sb-dark: #002B7A;
        --sb-success: #28A745;
        --sb-danger: #DC3545;
        --sb-warning: #FFC107;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, var(--sb-primary) 0%, var(--sb-secondary) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 61, 165, 0.15);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Form container styling */
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--sb-primary);
        box-shadow: 0 0 0 3px rgba(0, 61, 165, 0.1);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--sb-primary);
        box-shadow: 0 0 0 3px rgba(0, 61, 165, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: var(--sb-primary);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: var(--sb-primary);
    }
    
    .stSlider > div > div > div > div > div {
        background-color: var(--sb-primary);
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--sb-primary) 0%, var(--sb-secondary) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 61, 165, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 61, 165, 0.4);
    }
    
    /* Result container styling */
    .result-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 2rem;
    }
    
    .result-fraud {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(220, 53, 69, 0.3);
    }
    
    .result-legitimate {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
    }
    
    /* Label styling */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stSlider > label {
        font-weight: 600;
        color: var(--sb-dark);
        font-size: 16px;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--sb-primary) 0%, var(--sb-secondary) 100%);
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        padding: 1rem;
    }
    
    /* Voice control button */
    .voice-control {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: var(--sb-primary);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0, 61, 165, 0.3);
        transition: all 0.3s ease;
    }
    
    .voice-control:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(0, 61, 165, 0.4);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .form-container {
            padding: 1.5rem;
        }
    }
</style>

<!-- JavaScript for Microsoft Speech Synthesis -->
<script>
// Initialize speech synthesis
let speechSynthesis = window.speechSynthesis;
let speechUtterance = null;

// Function to speak text
function speakText(text, voiceName = null) {
    // Cancel any ongoing speech
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
    }
    
    // Create new utterance
    speechUtterance = new SpeechSynthesisUtterance(text);
    
    // Set properties for better quality
    speechUtterance.rate = 0.9;  // Slightly slower for clarity
    speechUtterance.pitch = 1.0; // Normal pitch
    speechUtterance.volume = 0.8; // Good volume
    
    // Try to use Microsoft voices if available
    if (voiceName) {
        const voices = speechSynthesis.getVoices();
        const microsoftVoice = voices.find(voice => 
            voice.name.includes(voiceName) || 
            voice.name.includes('Microsoft') ||
            voice.name.includes('Hannah') ||
            voice.name.includes('Zira')
        );
        if (microsoftVoice) {
            speechUtterance.voice = microsoftVoice;
        }
    }
    
    // Speak the text
    speechSynthesis.speak(speechUtterance);
}

// Function to get available voices
function getVoices() {
    return new Promise((resolve) => {
        let voices = speechSynthesis.getVoices();
        if (voices.length > 0) {
            resolve(voices);
        } else {
            speechSynthesis.onvoiceschanged = () => {
                voices = speechSynthesis.getVoices();
                resolve(voices);
            };
        }
    });
}

// Welcome message when page loads
window.addEventListener('load', function() {
    setTimeout(() => {
        speakText("Welcome to Standard Bank Fraud Detection System. Please enter your credit card information and verify the transaction details.");
    }, 1000);
});

// Function to speak fraud detection result
function speakResult(isFraud) {
    if (isFraud) {
        speakText("Warning! Fraudulent transaction detected. This transaction has been flagged as potentially fraudulent. Please review the details and contact customer support if necessary.");
    } else {
        speakText("Transaction approved! This transaction appears to be legitimate and can proceed normally.");
    }
}

// Make functions available globally
window.speakText = speakText;
window.speakResult = speakResult;
window.getVoices = getVoices;
</script>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoders.jb")
    return model, encoder

model, encoder = load_models()

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1),(lat2,lon2)).km

# Voice control sidebar
with st.sidebar:
    st.markdown("### 🔊 Voice Controls")
    
    # Voice settings
    voice_enabled = st.checkbox("Enable Voice Guidance", value=True)
    
    if voice_enabled:
        st.markdown("**Voice Features:**")
        st.markdown("• Welcome message on page load")
        st.markdown("• Fraud detection results")
        st.markdown("• Transaction status updates")
        
        # Test voice button
        if st.button("🎤 Test Voice"):
            st.markdown("""
            <script>
                if ('speechSynthesis' in window) {
                    const utterance = new SpeechSynthesisUtterance("Voice system is working correctly. You can now use the fraud detection system.");
                    utterance.rate = 0.9;
                    utterance.volume = 0.8;
                    speechSynthesis.speak(utterance);
                } else {
                    console.log('Speech synthesis not supported');
                }
            </script>
            """, unsafe_allow_html=True)
            st.success("Voice test completed! Check your browser's audio settings if you can't hear anything.")
            
        # Instructions for voice setup
        st.markdown("**🔧 Voice Setup Instructions:**")
        st.markdown("1. Make sure your browser allows audio")
        st.markdown("2. Check that your speakers/headphones are on")
        st.markdown("3. Try refreshing the page if voice doesn't work")
        st.markdown("4. Some browsers may require user interaction first")

# Header Section
st.markdown("""
<div class="main-header">
    <h1>💳 Credit Card Fraud Detection System</h1>
    <p>Advanced AI-powered fraud detection using machine learning and geospatial analysis</p>
</div>
""", unsafe_allow_html=True)

# Welcome message with voice
if voice_enabled:
    st.markdown("""
    <script>
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance("Welcome to Standard Bank Fraud Detection System. Please enter your credit card information and verify the transaction details.");
            utterance.rate = 0.9;
            utterance.volume = 0.8;
            speechSynthesis.speak(utterance);
        }
    </script>
    """, unsafe_allow_html=True)

# Main Form Container
st.markdown('<div class="form-container">', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Transaction Details")
    merchant = st.text_input("🏪 Merchant Name", placeholder="Enter merchant name")
    category = st.text_input("🏷️ Category", placeholder="e.g., grocery, electronics")
    amt = st.number_input("💰 Transaction Amount ($)", min_value=0.0, format="%.2f", value=100.0)
    cc_num = st.text_input("💳 Credit Card Number", placeholder="Enter card number")

with col2:
    st.markdown("### 🗺️ Location Information")
    lat = st.number_input("🌍 Customer Latitude", format="%.6f", value=40.7128)
    long = st.number_input("🌍 Customer Longitude", format="%.6f", value=-74.0060)
    merch_lat = st.number_input("🏢 Merchant Latitude", format="%.6f", value=40.7589)
    merch_long = st.number_input("🏢 Merchant Longitude", format="%.6f", value=-73.9851)

# Time and Personal Information
st.markdown("### ⏰ Transaction Timing")
col3, col4, col5 = st.columns(3)

with col3:
    hour = st.slider("🕐 Transaction Hour", 0, 23, 12)
    
with col4:
    day = st.slider("📅 Transaction Day", 1, 31, 15)
    
with col5:
    month = st.slider("📆 Transaction Month", 1, 12, 6)

st.markdown("### 👤 Customer Information")
gender = st.selectbox("👤 Gender", ["Male", "Female"])

# Calculate distance
distance = haversine(lat, long, merch_lat, merch_long)

# Display distance information
st.info(f"📍 **Distance between customer and merchant:** {distance:.2f} km")

st.markdown('</div>', unsafe_allow_html=True)

# Fraud Detection Button
if st.button("🔍 Check For Fraud", use_container_width=True):
    if merchant and category and cc_num:
        with st.spinner("🔍 Analyzing transaction..."):
            input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                      columns=['merchant','category','amt','distance','hour','day','month','gender','cc_num'])
            
            categorical_col = ['merchant','category','gender']
            for col in categorical_col:
                try:
                    input_data[col] = encoder[col].transform(input_data[col])
                except ValueError:
                    input_data[col] = -1

            input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
            prediction = model.predict(input_data)[0]
            
            # Result display
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("""
                <div class="result-fraud">
                    <h2>🚨 FRAUDULENT TRANSACTION DETECTED</h2>
                    <p>This transaction has been flagged as potentially fraudulent. Please review the transaction details and contact customer support if necessary.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Voice announcement for fraud
                if voice_enabled:
                    st.markdown("""
                    <script>
                        setTimeout(() => {
                            if ('speechSynthesis' in window) {
                                const utterance = new SpeechSynthesisUtterance("Warning! Fraudulent transaction detected. This transaction has been flagged as potentially fraudulent. Please review the details and contact customer support if necessary.");
                                utterance.rate = 0.9;
                                utterance.volume = 0.8;
                                speechSynthesis.speak(utterance);
                            }
                        }, 500);
                    </script>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-legitimate">
                    <h2>✅ LEGITIMATE TRANSACTION</h2>
                    <p>This transaction appears to be legitimate and can proceed normally.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Voice announcement for legitimate transaction
                if voice_enabled:
                    st.markdown("""
                    <script>
                        setTimeout(() => {
                            if ('speechSynthesis' in window) {
                                const utterance = new SpeechSynthesisUtterance("Transaction approved! This transaction appears to be legitimate and can proceed normally.");
                                utterance.rate = 0.9;
                                utterance.volume = 0.8;
                                speechSynthesis.speak(utterance);
                            }
                        }, 500);
                    </script>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional transaction details
            st.markdown("### 📊 Transaction Summary")
            col6, col7, col8 = st.columns(3)
            
            with col6:
                st.metric("💰 Amount", f"${amt:,.2f}")
            with col7:
                st.metric("📍 Distance", f"{distance:.2f} km")
            with col8:
                st.metric("🕐 Time", f"{hour:02d}:00")
                
    else:
        st.error("⚠️ Please fill in all required fields (Merchant Name, Category, and Credit Card Number)")
        
        # Voice reminder for missing fields
        if voice_enabled:
            st.markdown("""
            <script>
                if ('speechSynthesis' in window) {
                    const utterance = new SpeechSynthesisUtterance("Please fill in all required fields including merchant name, category, and credit card number.");
                    utterance.rate = 0.9;
                    utterance.volume = 0.8;
                    speechSynthesis.speak(utterance);
                }
            </script>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔒 Powered by Advanced Machine Learning | Built with ❤️ for secure transactions</p>
    <p>💳 Standard Bank Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)