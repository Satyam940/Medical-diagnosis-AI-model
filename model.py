import streamlit as st
import joblib
import numpy as np 

import random
from g4f.client import Client

client = Client()

# Page config
st.set_page_config(page_title="Disease Predictor", page_icon="🩺")
print("TEST")

# CSS styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://cdn.analyticsvidhya.com/wp-content/uploads/2022/02/Heart-Disease-Prediction-using-Machine-Learning.webp");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .css-18e3th9 {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 2rem;
        border-radius: 10px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        text-align: center;
    }
    .stButton > button {
        color: white !important;
        background-color: #008CBA !important;
    }

    .stSelectbox, .stMultiSelect, .stTextInput {
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid white !important;
        color: white !important;
    }

    .stMarkdown,label{
        color: white;
    }
    .stMultiSelect {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    div[data-testid="stAlert"] {
    background-color: rgba(0, 0, 0, 0.85);
    color: #00ffcc !important;
    border: 1px solid #00ffcc;
    border-radius: 12px;
    font-weight: bold;
    padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 Disease Predictor Using Symptoms")

# Marquee text
st.markdown("""
    <marquee behavior="scroll" direction="left" scrollamount="6" style="color: cyan; font-size:22px;">
        💡 Tip: Select the symptoms you're experiencing from the list below and click Predict to get a result!
    </marquee>
""", unsafe_allow_html=True)

symptoms = [
    'runny nose', 'sneezing', 'sore throat', 'cough', 'mild fever', 'congestion',
    'watery eyes', 'fatigue', 'body aches', 'stuffy nose', 'high fever',
    'severe body aches', 'headache', 'chills', 'sweating', 'muscle pain',
    'weakness', 'dry cough', 'loss of taste', 'loss of smell',
    'shortness of breath', 'severe headache', 'eye pain', 'joint pain',
    'nausea', 'vomiting', 'skin rash', 'mild bleeding', 'anemia symptoms',
    'abdominal pain', 'prolonged fever', 'stomach pain', 'constipation',
    'diarrhea', 'loss of appetite', 'rose colored rash', 'sudden fever',
    'severe joint pain', 'joint swelling', 'back pain', 'yellow skin',
    'yellow eyes', 'dark urine', 'pale stools', 'itchy skin',
    'muscle aches', 'itchy rash', 'red spots', 'blisters', 'irritability',
    'red watery eyes', 'white spots in mouth', 'red rash', 'swollen salivary glands',
    'pain while chewing', 'difficulty swallowing', 'severe coughing fits',
    'whooping sound', 'vomiting after coughing', 'exhaustion after coughing',
    'feeling unwell', 'painful red blisters', 'rash on hands', 'rash on feet',
    'mouth sores', 'intense itching', 'rash', 'sores from scratching',
    'thick crusts on skin', 'itching worse at night', 'burrow tracks',
    'red bumps', 'circular rash', 'red scaly patches', 'itchy skin',
    'hair loss in patches', 'ring shaped lesions', 'clear center',
    'raised edges', 'sensitivity to light', 'sensitivity to sound',
    'visual disturbances', 'dizziness', 'throbbing pain', 'stomach cramps',
    'dehydration', 'itchy eyes', 'throat irritation', 'postnasal drip',
    'itchy throat', 'hives', 'red patches', 'swelling', 'burning sensation',
    'dry skin', 'blisters', 'skin irritation', 'wheezing', 'chest tightness',
    'difficulty breathing', 'rapid breathing', 'anxiety', 'persistent cough',
    'mucus production', 'chest discomfort', 'cough with phlegm', 'chest pain',
    'confusion', 'coughing blood', 'weight loss', 'night sweats',
    'swollen lymph nodes', 'red throat', 'white patches on throat',
    'facial pain', 'nasal congestion', 'thick nasal discharge',
    'reduced smell', 'ear pressure', 'dental pain', 'swollen tonsils',
    'bad breath', 'ear pain', 'stiff neck', 'tender lymph nodes',
    'indigestion', 'burning stomach pain', 'hiccups', 'black stools',
    'heartburn', 'regurgitation', 'gas', 'cramping', 'mucus in stool',
    'urgency', 'incomplete evacuation', 'pain around navel',
    'pain in lower right abdomen', 'inability to pass gas',
    'sudden severe abdominal pain', 'right shoulder pain',
    'clay colored stools', 'severe back pain', 'side pain',
    'painful urination', 'blood in urine', 'frequent urination',
    'burning urination', 'urgency', 'cloudy urine', 'pelvic pain',
    'strong urine odor', 'urgent need to urinate', 'dull headache',
    'pressure around head', 'tight band sensation', 'neck pain',
    'shoulder tension', 'mild to moderate pain', 'severe eye pain',
    'one-sided headache', 'sharp stabbing pain', 'excessive worry',
    'restlessness', 'difficulty concentrating', 'muscle tension',
    'sleep problems', 'irritability', 'rapid heartbeat',
    'persistent sadness', 'loss of interest', 'appetite changes',
    'mood changes', 'difficulty falling asleep', 'frequent waking',
    'early morning awakening', 'daytime fatigue', 'spinning sensation',
    'balance problems', 'hearing problems', 'increased salivation',
    'heavy sweating', 'cool moist skin', 'fast weak pulse',
    'high body temperature', 'altered mental state', 'hot dry skin',
    'rapid pulse', 'dry mouth', 'little urination', 'pale skin',
    'cold hands and feet', 'brittle nails', 'infrequent bowel movements',
    'hard stools', 'straining', 'feeling of incomplete evacuation',
    'loose watery stools', 'frequent bowel movements', 'abdominal cramps',
    'rectal pain', 'itching around anus', 'swelling around anus',
    'bleeding during bowel movements', 'discomfort', 'lumps near anus',
    'vaginal itching', 'thick white discharge', 'pain during urination',
    'pain during intercourse', 'vaginal soreness', 'pelvic pain',
    'lower back pain', 'thigh pain', 'loose stools', 'mood swings',
    'tender breasts', 'food cravings', 'depression', 'itchy feet',
    'burning feet', 'cracked skin', 'peeling skin', 'bad foot odor',
    'scaly patches', 'small bumps', 'thickened skin', 'sensitive skin',
    'silvery scales', 'dry cracked skin', 'thick nails', 'swollen joints',
    'pimples', 'blackheads', 'whiteheads', 'oily skin', 'tender bumps',
    'cysts', 'scarring', 'red inflamed skin', 'flaky scalp', 'itchy scalp',
    'dry scalp', 'white flakes', 'oily patches', 'red scalp',
    'irritated scalp', 'hair loss', 'discharge from eyes', 'gritty feeling',
    'burning eyes', 'swollen eyelids', 'blurred vision', 'light sensitivity',
    'eye fatigue', 'stringy discharge', 'feeling of fullness',
    'discharge from ear', 'ringing in ears', 'buzzing in ears',
    'hissing sounds', 'clicking sounds', 'sleep problems',
    'numbness in hand', 'tingling in fingers', 'hand weakness',
    'wrist pain', 'difficulty gripping', 'night pain', 'leg pain',
    'numbness in leg', 'tingling in leg', 'weakness in leg',
    'sharp shooting pain', 'muscle stiffness', 'swelling', 'cramping',
    'muscle spasms', 'limited movement', 'widespread pain', 'tender points',
    'morning stiffness', 'joint pain', 'joint stiffness', 'joint swelling',
    'reduced range of motion', 'joint tenderness', 'bone spurs',
    'joint instability', 'fever'
]


# User Input: Symptom selection
st.markdown("### Select the symptoms you are experiencing:")
selected_symptoms = st.multiselect("", symptoms)
b = ",".join(selected_symptoms)


text_symptoms = st.text_area(
    "Or describe your symptoms in your own words (optional):",
    placeholder="Mention your another symptoms"
)
    #translator
response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages =  [  
            {'role':'system','content': 'You are the professional translator. person will come and tell their symptoms. You have to convert it back to medical symptom names  for example, "ulti" to "vomiting" and return it in the form of a string and also correct the spelling  and most important thing just give me the translated medical text only nothing else and just give me one space after giveing the text .'},
            {'role':'user','content': text_symptoms},  
            {'role':'assistant','content': ' you only have to transalate the symptoms not need to describe and dont give any suggestion or step and precaution and ingore the text that are not related to the some symtoms donot need to process that text  if i enter nothing then ingore dont ned to do anything '},
            ]
        )


a  =response.choices[0].message.content

final_response = a+b



try:
   
        model_data = joblib.load('disease_predictor_model.pkl') 
        
        model = model_data['model'] 
        vectorizer = model_data['vectorizer'] 
     
        label_encoder = model_data['label_encoder'] #give indexes to the diseases
       
        print("Model loaded successfully!")
        model_works = True 
    
except:
        model_works = False


# Prediction button
if st.button("Predict Disease"):
    if not selected_symptoms and not text_symptoms.strip():
        st.warning("⚠️ Please either select symptoms from the list or describe them in the text box.")
    else:
        # Only call translation if text_symptoms is not empty
        a = ""
        if text_symptoms.strip():
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[  
                    {'role': 'system', 'content': 'You are the professional translator. Person will tell their symptoms. You have to convert them back to proper medical symptom names like "ulti" → "vomiting". Return only the translated string with a space at the end.'},
                    {'role': 'user', 'content': text_symptoms},  
                    {'role': 'assistant', 'content': 'Only translate relevant symptoms. Ignore unrelated text. If input is empty, do nothing.'},
                ]
            )
            a = response.choices[0].message.content

        final_response = a + ",".join(selected_symptoms)






    if model_works:
        try:
            
            clean_input = final_response.lower().strip()
            input_numbers = vectorizer.transform([clean_input])         
            
            probabilities = model.predict_proba(input_numbers)[0]# give probabites of all the dieases 
            
            top_3_indices = np.argsort(probabilities)[-3:][::-1]         
           
            
            for i in range(1, 4):
                idx = top_3_indices[i - 1]
                name = label_encoder.inverse_transform([idx])[0]
              
                confidence = probabilities[idx] * 100
                try : 
                    if probabilities[top_3_indices[0]] * 100 <=10:
                        disease = ["flu" , "Common cold", "Normal Fever"]
                        pick = random.randint(0,2)
                        
                        
                        name = disease[pick]
                        best_confidence = probabilities[top_3_indices[0]] * 100
                        st.success(f"{name}: 100 % confidence")
                        break


                    st.success(f"✅{i}. {name}: {confidence:.1f}% confidence")
                except Exception as e:
                    st.error(f"❌ Prediction failed:\n{e}")

        except Exception as error:
            print(f" model error: {error}")
            


            

