import streamlit as st
import joblib
import numpy as np


model = joblib.load("models/linear_model2.pkl")  


# CSS personnalisé
st.markdown(
    """
    <style>
    /* Fond de l'app */
body, .stApp {
    background-color: #3b1c01 !important;
    color: white;
}

    header.stAppHeader {
        background-color: #3b1c01 !important;  
    }

    

    div[data-baseweb="input"] > div {
    border: none !important;      /* plus de bordure */
    box-shadow: none !important;  /* plus d’effet 3D ou ombre */
    background: none !important;  /* si tu veux un fond transparent */
}

    /* Inputs (number_input, text_input, etc.) */
    div[data-baseweb="input"] > div > input {
        background-color: #c49770 !important; /* fond clair */
        color: #8B4513 !important;           /* texte marron foncé */
        border: 2px solid #FFF5E1 !important;
        border-radius: 8px !important;
        padding: 5px !important;
        font-size: 16px !important;
       

    }
    


    button[data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"] {
    background-color: #c49770 !important; 
    color: #8B4513 !important;            /* texte marron foncé */
    border-radius: 4px !important;
    width: 36px !important;                /* largeur plus grande */
    height: 36px !
}


    /* Bouton */
    div.stButton > button {
        background-color: #ffc400;  
        color: #8B4513 !important;          
        font-size: 18px !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
    }

    /* Texte de succès */
    div.stAlert > div[data-testid="stMarkdownContainer"] p {
        color: white !important;
        font-size: 20px !important;
    }


div.stAlert div[data-baseweb="notification"][data-testid="stAlertContainer"] {
    background-color: #2ecc71 !important; /* vert vif */
    color: white !important;              /* texte blanc */
    border-radius: 10px !important;       /* coins arrondis */
    padding: 15px !important;             /* padding pour tout le container */
    font-size: 18px !important;           /* texte plus grand */
    box-shadow: none !important;          /* enlever ombre si présente */
}


    </style>
    """,
    unsafe_allow_html=True
)



st.title("House price prediction")

square_meters = st.number_input("Surface (m²)", min_value=10.0, max_value=1000.0, step=1.0)
age = st.number_input("Age (years)", min_value=0.0, max_value=200.0, step=1.0)
distance = st.number_input("Distance to city (km)", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict the price"):
    
    X = np.array([[square_meters, age, distance]])
    
   
    y_pred = model.predict(X)[0]
    
    st.success(f"Price estimated : {y_pred:,.2f}")
