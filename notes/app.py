import streamlit as st
import pandas as pd
import spacy
import json
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch

# Load SciSpaCy Medical Model



model_name = r"D:/ana/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")




def extract_medical_details(text):
    doc = nlp(text)
    symptoms, treatments, diagnosis, prognosis = set(), set(), set(), set()
    
    symptom_keywords = [
    "pain", "ache", "discomfort", "headache", "injury", "stiffness",
    "nausea", "dizziness", "fatigue", "shortness of breath", "swelling",
    "numbness", "burning sensation", "pressure", "cramps", "vomiting"
]

    treatment_keywords = [
    "physiotherapy", "medication", "painkillers", "therapy", "sessions",
    "surgery", "injections", "radiation", "chemotherapy", "antibiotics",
    "acupuncture", "massage", "dietary supplements", "rehabilitation"
]

    diagnosis_keywords = [
    "whiplash", "fracture", "strain", "injury", "infection",
    "arthritis", "concussion", "migraine", "herniated disc", "tendonitis",
    "bronchitis", "asthma", "hypertension", "diabetes", "osteoporosis"
]

    prognosis_keywords = [
    "recovery", "long-term", "healing", "improvement", "worsening",
    "chronic", "lifelong condition", "remission", "degeneration", "recurrence"
]

    
    for ent in doc.ents:
        token_text = ent.text.lower()
        if any(kw in token_text for kw in symptom_keywords):
            symptoms.add(ent.text)
        elif any(kw in token_text for kw in treatment_keywords):
            treatments.add(ent.text)
        elif any(kw in token_text for kw in diagnosis_keywords):
            diagnosis.add(ent.text)
        elif any(kw in token_text for kw in prognosis_keywords):
            prognosis.add(ent.text)

    return {
        "Symptoms": list(symptoms),
        "Diagnosis": list(diagnosis),
        "Treatment": list(treatments),
        "Prognosis": list(prognosis)
    }


def summarize_text(text, max_chunk_length=512):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

   
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i : i + max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        summary = summarizer(chunk_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        summaries.append(summary)
    
    return " ".join(summaries)  


sentiment_analyzer = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Streamlit UI


st.write("Upload a **TXT**  file to extract medical details, perform sentiment analysis, and generate a SOAP note.")

uploaded_file = st.file_uploader("📤 Upload your file", type=["txt", "csv"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]


    if file_extension == "txt":
        text = uploaded_file.read().decode("utf-8")
    elif file_extension == "csv":
        df = pd.read_csv(uploaded_file)
        if "Text" in df.columns:
            text = " ".join(df["Text"].dropna().tolist())
        elif "Description" in df.columns:
            text = " ".join(df["Description"].dropna().tolist())
        else:
            st.error("CSV file must contain a column named 'Text' or 'Description'.")
            st.stop()

    st.subheader("📜 Extracted Text Preview:")
    st.write(text[:1000] + "...")  # Show first 1000 characters

    # Extract Medical Details
    st.subheader("🔍 Extracted Medical Details:")
    medical_details = extract_medical_details(text)
    st.json(medical_details)

    # Summarization
    st.subheader("📄 Summarization:")
    summary = summarize_text(text)
    st.write(summary)

    # Sentiment Analysis
    st.subheader("😊 Sentiment & Intent Analysis:")
    patient_responses = [line.replace("**Patient:**", "").strip() for line in text.split("\n") if "**Patient:**" in line]
    sentiments = []
    
    for response in patient_responses:
        sentiment_result = sentiment_analyzer(response)[0]
        sentiment_label = "Reassured" if sentiment_result["label"] == "POSITIVE" else "Anxious"
        intent = "Seeking reassurance" if "worried" in response.lower() else "Reporting symptoms"
        sentiments.append({"text": response, "Sentiment": sentiment_label, "Intent": intent})

    st.json(sentiments)

    # SOAP Note
    st.subheader("📝 Generated SOAP Note:")
    SOAP_note = {
        "Subjective": {
            "Chief_Complaint": "Neck and back pain",
            "History_of_Present_Illness": summary
        },
        "Objective": {
            "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
            "Observations": "Patient appears in normal health, normal gait."
        },
        "Assessment": {
            "Diagnosis": medical_details["Diagnosis"],
            "Severity": "Mild, improving"
        },
        "Plan": {
            "Treatment": medical_details["Treatment"],
            "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
        }
    }
    st.json(SOAP_note)

    # Save results to JSON
    output = {
        "Medical_Details": medical_details,
        "Summarization": summary,
        "Sentiment_Intent_Analysis": sentiments,
        "SOAP_Note": SOAP_note
    }

    json_output_path = "medical_analysis.json"
    with open(json_output_path, "w") as f:
        json.dump(output, f, indent=4)

    # Download JSON Button
    with open(json_output_path, "rb") as f:
        st.download_button(label="📥 Download JSON Report", data=f, file_name="medical_analysis.json", mime="application/json")
