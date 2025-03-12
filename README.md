# 🩺 Medical Text Analyzer  

A **Streamlit-based Medical NLP Analyzer** that extracts medical details, summarizes text, and performs sentiment & intent analysis on patient conversations.  

## 🚀 Features  
✅ Extracts **Symptoms, Diagnosis, Treatment, and Prognosis** from medical text  
✅ **Summarizes** long-form patient conversations  
✅ **Sentiment & Intent Analysis** for patient responses  
✅ **Generates SOAP Notes** for structured medical documentation  
✅ **File Upload Support** for `.txt` and `.csv` files  
✅ **Download JSON Report** for extracted insights  

---

## 🛠️ Setup & Installation  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/sOnU1002/Notes-analyser.git  
```
##Install Dependencies
```
pip install -r requirements.txt  

```
##Download and Setup Medical Models
```
python -m spacy download en_core_sci_md
```
##Run the Streamlit App
```
streamlit run app.py
```

## 📝 Methodologies Used  

### **1️⃣ Named Entity Recognition (NER)**  
- Uses **SciSpaCy** for extracting medical entities from text.  
- **BioBERT** enhances NER by detecting **diseases, symptoms, and treatments** with higher accuracy.  

### **2️⃣ Text Summarization**  
- Uses the **Facebook BART Large Model** for generating concise summaries of medical text.  
- Ensures key medical details are retained while reducing text length.  

### **3️⃣ Sentiment & Intent Analysis**  
- Utilizes a **DistilBERT-based sentiment classifier** to analyze patient responses.  
- Identifies whether the sentiment is **Anxious or Reassured** and categorizes patient intent (e.g., Reporting symptoms, Seeking reassurance).  

### **4️⃣ SOAP Note Generation**  
- Converts extracted medical data into the **SOAP (Subjective, Objective, Assessment, Plan) format** for structured medical documentation.  

![Medical NLP App]([https://i.imgur.com/example.png](https://github.com/sOnU1002/Notes-analyser/blob/main/notes/Screenshot%20(28).png))
![Medical NLP App]([https://i.imgur.com/example.png]([https://github.com/sOnU1002/Notes-analyser/blob/main/notes/Screenshot%20(28).png](https://github.com/sOnU1002/Notes-analyser/blob/main/notes/Screenshot%20(29).png)))



