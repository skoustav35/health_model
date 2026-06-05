import json
import random

# Core medical condition profiles tailored to common Indian clinical presentations and epidemiological data
conditions = [
    {
        "type": "Metabolic (Type 2 Diabetes)", 
        "input": "Patient presents with chronic fatigue, polyuria, and a classic South Asian lean-diabetes phenotype (BMI 23.5 kg/m²). Laboratory evaluation reveals a fasting plasma glucose of 188 mg/dL, a postprandial glucose of 290 mg/dL, and an HbA1c of 9.2%. Spot urine microalbumin-to-creatinine ratio is elevated at 150 mg/g.", 
        "diagnosis": "Uncontrolled Type 2 Diabetes Mellitus with early-stage Diabetic Nephropathy.", 
        "future": "High lifetime risk of rapidly progressing to macroalbuminuria, diabetic retinopathy, and accelerated premature coronary artery disease. Immediate intervention: initiate Metformin 500mg BID coupled with an SGLT2 inhibitor (Empagliflozin 10mg daily) for renal protection. Strict glycemic tracking and fundoscopic eye exam within 4 weeks mandatory."
    },
    {
        "type": "Oncology (Oral Cancer)", 
        "input": "Patient with a 15-year history of smokeless tobacco (gutka/khaini) use presents with a non-healing, indurated 2.5cm ulceration on the left buccal mucosa accompanied by ipsilateral level II cervical lymphadenopathy. Histopathology from a punch biopsy confirms invasive squamous cell carcinoma, moderately differentiated.", 
        "diagnosis": "Stage II (cT2 cN1 M0) Oral Cavity Squamous Cell Carcinoma.", 
        "future": "Elevated risk of local infiltration into the mandible and deep masticator spaces, significantly reducing surgical resectability if delayed. Plan for urgent wide local excision with segmental or marginal mandidbulectomy and comprehensive ipsilateral neck dissection, followed by adjuvant radiation therapy."
    },
    {
        "type": "Oncology (Cervical Cancer)", 
        "input": "Multiporous patient presents with a history of post-coital bleeding, foul-smelling vaginal discharge, and a friable, exophytic mass noted on the ectocervix during speculum examination. Pelvic MRI demonstrates a 3.2cm cervical lesion with parametrial invasion limited to the upper two-thirds of the vagina; no pelvic lymphadenopathy.", 
        "diagnosis": "Stage IIB Cervical Carcinoma.", 
        "future": "High probability of local disease progression resulting in bilateral ureteral obstruction, hydronephrosis, and subsequent renal failure if untargeted. Primary curative strategy requires definitive concurrent chemoradiation therapy (CCRT) utilizing weekly Cisplatin and external beam radiation, followed immediately by high-dose-rate (HDR) brachytherapy."
    },
    {
        "type": "Infectious Disease (Pulmonary Tuberculosis)", 
        "input": "Patient presents with a 3-week history of productive cough, low-grade evening fever peaks, drenching night sweats, and unintentional weight loss of 5kg. Chest X-ray reveals a thick-walled cavitary lesion in the right upper lobe. Sputum smear microscopy is positive (3+) for Acid-Fast Bacilli (AFB).", 
        "diagnosis": "Active Pulmonary Tuberculosis (Sputum Smear Positive).", 
        "future": "High infectious transmission risk within crowded household contacts. Risk of permanent pulmonary fibrosis, bronchiectasis, or hemoptysis if treatment compliance drops. Initiate standard 6-month anti-tubercular therapy (ATT) using fixed-dose combinations (Rifampicin, Isoniazid, Pyrazinamide, Ethambutol) under strict compliance monitoring. Send sample for GeneXpert to rule out drug resistance."
    },
    {
        "type": "Cardiology (Premature CAD)", 
        "input": "Patient presents with acute-onset substernal chest pressure radiating to the left arm. Lipid profile demonstrates a classic South Asian atherogenic dyslipidemia triad: high triglycerides (260 mg/dL), severely low HDL (31 mg/dL), and elevated ApoB. ECG shows 2mm ST-segment elevation in leads V1-V4.", 
        "diagnosis": "Acute Anterior Wall ST-Elevation Myocardial Infarction (STEMI) secondary to premature Coronary Artery Disease.", 
        "future": "High immediate threat of ventricular arrhythmias, cardiogenic shock, or ischemic heart failure if myocardial reperfusion is delayed. Immediate management: execute emergency primary percutaneous coronary intervention (PCI) with drug-eluting stent deployment. Long-term dual antiplatelet therapy (DAPT) and aggressive high-intensity statin titration required."
    }
]

dataset = []

print("Generating 6,500 premium synthetic medical samples for Indian Clinical Profiles...")
for i in range(6500):
    cond = random.choice(conditions)
    age = random.randint(28, 76)
    gender = random.choice(["Male", "Female"])
    
    instruction = f"Analyze the following clinical data for a {age}-year-old {gender} client. Identify the current underlying disease or condition and project potential future clinical outcomes or complications.\n\nData Context ({cond['type']}): {cond['input']}"
    response = f"### CURRENT DIAGNOSIS:\n{cond['diagnosis']}\n\n### FUTURE PROJECTIONS & PROGNOSIS:\n{cond['future']}"
    
    combined_text = f"User: {instruction}\nAssistant: {response}"
    dataset.append({"text": combined_text})

output_filename = "6500_samples.json"
with open(output_filename, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Successfully generated '{output_filename}' with 6,500 high-quality records!")
