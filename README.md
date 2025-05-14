Transformer-Based Model for Contextual Feedback Validation for Zero Education Center

Overview
This project provides an end-to-end solution for automatically validating student feedback using a transformer-based NLP model (BERT) and a Streamlit web interface. The system classifies feedback as "valid" or "invalid" to streamline customer care operations and improve actionable insights.

Features
Fine-tuned BERT model for contextual feedback validation

Easy-to-use Streamlit web app for real-time prediction

Simple retraining pipeline for new datasets

Supports CSV-based data ingestion and output

Table of Contents

1. Project Structure

2. Installation

3. Usage

4. Training the Model

5. Running the Streamlit App

6. Sample Data Format

7. Customization

1. Project Structure.

.
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── my_feedback_model/      # Directory for saved model and tokenizer
├── train.csv               # Training dataset
├── val.csv                 # Validation dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

2. Installation
a. Clone the repository:

git clone https://github.com/ASWINI-2004-WEB/feedback-validation-transformer
cd feedback-validation-transformer

b. Create and activate a virtual environment (recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

3. Usage
a. Prepare Your Data
Ensure train.csv and val.csv are present in the project root, formatted as below.

4. Train the Model

python train.py

this will fine-tune the BERT model and save it in the my_feedback_model/ directory.

5. Run the Streamlit App

streamlit run app.py

Open your browser to the local URL provided (typically http://localhost:8501).

6. Sample Data Format
Both train.csv and val.csv should have the following columns:

text,label
"satisfied with the service,good quality of service",1
"can you reactivate my subscription please,unable to unsubscribe",0
...
text: Customer feedback (string)

label: 1 = valid, 0 = invalid

7. Customization
Model: You can switch to other transformer models (e.g., RoBERTa) by editing train.py and app.py.

Thresholds: Adjust classification thresholds or add more classes as needed.

Deployment: The app can be containerized or deployed to cloud services for production use.

