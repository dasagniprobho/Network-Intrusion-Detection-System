# 🛡️ Network Intrusion Detection System (NIDS)

## 📌 Overview

The **Network Intrusion Detection System (NIDS)** is a machine learning
project developed using the **CIC-IDS2017 cybersecurity dataset**. It
analyses network-flow records to distinguish benign traffic from
malicious activity and classify traffic into broader attack families.

The project combines statistical exploratory data analysis, data
cleaning, feature engineering, feature selection, supervised
classification, anomaly detection and model validation. An interactive
**Streamlit dashboard** provides a graphical interface for exploring the
data, training and evaluating models, and obtaining flow-level
predictions.

The repository is intended as an experimental and analytical framework
for studying machine-learning approaches to network intrusion detection.
It is not a live packet-monitoring or production security system.

## 🎯 Objectives

-   Explore the statistical properties and class distribution of network
    traffic.
-   Clean and prepare CIC-IDS2017 flow records for analysis.
-   Engineer useful indicators and consolidate attack labels into seven
    classes.
-   Investigate multicollinearity using Spearman correlation and
    Variance Inflation Factor (VIF).
-   Train and compare multiple classification approaches.
-   Examine class imbalance, destination-port shortcuts and possible
    data leakage.
-   Evaluate generalisation using session-held-out validation.
-   Provide an interactive dashboard for model evaluation and flow-level
    analysis.

## 📊 Dataset

**CIC-IDS2017** was developed by the Canadian Institute for
Cybersecurity at the University of New Brunswick.

-   **Dataset:** CIC-IDS2017
-   **Scale:** Approximately 2.83 million network flows
-   **Format:** CSV flow records
-   **Task:** Multiclass network traffic classification
-   **Target:** Original traffic label, subsequently consolidated into
    seven broader classes

The dataset contains benign traffic and multiple attack types captured
across different days and network scenarios. Its class distribution is
highly imbalanced, making class-wise evaluation important.

**Dataset source:** [CIC-IDS2017 -- Canadian Institute for
Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

## 🧭 Workflow

``` text
CIC-IDS2017 CSV files
        |
        v
Exploratory Data Analysis
        |
        v
Data Cleaning and Integrity Checks
        |
        v
Feature Engineering and Label Grouping
        |
        v
Train-Test Partitioning
        |
        v
Feature Selection
(Spearman Correlation and VIF)
        |
        v
Preprocessing
        |
        v
Model Training and Evaluation
        |
        +-------------------------------+
        |               |               |
        v               v               v
     XGBoost           MLP           Ensemble
        |               |               |
        +---------------+---------------+
                        |
                        v
             Validation and Diagnostics
                        |
          +-------------+--------------+
          |             |              |
          v             v              v
       Port Test    Leakage Checks   Session-held-out
                                      Evaluation
                        |
                        v
               Isolation Forest
                        |
                        v
             Streamlit Flow Analysis
```

## 🔬 Main Features and Methodology

### 1. Exploratory Data Analysis

The EDA notebook investigates dataset structure, traffic and
attack-label distributions, numerical feature characteristics, capture
sessions, destination ports and data-quality issues. Descriptive
summaries and visualisations are used to understand the dataset before
modelling.

### 2. Data Cleaning

The preprocessing workflow includes checks for missing and infinite
values, invalid records, constant or redundant columns and duplicate
observations. These checks help establish a consistent dataset for
subsequent analysis.

### 3. Feature Engineering and Attack Grouping

Additional indicators are created for selected network-flow
characteristics, including unknown TCP initial-window values and
Active/Idle feature availability. Destination ports can also be grouped
into broader service categories.

The original labels are consolidated into seven broader classes:

  -----------------------------------------------------------------------
  Class                               Description
  ----------------------------------- -----------------------------------
  `BENIGN`                            Legitimate network traffic

  `DoS_DDoS`                          Denial-of-service and distributed
                                      denial-of-service attacks

  `Reconnaissance`                    Port-scanning activity

  `Brute_Force`                       FTP, SSH and web brute-force
                                      activity

  `Web_Attack`                        Web attacks such as XSS and SQL
                                      injection

  `Bot`                               Bot-related traffic

  `Other_Rare`                        Other infrequent attack categories,
                                      including Heartbleed and
                                      Infiltration
  -----------------------------------------------------------------------

This grouping simplifies the multiclass task while retaining broad
distinctions between attack families.

### 4. Feature Selection and Multicollinearity

Two statistical techniques are used to investigate redundant numerical
predictors:

-   **Spearman rank correlation:** identifies strongly correlated
    feature pairs. A configurable absolute-correlation threshold is used
    to flag redundancy.
-   **Variance Inflation Factor (VIF):** examines multicollinearity
    among the remaining numerical predictors. Features exceeding the
    configured threshold can be removed iteratively.

Feature-selection steps are designed to use training data rather than
information from the held-out test data.

### 5. Data Preprocessing

Numerical and categorical variables are handled separately. Depending on
the selected pipeline, numerical processing includes imputation,
transformation and scaling, while categorical variables are imputed and
one-hot encoded. Preprocessing is fitted on the relevant training
partition to reduce the risk of information leakage.

### 6. Classification Models

The modelling workflow includes:

-   **XGBoost:** gradient-boosted decision trees for multiclass
    classification, with configurable training parameters and
    class-imbalance handling.
-   **Multi-Layer Perceptron (MLP):** a neural-network classifier
    implemented using NumPy, with configurable architecture and
    regularisation options.
-   **Ensemble:** combines model predictions using configurable weights.
    Class-specific threshold analysis can also be performed where
    supported by the training workflow.
-   **Logistic Regression baseline:** provides a simpler reference model
    for selected comparative experiments.

### 7. Model Evaluation

Model performance is examined using measures such as:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Macro-F1
-   Confusion matrix
-   Class-wise performance

Macro-F1 is particularly relevant because it gives each class equal
weight and therefore helps reveal performance on less frequent attack
categories. Results should be interpreted alongside class-wise metrics
and the evaluation split used.

### 8. Destination-Port Shortcut Analysis

The project investigates how much classification performance depends on
destination-port information. Experiments compare feature configurations
that include raw destination-port information, exclude it, or retain
grouped port information.

This analysis helps assess whether a model may be relying heavily on a
convenient feature rather than learning broader traffic characteristics.

### 9. Leakage Checks and Session-Held-Out Validation

Additional diagnostics investigate potentially problematic predictors,
overlapping records or feature patterns, and the influence of capture
sessions.

The project also includes **Leave-One-Session-Out (LOSO) evaluation**,
in which a capture session is held out while the model is trained on the
remaining sessions. This provides a more demanding assessment of
performance on a session not used for training than a conventional
random split.

### 10. Anomaly Detection

An **Isolation Forest** component is included as an unsupervised
anomaly-detection approach. It provides a complementary way to flag
flows that differ from the benign traffic used to fit the detector.
Anomaly scores should not be treated as definitive attack labels without
further validation.

### 11. Flow-Level Prediction and Explainability

The Streamlit application includes a flow-level analysis interface.
Depending on the selected data and available model bundle, users can
inspect a flow's predicted class, associated probabilities or confidence
information, and feature-level explanations such as SHAP-based
contributions for supported tree models.

CSV-based flow analysis is also supported by the application workflow.

## 🖥️ Streamlit Dashboard

The repository includes an interactive dashboard implemented in
`app.py`. Its sections cover the main stages of the project:

  -----------------------------------------------------------------------
  Section                             Purpose
  ----------------------------------- -----------------------------------
  Overview                            Project introduction and workflow

  Data                                Dataset loading, inspection and
                                      label distributions

  Cleaning                            Data-quality checks and cleaning
                                      operations

  Features                            Feature engineering, partitioning
                                      and feature selection

  Train & Evaluate                    Model training and performance
                                      assessment

  Port Test & Baseline                Destination-port experiments and
                                      baseline comparison

  Validation                          Validation diagnostics, anomaly
                                      detection and session-held-out
                                      evaluation

  Live Detection                      Flow-level prediction and supported
                                      explanation tools
  -----------------------------------------------------------------------

The application can use the data-loading options provided in its
interface, including uploaded dataset files and demonstration data where
available. Demonstration or synthetic data is intended for testing the
interface only; it must not be interpreted as real CIC-IDS2017 model
performance.

## 📁 Repository Structure

``` text
Network-Intrusion-Detection-System/
│
├── .streamlit/
│   └── Streamlit configuration
│
├── Network_Intrusion_Detection_Analysis.ipynb
│   └── Exploratory data analysis
│
├── NIDS_model_final.ipynb
│   └── Model development, training and evaluation
│
├── nids_core.py
│   └── Core data processing and modelling functions
│
├── app.py
│   └── Streamlit dashboard
│
├── ids_demo_bundle.joblib
│   └── Serialised demonstration/inference bundle
│
├── NIDS_Statistical_Report_1.pdf
│   └── Statistical analysis report
│
├── NIDS_Model_Training_and_Evaluation_Report.pdf
│   └── Model training and evaluation report
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
└── README.md
```

## ⚙️ Technologies

  Technology             Role
  ---------------------- ------------------------------------------------
  Python                 Main programming language
  Pandas                 Data manipulation
  NumPy                  Numerical computing and MLP implementation
  SciPy                  Statistical computation
  Scikit-learn           Preprocessing, machine learning and evaluation
  XGBoost                Gradient-boosting classifier
  Matplotlib / Seaborn   Data visualisation
  Streamlit              Interactive dashboard
  Joblib                 Model-bundle serialisation

See `requirements.txt` for the repository's dependency list.

## 🚀 Installation and Usage

### Prerequisites

-   Python version compatible with the packages in `requirements.txt`
-   Git
-   pip

### 1. Clone the repository

``` bash
git clone https://github.com/dasagniprobho/Network-Intrusion-Detection-System.git
cd Network-Intrusion-Detection-System
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**

``` bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the dashboard

``` bash
streamlit run app.py
```

Open the local address printed by Streamlit in your browser (usually
`http://localhost:8501`).

### 5. Load data and explore

Use the dashboard's data controls to select an available demonstration
source or upload the required CIC-IDS2017 CSV data. Follow the
application workflow to inspect the data, prepare features, train models
and review evaluation or flow-level outputs.

**Dataset note:** The full CIC-IDS2017 dataset is not assumed to be
included in this repository. Download it from the official source and
follow the input requirements shown by the application.

## ⚠️ Limitations

-   Results on CIC-IDS2017 may not generalise directly to different
    networks, devices or contemporary attack patterns.
-   The seven-class grouping combines multiple original labels and
    therefore does not identify every individual attack subtype.
-   Strong class imbalance can make rare-attack performance difficult to
    estimate reliably.
-   Random train-test evaluation and session-held-out evaluation measure
    different forms of generalisation; results should be reported with
    the split clearly identified.
-   Isolation Forest anomaly scores indicate unusual observations, not
    confirmed malicious activity.
-   The dashboard analyses supplied flow records; it is not, by itself,
    a live packet-capture system or production network defence platform.
-   Demonstration data and demonstration model artefacts must not be
    confused with validated real-world detection performance.

## 🔭 Future Scope

Possible extensions include live network-flow ingestion, evaluation on
additional intrusion-detection datasets, online learning, improved
detection of previously unseen attacks, further explainability studies
and integration with network-monitoring or security-information systems.

## 📚 References

1.  Canadian Institute for Cybersecurity. *CIC-IDS2017 Dataset*.
    University of New Brunswick.
    https://www.unb.ca/cic/datasets/ids-2017.html

## 👨‍💻 Project Information

-   **Project:** Network Intrusion Detection System
-   **Domain:** Cybersecurity, Statistical Analysis and Machine Learning
-   **Dataset:** CIC-IDS2017
-   **Language:** Python
-   **Dashboard:** Streamlit
-   **Repository:**
    [dasagniprobho/Network-Intrusion-Detection-System](https://github.com/dasagniprobho/Network-Intrusion-Detection-System)
-   **Project Mentor:** Arpan Roy
-   **Team Members(Collaborators):**
      1. Agniprobho Das (dasagniprobho)
      2. Rajanya Maity (Rajanya-Maity)
      3. Yagyaseni Chakraborty (chakrabortyyagyaseni464-dev)


------------------------------------------------------------------------

If you find this project useful or interesting, consider starring the
repository.
