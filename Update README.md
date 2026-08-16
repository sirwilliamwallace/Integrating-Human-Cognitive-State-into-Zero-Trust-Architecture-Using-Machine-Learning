# Neurobehavioral Resilience within Zero Trust Architecture

A human-adaptive defence proof-of-concept integrating cognitive-state data into a Zero Trust-style policy model.

**Requirements:**
- Python 3.12

**Dataset**
- Option A: extract the dataset from the original repository: https://github.com/sirwilliamwallace/Integrating-Human-Cognitive-State-into-Zero-Trust-Architecture-Using-Machine-Learning
- Option B: download the Pillai dataset from Mendeley: https://data.mendeley.com/preview/dp8g983t38?a=1a80ae60-6591-4a0a-b20a-4be2de002df3
- Extract the dataset into the project `Data` folder (path: `./Data`).

**Install dependencies**
Install the Python requirements with:

```bash
pip install -r requirements.txt
```

**Run**
- Notebook: start Jupyter and open `main.ipynb`:

```bash
jupyter notebook main.ipynb
```

- Script: run the main script directly:

```bash
python script.py
```

**Expected outputs**
Running the notebook or script will create an `Evaluations-Figures` folder containing the analysis figures:
- `1_confusion_matrix.png`, `1_confusion_matrix.pdf`
- `2_roc_auc_curve.png`, `2_roc_auc_curve.pdf`
- `3_feature_importance.png`, `3_feature_importance.pdf`
- `4_pupil_delta_distribution.png`, `4_pupil_delta_distribution.pdf`
- `5_zta_policy_distribution.png`, `5_zta_policy_distribution.pdf`
    
**Author**
- Amirhossein Shekooh
- York St John University
- Date: Aug 2026

**Disclaimer**
This repository contains the implementation developed for an undergraduate dissertation. It is a research proof-of-concept and not a production-ready Zero Trust implementation.
