# Neurobehavioral Resilience within Zero Trust Architecture:  A Human Adaptive Defence Framework

## Adaptive Zero Trust Architecture

### Requirements

- Python 3.12

#### Dataset

- Extract the dataset from this repository 

OR

- Download the Pillai dataset from: https://data.mendeley.com/preview/dp8g983t38?a=1a80ae60-6591-4a0a-b20a-4be2de002df3

- Extract it to:

```text
/Data
```

- Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run?

Run the notebook:

```bash
jupyter notebook main.ipynb
```

Or run the script:

```bash
python script.py
```

## Expected outputs

- A folder named Output will get created containing:
    - 1_confusion_matrix.png
    - 1_confusion_matrix.pdf
    - 2_roc.png
    - 2_roc.pdf
    - 3_feature_importance.png
    - 3_feature_importance.pdf
    - 4_pupil_delta_distribution.png
    - 4_pupil_delta_distribution.pdf
    - 5_policy_distribution.png
    - 5_policy_distribution.pdf


```
Author: Amirhossein Shekooh
Date: Aug 2026
York St John University
```

Disclaimer

This repository contains the implementation developed for an undergraduate dissertation at York St John University. It is intended as a research proof-of-concept and should not be considered a production-ready Zero Trust implementation.