# Heart Disease Dataset Dictionary

## Overview
This dataset contains medical parameters for diagnosing heart disease, collected from multiple medical centers.

## Features

| Feature   | Description                                            | Type    | Range/Values                               |
|-----------|--------------------------------------------------------|---------|-------------------------------------------|
| age       | Age of the patient in years                            | numeric | 29-77                                      |
| sex       | Gender of the patient                                  | binary  | 0: female, 1: male                         |
| cp        | Chest pain type                                        | nominal | 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic |
| trestbps  | Resting blood pressure (in mm Hg)                      | numeric | 94-200                                     |
| chol      | Serum cholesterol in mg/dl                             | numeric | 126-564                                    |
| fbs       | Fasting blood sugar > 120 mg/dl                        | binary  | 0: false, 1: true                          |
| restecg   | Resting electrocardiographic results                   | nominal | 0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy |
| thalach   | Maximum heart rate achieved                            | numeric | 71-202                                     |
| exang     | Exercise induced angina                                | binary  | 0: no, 1: yes                              |
| oldpeak   | ST depression induced by exercise relative to rest     | numeric | 0-6.2                                      |
| slope     | Slope of the peak exercise ST segment                  | ordinal | 1: upsloping, 2: flat, 3: downsloping     |
| ca        | Number of major vessels colored by fluoroscopy         | ordinal | 0-3                                        |
| thal      | Thalassemia                                            | nominal | 3: normal, 6: fixed defect, 7: reversible defect |
| target    | Presence of heart disease                              | binary  | 0: healthy, 1: disease                     |
| source    | Source of the data                                     | nominal | cleveland, hungarian, switzerland, va      |

## Target Variable
The `target` variable indicates the presence of heart disease. Originally, values ranged from 0-4, where:
- 0: No heart disease
- 1-4: Heart disease present (different degrees of severity)

For this project, the target has been binarized:
- 0: No heart disease (original value 0)
- 1: Heart disease present (original values 1-4)

## Source
The dataset was obtained from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/Heart+Disease
