# Project-Panopticon-Proctoring-AI
Time-series ML pipeline for exam proctoring — asynchronous data merging, rolling window features, and ethical AI threshold tuning
# Project Panopticon: Intelligent Exam Proctoring AI

A time-series classification pipeline that detects suspicious behavior 
during online exams while minimizing false accusations of innocent students.

## Key Challenges Solved
- Asynchronous merge of irregular event logs with continuous sensor data (`merge_asof`)
- Missing data imputation for dropped video frames
- Rolling window feature engineering to filter out noise (sneezes, brief movements)
- Class-imbalanced model training with `class_weight='balanced'`
- Ethical AI threshold tuning using `predict_proba()` — shifted decision 
  boundary to 90% confidence to eliminate false positives

## Results
- Default threshold (50%): 62 false accusations
- Strict threshold (90%): 0 false accusations
- Precision-Recall analysis included

## Tech Stack
Python, Pandas, Scikit-learn (RandomForest), Matplotlib, Seaborn
