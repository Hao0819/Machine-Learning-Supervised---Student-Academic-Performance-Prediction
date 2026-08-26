# Student Academic Performance Prediction

Supervised machine learning project for **BMCS2203 Artificial Intelligence** (Session 202605). The project uses student-related factors to classify academic performance as **Low**, **Medium**, or **High**, and compares three classification algorithms under a shared evaluation setup.

## Project overview

Educational institutions collect academic, personal, family, and school-related data, but these data may not be fully used to identify students who need support. This project investigates whether supervised machine learning can classify student performance from those factors.

The project implements and compares:

- Decision Tree
- K-Nearest Neighbors (KNN)
- Logistic Regression

Each group member is responsible for one classification method. The models are evaluated using accuracy, balanced accuracy, precision, recall, F1-score, and confusion matrices.

## Project objectives

1. Build a supervised multiclass classification solution for student academic performance.
2. Prepare the dataset by handling missing values, encoding categorical data, and scaling numerical data where required.
3. Implement Decision Tree, KNN, and Logistic Regression models.
4. Tune model hyperparameters using stratified cross-validation.
5. Compare the models using a common held-out test set and appropriate classification metrics.
6. Identify the most suitable model for this dataset.

## Team responsibilities

| Team member | Assigned model |
|---|---|
| Gan Koh Jun | Decision Tree |
| Lim Jun Hao | K-Nearest Neighbors (KNN) |
| Tan Keng Ting | Logistic Regression |

## Problem formulation

This is a **supervised multiclass classification** problem.

### Input: `X`

`X` contains 19 student-related features:

| Category | Features |
|---|---|
| Academic | `Hours_Studied`, `Attendance`, `Previous_Scores`, `Tutoring_Sessions`, `Extracurricular_Activities` |
| Personal | `Sleep_Hours`, `Motivation_Level`, `Internet_Access`, `Physical_Activity`, `Learning_Disabilities`, `Gender`, `Peer_Influence` |
| Family | `Parental_Involvement`, `Family_Income`, `Parental_Education_Level` |
| School/resources | `Access_to_Resources`, `Teacher_Quality`, `School_Type`, `Distance_from_Home` |

### Output: `y`

The output is the `Performance` class created from `Exam_Score`:

| Exam score | Performance class |
|---:|---|
| 64 or below | Low |
| 65-69 | Medium |
| 70 or above | High |

`Exam_Score` is removed from `X` because it is used to create the target. Keeping it as an input would reveal the answer and cause target leakage.

## Dataset

The included `StudentPerformanceFactors.csv` contains:

- 6,607 student records
- 20 original attributes
- Numerical and categorical variables
- No fully duplicated rows in the current dataset

Missing values are present in three categorical attributes:

| Attribute | Missing values |
|---|---:|
| `Teacher_Quality` | 78 |
| `Parental_Education_Level` | 90 |
| `Distance_from_Home` | 67 |

The project documentation identifies the dataset as a public Kaggle dataset. The exact original dataset page should also be cited in the final report and AI/source disclosure materials.

## Shared experimental design

All models use the same target definition and the same stratified train-test split:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

- 80% training data
- 20% held-out testing data
- 1,322 test records
- Class order fixed as `Low`, `Medium`, `High`
- Five-fold `StratifiedKFold` cross-validation for hyperparameter selection
- Preprocessing fitted inside each model Pipeline to prevent data leakage

The held-out test set is used for final evaluation, not for fitting preprocessing steps or model parameters.

## Data preprocessing

### Numerical features

- Missing values are imputed using the training-set median.
- `StandardScaler` is used for KNN and Logistic Regression because they are sensitive to feature scale.
- Decision Tree does not require numerical scaling.

### Categorical features

- Missing values are imputed using the most frequent training value.
- `OneHotEncoder(handle_unknown="ignore")` converts categorical values into numerical indicators without imposing an artificial order.

### Pipelines

`ColumnTransformer` and `Pipeline` combine preprocessing and model training. This ensures that imputation, encoding, scaling, and feature selection learn only from the relevant training data during cross-validation.

### Feature engineering

The Decision Tree notebook derives two additional features from existing columns:

- `Support_Index` sums the ordinal codes of `Parental_Involvement`, `Access_to_Resources`, and `Teacher_Quality`.
- `Study_Consistency` multiplies `Hours_Studied` by `Attendance` and divides by 100.

Both are computed row by row, so no information crosses between records and the train/test split remains valid.

## Model implementations

### Decision Tree

The Decision Tree learns hierarchical decision rules from student features. `GridSearchCV` compares:

- Split criterion: Gini or entropy
- Maximum tree depth
- Minimum samples required to split a node
- Minimum samples required in a leaf
- Cost-complexity pruning parameter `ccp_alpha`

Best parameters in the current run:

```text
criterion = gini
max_depth = None
min_samples_leaf = 5
min_samples_split = 2
ccp_alpha = 0.0005
```

Instead of limiting the depth in advance, the selected model grows the tree fully and then
prunes it with `ccp_alpha`, which produced a higher cross-validated macro F1 (0.7699).

### K-Nearest Neighbors

KNN classifies a student using nearby training records. Numerical scaling is required because KNN calculates distances.

The KNN Pipeline also uses `SelectKBest(f_classif)` to remove weaker transformed features from the distance calculation. `GridSearchCV` compares:

- Number of selected transformed features
- Number of neighbours
- Uniform or distance-based voting
- Manhattan or Euclidean distance

Best parameters in the current run:

```text
selected transformed features = 12
n_neighbors = 15
weights = distance
p = 2 (Euclidean distance)
```

The selected transformed features mainly originate from attendance, study hours, previous scores, tutoring sessions, access to resources, parental involvement, learning disabilities, and parental education level.

### Logistic Regression

Multiclass Logistic Regression estimates the probability of each performance class. Numerical features are standardized before training. `GridSearchCV` compares:

- Regularization strength `C`
- Solver
- Class weighting

Best parameters in the current run:

```text
C = 100
solver = lbfgs
class_weight = None
```

## Evaluation metrics

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Balanced accuracy | Average recall across Low, Medium, and High |
| Precision | How often a predicted class is correct |
| Recall | How many actual members of a class are identified |
| F1-score | Balance between precision and recall |
| Macro average | Gives each class equal importance |
| Weighted average | Weights each class by its number of test records |
| Confusion matrix | Shows correct predictions and specific class errors |

Macro metrics and balanced accuracy are important because the Medium class contains more records than Low and High.

## Current results

All values below come from the saved outputs in the current notebooks.

| Model | Test accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 77.31% | 75.76% | 76.72% | 75.76% | 76.20% | 77.22% |
| KNN | 82.30% | 77.89% | 85.28% | 77.89% | 80.68% | 81.99% |
| Logistic Regression | **96.82%** | **96.45%** | **96.91%** | **96.45%** | **96.68%** | **96.82%** |

Logistic Regression currently produces the strongest overall performance on this dataset. KNN exceeds 80% test accuracy after feature selection. Decision Tree provides more interpretable decision rules but has lower predictive performance in the current experiment.

### Confusion matrices

Rows are actual classes and columns are predicted classes.

#### Decision Tree

| Actual / Predicted | Low | Medium | High |
|---|---:|---:|---:|
| Low | 194 | 97 | 0 |
| Medium | 74 | 567 | 65 |
| High | 1 | 63 | 261 |

#### KNN

| Actual / Predicted | Low | Medium | High |
|---|---:|---:|---:|
| Low | 195 | 96 | 0 |
| Medium | 20 | 651 | 35 |
| High | 2 | 81 | 242 |

#### Logistic Regression

| Actual / Predicted | Low | Medium | High |
|---|---:|---:|---:|
| Low | 280 | 11 | 0 |
| Medium | 6 | 690 | 10 |
| High | 2 | 13 | 310 |

## Repository structure

```text
.
|-- README.md
|-- requirements.txt
|-- StudentPerformanceFactors.csv
|-- decisionTree.ipynb
|-- KNN.ipynb
|-- LogisticRegression.ipynb
|-- app.py                 web interface (Logistic Regression), needs Python + Flask
|-- run_app.bat            Windows launcher for app.py
|-- predictor.html         standalone interface, double-click to open, no install needed
|-- build_predictor.py     regenerates predictor.html from the trained model
|-- verify_predictor.js    checks predictor.html against scikit-learn (needs Node)
|-- templates/
|   |-- layout.html
|   |-- index.html
|   `-- result.html
`-- static/
    `-- style.css
```

Jupyter checkpoint files and local `.bak` files are development artifacts and are not required for submission.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Flask (only for the web interface)

Install everything with:

```bash
pip install -r requirements.txt
```

Or install the packages individually:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter flask
```

## How to run

1. Clone or download this repository.
2. Keep `StudentPerformanceFactors.csv` in the same directory as the notebooks.
3. Start Jupyter Notebook or JupyterLab.
4. Open one of the model notebooks.
5. Select **Restart Kernel and Run All Cells**.
6. Wait for `GridSearchCV` to complete.
7. Review the best parameters, classification report, evaluation metrics, and confusion matrix.

Example:

```bash
git clone https://github.com/Hao0819/Machine-Learning-Supervised---Student-Academic-Performance-Prediction.git
cd Machine-Learning-Supervised---Student-Academic-Performance-Prediction
jupyter notebook
```

KNN performs 960 cross-validation fits in its current search grid, so it may take longer to run than the other notebooks.

## How to run the web interface

`app.py` is a small Flask application that lets a student enter their own factors in a browser
and receive a predicted performance class together with an explanation of the result. It loads
the same pipeline, the same engineered features, and the same tuned hyperparameters used in
`LogisticRegression.ipynb`, so the prediction on screen matches the notebook.

**Windows:** double-click `run_app.bat`.

**Any operating system:** open a terminal in this folder and run

```bash
pip install -r requirements.txt
python app.py
```

The window prints the address to open, normally <http://127.0.0.1:5000>. If port 5000 is already
in use the app moves to the next free port and prints that address instead.

Notes:

- The model is trained when the app starts, which takes a few seconds. Wait for the line
  `Model ready.` before opening the browser.
- Keep the terminal window open while using the app. Closing it, or pressing `CTRL+C`, stops the
  server, and the browser will then report `ERR_CONNECTION_REFUSED`.
- `StudentPerformanceFactors.csv` must stay in the same folder as `app.py`. The path is resolved
  relative to `app.py`, so the app can be started from any working directory.
- If `python` is not recognised, use the full path to your interpreter, for example
  `"%USERPROFILE%\anaconda3\python.exe" app.py` on Windows.

### Standalone version: `predictor.html`

`predictor.html` is the same interface as a single self-contained file. **Double-click it and it
opens in any browser.** It needs no Python, no Flask, no terminal, and no installation, so it can
be copied to another computer or sent to someone directly.

This works because Logistic Regression predicts with a plain weighted sum. `build_predictor.py`
trains the model, exports the imputation values, the scaler statistics, the one-hot category
order, and the fitted coefficients into the page, and the page applies them in the same order in
JavaScript.

To regenerate it after retraining the model:

```bash
python build_predictor.py     # writes predictor.html and predictor_check.json
node verify_predictor.js      # checks the page against scikit-learn (optional)
```

The verification runs all 1,322 held-out test students through the JavaScript in the page and
compares each result with the prediction scikit-learn produced for the same student. The current
build reports **0 mismatches** and a largest probability difference of **1.55e-15**, which is
ordinary floating-point rounding, so the page and the notebook give the same answers.

| | `app.py` (Flask) | `predictor.html` (standalone) |
|---|---|---|
| Needs Python and Flask | Yes | No |
| Needs a terminal | Yes | No |
| Can be sent to someone else | They must install and run it | Yes, it is one file |
| Runs the live scikit-learn model | Yes | No, it uses the exported coefficients |
| Must be rebuilt after retraining | No | Yes, run `build_predictor.py` |

### What the interface does

| Page | Contents |
|---|---|
| Input form | The 19 student factors, grouped into Academic, Personal, Family, and School. Numeric fields show the range accepted by the model; categorical fields are dropdowns. |
| Result | The predicted class, the probability of each of Low, Medium, and High, and the model's confidence. |
| Explanation | The contribution of each factor to the result, split into factors that supported the predicted class and factors that worked against it, measured against an average student in the training data. |
| What-if analysis | Realistic changes re-run through the model one at a time, showing which would most improve the predicted class. |

All input is validated on the server: every field is required, numeric values must fall inside the
range seen in training, and categorical values must be one of the categories seen in training.
Invalid submissions return to the form with the specific field highlighted.

## Avoiding data leakage

The following rules are important to the validity of the results:

- Do not include `Exam_Score` in `X`.
- Split the data before fitting preprocessing steps.
- Keep imputation, encoding, scaling, and feature selection inside a Pipeline.
- Fit GridSearchCV using only `X_train` and `y_train`.
- Use the held-out test set only for final evaluation.
- Pass an explicit class order to classification reports and confusion matrices.

## Assignment deliverables

The assignment consists of:

1. Documentation - 40%
2. Prototype development and source code - 60%

The final group submission should include the completed report, the three model notebooks, the dataset, required plagiarism forms, and the AI Disclosure Statement. The exact upload format must follow the instructions in Google Classroom.

The assignment specification states a submission deadline of **28 August 2026 before 12:00 PM**, followed by prototype demonstration and Q&A sessions in Weeks 12-14.

## Documentation checklist

The report should cover:

- Background, problem statement, objectives, significance, and research gap
- Related studies and critical comparison of previous work
- Dataset description and source
- System flow and preprocessing methodology
- Explanation and justification of each algorithm
- Evaluation metrics
- Actual results and confusion matrices
- Fair comparison of the three models
- Achievements, limitations, and future improvements
- APA-formatted references
- Dataset and tool acknowledgements
- AI Disclosure Statement and verification steps

## Limitations

- The target classes are project-defined from `Exam_Score`; they are not presented as a universal educational grading standard.
- The dataset comes from a single public source and may not represent every institution or student population.
- Medium is the largest class, so accuracy alone is insufficient for evaluating class-balanced performance.
- Results apply to the current dataset, target definition, preprocessing pipelines, hyperparameter grids, and held-out split.
- External validation on data from other educational settings has not been performed.

## Academic integrity and AI disclosure

This repository is coursework material. Any AI assistance used for brainstorming, code support, debugging, or writing must be disclosed according to the assignment requirements. Team members remain responsible for verifying the code, understanding every step, reporting only actual execution results, and answering questions during the demonstration.
