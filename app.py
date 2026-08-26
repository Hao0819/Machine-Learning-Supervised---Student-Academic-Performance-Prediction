"""
Student Academic Performance Prediction - Web Interface
=======================================================

BMCS2203 Artificial Intelligence
Algorithm: Multiclass Logistic Regression
Module owner: Tan Keng Ting (26WMR12927)

A local web application that lets a student enter their own study, family, and school
factors and receive a predicted performance class (Low / Medium / High) together with an
explanation of *why* the model produced that result.

The model is exactly the one selected in `LogisticRegression.ipynb`:
the same preprocessing pipeline, the same engineered features, the same train/test split,
and the hyperparameters chosen there by GridSearchCV (C=100, solver='lbfgs').

Run with:

    python app.py

then open http://127.0.0.1:5000 in a browser.
"""

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------

DATA_FILE = "StudentPerformanceFactors.csv"

# Best hyperparameters selected by GridSearchCV in LogisticRegression.ipynb (Section 5).
BEST_PARAMETERS = {"C": 100, "solver": "lbfgs", "class_weight": None}

DISPLAY_ORDER = ["Low", "Medium", "High"]
CLASS_RANK = {"Low": 0, "Medium": 1, "High": 2}

SUPPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}
ENGINEERED_FEATURES = ["Support_Index", "Study_Consistency"]

# Readable labels shown in the interface instead of raw column names.
FEATURE_LABELS = {
    "Hours_Studied": "Hours studied per week",
    "Attendance": "Class attendance (%)",
    "Parental_Involvement": "Parental involvement",
    "Access_to_Resources": "Access to learning resources",
    "Extracurricular_Activities": "Extracurricular activities",
    "Sleep_Hours": "Sleep hours per night",
    "Previous_Scores": "Previous exam scores",
    "Motivation_Level": "Motivation level",
    "Internet_Access": "Internet access",
    "Tutoring_Sessions": "Tutoring sessions per month",
    "Family_Income": "Family income",
    "Teacher_Quality": "Teacher quality",
    "School_Type": "School type",
    "Peer_Influence": "Peer influence",
    "Physical_Activity": "Physical activity (hours/week)",
    "Learning_Disabilities": "Learning disabilities",
    "Parental_Education_Level": "Parental education level",
    "Distance_from_Home": "Distance from home to school",
    "Gender": "Gender",
    "Support_Index": "Overall support level (parents + resources + teacher)",
    "Study_Consistency": "Study consistency (hours studied x attendance)",
}

# How the 19 input fields are grouped on the form, matching the categories used in the
# project README so the interface reads in a logical order.
FIELD_GROUPS = [
    ("Academic", [
        "Hours_Studied", "Attendance", "Previous_Scores",
        "Tutoring_Sessions", "Extracurricular_Activities",
    ]),
    ("Personal", [
        "Sleep_Hours", "Motivation_Level", "Physical_Activity",
        "Learning_Disabilities", "Peer_Influence", "Gender",
    ]),
    ("Family", [
        "Parental_Involvement", "Family_Income", "Parental_Education_Level",
    ]),
    ("School and resources", [
        "Access_to_Resources", "Teacher_Quality", "School_Type",
        "Internet_Access", "Distance_from_Home",
    ]),
]

# Fields a student can realistically change, used for the what-if analysis.
# Each entry lists the improved values to try, from the current value upwards.
ACTIONABLE_NUMERIC = {
    "Hours_Studied": [2, 5, 8, 12],      # additional hours per week to try
    "Attendance": [5, 10, 15, 25],       # additional percentage points to try
    "Tutoring_Sessions": [1, 2, 3, 4],   # additional sessions per month to try
    "Sleep_Hours": [1, 2],               # additional hours per night to try
}

# Used when no single change is enough on its own. The improvements are applied one after
# another, in this order, until the predicted class changes.
COMBINED_PLAN = [
    ("Attendance", "add", 20),
    ("Hours_Studied", "add", 12),
    ("Tutoring_Sessions", "add", 3),
    ("Motivation_Level", "set", "High"),
    ("Access_to_Resources", "set", "High"),
    ("Parental_Involvement", "set", "High"),
]

ACTIONABLE_ORDINAL = {
    "Motivation_Level": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
}


# ----------------------------------------------------------------------------------
# Model preparation
# ----------------------------------------------------------------------------------

def add_engineered_features(frame):
    """Add the two features engineered in Section 2 of the notebook."""
    frame = frame.copy()
    frame["Support_Index"] = (
        frame["Parental_Involvement"].map(SUPPORT_MAP)
        + frame["Access_to_Resources"].map(SUPPORT_MAP)
        + frame["Teacher_Quality"].map(SUPPORT_MAP)
    )
    frame["Study_Consistency"] = frame["Hours_Studied"] * frame["Attendance"] / 100
    return frame


def build_model():
    """Load the dataset, train the selected Logistic Regression model, and collect
    everything the web pages need to validate input and explain a prediction."""
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{DATA_FILE}' was not found. Keep the CSV file in the same folder as app.py."
        )

    if data.empty:
        raise ValueError(f"'{DATA_FILE}' contains no records.")

    if "Exam_Score" not in data.columns:
        raise KeyError("The dataset must contain an 'Exam_Score' column.")

    # Target: same three classes and same cut points as the notebook.
    data["Performance"] = pd.cut(
        data["Exam_Score"],
        bins=[-np.inf, 64, 69, np.inf],
        labels=DISPLAY_ORDER,
        ordered=True,
    )

    data = add_engineered_features(data)

    features = data.drop(columns=["Exam_Score", "Performance"])
    target = data["Performance"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("logreg", LogisticRegression(max_iter=2000, random_state=42, **BEST_PARAMETERS)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred) * 100,
        "macro_f1": f1_score(y_test, y_pred, average="macro") * 100,
        "training_records": len(X_train),
        "test_records": len(X_test),
    }

    # The 19 original features a student fills in; the other two are derived.
    input_features = [c for c in X_train.columns if c not in ENGINEERED_FEATURES]

    numeric_inputs = [c for c in input_features if c in numeric_features]
    categorical_inputs = [c for c in input_features if c in categorical_features]

    # Validation rules and form defaults, all taken from the training data only.
    numeric_rules = {}
    for column in numeric_inputs:
        series = X_train[column]
        numeric_rules[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "default": float(series.median()),
            "step": 1 if pd.api.types.is_integer_dtype(series) else 0.5,
        }

    category_rules = {}
    for column in categorical_inputs:
        values = sorted(X_train[column].dropna().unique().tolist())
        category_rules[column] = {
            "values": values,
            "default": X_train[column].mode()[0],
            "share": {
                value: float((X_train[column] == value).mean())
                for value in values
            },
        }

    # Column means of the transformed training data. Contributions are measured
    # relative to these, so an explanation always compares the student against the
    # average student in the training data.
    transformed_train = model.named_steps["preprocessor"].transform(X_train)
    transformed_means = transformed_train.mean(axis=0)

    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()

    # Map every transformed column back to the original feature it came from.
    column_origin = []
    for name in transformed_names:
        if name.startswith("numeric__"):
            column_origin.append(name[len("numeric__"):])
        else:
            stripped = name[len("categorical__"):]
            origin = stripped
            for candidate in categorical_features:
                if stripped.startswith(candidate + "_"):
                    origin = candidate
                    break
            column_origin.append(origin)

    return {
        "model": model,
        "columns": X_train.columns.tolist(),
        "input_features": input_features,
        "numeric_rules": numeric_rules,
        "category_rules": category_rules,
        "transformed_means": transformed_means,
        "column_origin": column_origin,
        "metrics": metrics,
    }


# ----------------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------------

def validate_form(form, bundle):
    """Check one submitted form. Returns (cleaned_record, errors)."""
    numeric_rules = bundle["numeric_rules"]
    category_rules = bundle["category_rules"]

    record = {}
    errors = {}

    for column in bundle["input_features"]:
        raw_value = form.get(column, "")
        raw_value = raw_value.strip() if isinstance(raw_value, str) else raw_value

        if raw_value == "" or raw_value is None:
            errors[column] = "This field is required."
            continue

        if column in numeric_rules:
            rules = numeric_rules[column]
            try:
                number = float(raw_value)
            except ValueError:
                errors[column] = "Please enter a number."
                continue

            if np.isnan(number) or np.isinf(number):
                errors[column] = "Please enter a valid number."
                continue

            if number < rules["min"] or number > rules["max"]:
                errors[column] = (
                    f"Please enter a value between {rules['min']:g} and {rules['max']:g}. "
                    "This is the range covered by the training data, so the model cannot "
                    "make a reliable prediction outside it."
                )
                continue

            record[column] = number
        else:
            allowed = category_rules[column]["values"]
            if raw_value not in allowed:
                errors[column] = f"Please choose one of: {', '.join(allowed)}."
                continue
            record[column] = raw_value

    return record, errors


# ----------------------------------------------------------------------------------
# Prediction and explanation
# ----------------------------------------------------------------------------------

def build_student_frame(record, bundle):
    """Turn one validated record into the single-row DataFrame the model expects."""
    frame = pd.DataFrame([record])
    frame = add_engineered_features(frame)
    return frame[bundle["columns"]]


def explain_prediction(student_frame, record, predicted_class, bundle):
    """Explain why this student received this class.

    Logistic Regression scores each class with `weights . features + intercept`, so the
    prediction can be decomposed exactly into one contribution per feature. The weights
    used here are the predicted class's weights minus the average of the other classes',
    which is what actually decides the winner, and each contribution is measured relative
    to the average student in the training data.
    """
    model = bundle["model"]
    classifier = model.named_steps["logreg"]
    classes = list(classifier.classes_)

    transformed = model.named_steps["preprocessor"].transform(student_frame)[0]
    centred = transformed - bundle["transformed_means"]

    predicted_index = classes.index(predicted_class)
    other_indices = [i for i in range(len(classes)) if i != predicted_index]
    effective_weights = (
        classifier.coef_[predicted_index] - classifier.coef_[other_indices].mean(axis=0)
    )

    contributions = effective_weights * centred

    # Group the one-hot columns back into their original feature.
    grouped = {}
    for origin, value in zip(bundle["column_origin"], contributions):
        grouped[origin] = grouped.get(origin, 0.0) + float(value)

    rows = []
    for feature, contribution in grouped.items():
        rows.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature.replace("_", " ")),
            "value": describe_value(feature, record, student_frame),
            "comparison": compare_with_average(feature, record, bundle),
            "contribution": contribution,
        })

    rows.sort(key=lambda item: item["contribution"], reverse=True)

    largest = max((abs(row["contribution"]) for row in rows), default=1.0) or 1.0
    for row in rows:
        row["width"] = min(100.0, abs(row["contribution"]) / largest * 100.0)

    supporting = [row for row in rows if row["contribution"] > 0.01][:6]
    opposing = [row for row in rows if row["contribution"] < -0.01]
    opposing = sorted(opposing, key=lambda item: item["contribution"])[:6]

    return supporting, opposing


def describe_value(feature, record, student_frame):
    """The value to show next to a factor in the explanation table."""
    if feature in record:
        value = record[feature]
        return f"{value:g}" if isinstance(value, float) else str(value)

    # Engineered features are not entered by the student, so read them back.
    value = student_frame.iloc[0][feature]
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def compare_with_average(feature, record, bundle):
    """A short phrase comparing the student's value with the training data."""
    numeric_rules = bundle["numeric_rules"]
    category_rules = bundle["category_rules"]

    if feature in numeric_rules:
        average = numeric_rules[feature]["mean"]
        value = record[feature]
        difference = value - average
        if abs(difference) < 0.05 * max(abs(average), 1):
            return f"about average ({average:.1f})"
        direction = "above" if difference > 0 else "below"
        return f"{abs(difference):.1f} {direction} the average of {average:.1f}"

    if feature in category_rules:
        share = category_rules[feature]["share"].get(record.get(feature), None)
        if share is None:
            return ""
        return f"{share * 100:.0f}% of students are in this group"

    if feature == "Support_Index":
        return "combines parental involvement, resources, and teacher quality"

    if feature == "Study_Consistency":
        return "combines hours studied with attendance"

    return ""


def run_what_if(record, predicted_class, bundle):
    """Try realistic improvements one at a time and report which would help most.

    Each candidate change is re-run through the full pipeline, so the engineered
    features are recalculated correctly (for example, raising attendance also raises
    Study_Consistency).
    """
    if predicted_class == "High":
        return []

    model = bundle["model"]
    classes = list(model.named_steps["logreg"].classes_)
    target_class = "High" if predicted_class == "Medium" else "Medium"
    target_index = classes.index(target_class)

    base_frame = build_student_frame(record, bundle)
    base_probability = model.predict_proba(base_frame)[0][target_index]

    candidates = []

    for feature, increments in ACTIONABLE_NUMERIC.items():
        rules = bundle["numeric_rules"][feature]
        for increment in increments:
            new_value = min(record[feature] + increment, rules["max"])
            if new_value <= record[feature]:
                continue
            changed = dict(record)
            changed[feature] = new_value
            candidates.append({
                "feature": feature,
                "description": (
                    f"Raise {FEATURE_LABELS[feature].lower()} from "
                    f"{record[feature]:g} to {new_value:g}"
                ),
                "record": changed,
            })

    for feature, ladder in ACTIONABLE_ORDINAL.items():
        if feature not in record or record[feature] not in ladder:
            continue
        current_position = ladder.index(record[feature])
        for higher in ladder[current_position + 1:]:
            changed = dict(record)
            changed[feature] = higher
            candidates.append({
                "feature": feature,
                "description": (
                    f"Improve {FEATURE_LABELS[feature].lower()} from "
                    f"{record[feature]} to {higher}"
                ),
                "record": changed,
            })

    if not candidates:
        return []

    batch = pd.concat(
        [build_student_frame(item["record"], bundle) for item in candidates],
        ignore_index=True,
    )
    probabilities = model.predict_proba(batch)
    predictions = model.predict(batch)

    results = []
    for item, probability_row, prediction in zip(candidates, probabilities, predictions):
        gain = probability_row[target_index] - base_probability
        if gain <= 0.005:
            continue
        results.append({
            "description": item["description"],
            "gain": gain * 100,
            "target_class": target_class,
            "changes_class": CLASS_RANK[prediction] > CLASS_RANK[predicted_class],
            "new_class": prediction,
        })

    results.sort(key=lambda item: item["gain"], reverse=True)

    # Keep only the strongest suggestion per underlying change, at most four.
    seen = set()
    trimmed = []
    for result in results:
        key = result["description"].split(" from ")[0]
        if key in seen:
            continue
        seen.add(key)
        trimmed.append(result)
        if len(trimmed) == 4:
            break

    if trimmed:
        return trimmed

    # No single change was enough on its own, which happens when the model is very
    # confident. Apply the improvements together and report the smallest combination
    # that makes a difference.
    return run_combined_what_if(
        record, predicted_class, bundle, target_class, target_index, base_probability
    )


def run_combined_what_if(record, predicted_class, bundle, target_class,
                         target_index, base_probability):
    """Apply several improvements together until the prediction actually moves."""
    model = bundle["model"]

    current = dict(record)
    steps = []
    frames = []
    step_history = []

    for feature, mode, amount in COMBINED_PLAN:
        if feature not in current:
            continue

        old_value = current[feature]

        if mode == "add":
            rules = bundle["numeric_rules"][feature]
            new_value = min(old_value + amount, rules["max"])
        else:
            ladder = ACTIONABLE_ORDINAL.get(feature, [])
            if old_value not in ladder or ladder.index(old_value) >= ladder.index(amount):
                continue
            new_value = amount

        if new_value == old_value:
            continue

        old_text = f"{old_value:g}" if isinstance(old_value, float) else str(old_value)
        new_text = f"{new_value:g}" if isinstance(new_value, float) else str(new_value)
        steps.append(f"{FEATURE_LABELS[feature]}: {old_text} to {new_text}")

        current[feature] = new_value
        frames.append(build_student_frame(current, bundle))
        step_history.append(list(steps))

    if not frames:
        return []

    batch = pd.concat(frames, ignore_index=True)
    probabilities = model.predict_proba(batch)
    predictions = model.predict(batch)

    for probability_row, prediction, applied in zip(probabilities, predictions, step_history):
        gain = probability_row[target_index] - base_probability
        flipped = CLASS_RANK[prediction] > CLASS_RANK[predicted_class]
        if flipped or gain > 0.01:
            return [{
                "description": (
                    "No single change is enough on its own. These changes made together "
                    "would move the prediction:"
                ),
                "steps": applied,
                "gain": gain * 100,
                "target_class": target_class,
                "changes_class": flipped,
                "new_class": prediction,
            }]

    return [{
        "description": (
            "Even applying every realistic improvement together does not change the "
            "predicted class for this student:"
        ),
        "steps": step_history[-1],
        "gain": (probabilities[-1][target_index] - base_probability) * 100,
        "target_class": target_class,
        "changes_class": False,
        "new_class": predictions[-1],
    }]


# ----------------------------------------------------------------------------------
# Flask application
# ----------------------------------------------------------------------------------

app = Flask(__name__)
app.jinja_env.globals["best_parameters"] = BEST_PARAMETERS

print("Training the Logistic Regression model, please wait...")
BUNDLE = build_model()
print(
    f"Model ready. Test accuracy {BUNDLE['metrics']['accuracy']:.2f}% "
    f"on {BUNDLE['metrics']['test_records']} held-out students."
)


def form_defaults():
    """Starting values for the form: the median of each numeric feature and the most
    common value of each categorical feature."""
    values = {}
    for column, rules in BUNDLE["numeric_rules"].items():
        values[column] = f"{rules['default']:g}"
    for column, rules in BUNDLE["category_rules"].items():
        values[column] = rules["default"]
    return values


def render_form(values, errors=None, message=None):
    return render_template(
        "index.html",
        input_features=BUNDLE["input_features"],
        field_groups=FIELD_GROUPS,
        numeric_rules=BUNDLE["numeric_rules"],
        category_rules=BUNDLE["category_rules"],
        labels=FEATURE_LABELS,
        values=values,
        errors=errors or {},
        message=message,
        metrics=BUNDLE["metrics"],
    )


@app.route("/", methods=["GET"])
def index():
    return render_form(form_defaults())


@app.route("/predict", methods=["POST"])
def predict():
    submitted = {key: request.form.get(key, "") for key in BUNDLE["input_features"]}
    record, errors = validate_form(request.form, BUNDLE)

    if errors:
        return render_form(
            submitted,
            errors=errors,
            message="Please correct the highlighted fields and submit again.",
        ), 400

    try:
        student_frame = build_student_frame(record, BUNDLE)
        model = BUNDLE["model"]
        classes = list(model.named_steps["logreg"].classes_)

        predicted_class = model.predict(student_frame)[0]
        probability_row = model.predict_proba(student_frame)[0]
    except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
        return render_form(
            submitted,
            message=f"The prediction could not be completed: {error}",
        ), 500

    probabilities = [
        {
            "label": label,
            "value": float(probability_row[classes.index(label)]) * 100,
        }
        for label in DISPLAY_ORDER
    ]

    supporting, opposing = explain_prediction(
        student_frame, record, predicted_class, BUNDLE
    )
    suggestions = run_what_if(record, predicted_class, BUNDLE)

    confidence = max(item["value"] for item in probabilities)
    runner_up = sorted(probabilities, key=lambda item: item["value"], reverse=True)[1]

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        probabilities=probabilities,
        confidence=confidence,
        runner_up=runner_up,
        supporting=supporting,
        opposing=opposing,
        suggestions=suggestions,
        record=record,
        labels=FEATURE_LABELS,
        input_features=BUNDLE["input_features"],
        derived={
            "Support_Index": student_frame.iloc[0]["Support_Index"],
            "Study_Consistency": student_frame.iloc[0]["Study_Consistency"],
        },
        metrics=BUNDLE["metrics"],
    )


@app.errorhandler(404)
def not_found(error):  # noqa: ARG001
    return render_form(form_defaults(), message="That page does not exist."), 404


if __name__ == "__main__":
    print()
    print("Open http://127.0.0.1:5000 in your browser. Press CTRL+C to stop.")
    print()
    app.run(host="127.0.0.1", port=5000, debug=False)
