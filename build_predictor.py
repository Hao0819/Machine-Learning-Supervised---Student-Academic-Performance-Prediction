"""
Build a standalone, single-file version of the prediction interface.
======================================================================

BMCS2203 Artificial Intelligence - Logistic Regression module

`app.py` needs Python and Flask running in the background. This script produces
`predictor.html`, a single file that runs entirely in the browser: no Python, no
server, no installation. Anyone can double-click it and use the predictor.

This is possible because Logistic Regression predicts with a plain weighted sum.
The script trains exactly the same pipeline used in `LogisticRegression.ipynb`,
then exports the numbers the browser needs to reproduce it:

  * the median and most-frequent values used for imputation,
  * the mean and scale used by StandardScaler,
  * the category order used by OneHotEncoder,
  * the fitted coefficients and intercepts.

The JavaScript in the generated page applies those numbers in the same order as
scikit-learn, so the prediction, the probabilities, and the explanation match the
notebook exactly. `verify_predictor.js` checks this on every test record:

    node verify_predictor.js

Run with:

    python build_predictor.py

Re-run it whenever the model is retrained, so the exported numbers stay current.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The model definition is shared with the Flask app so the two cannot drift apart.
from app import (
    BEST_PARAMETERS,
    CLASS_RANK,
    DISPLAY_ORDER,
    ENGINEERED_FEATURES,
    FEATURE_LABELS,
    FIELD_GROUPS,
    SUPPORT_MAP,
    ACTIONABLE_NUMERIC,
    ACTIONABLE_ORDINAL,
    COMBINED_PLAN,
    PROJECT_FOLDER,
    DATA_FILE,
    add_engineered_features,
)

OUTPUT_FILE = PROJECT_FOLDER / "predictor.html"
STYLE_FILE = PROJECT_FOLDER / "static" / "style.css"


def train_and_export():
    """Train the selected model and collect every number the browser needs."""
    data = pd.read_csv(DATA_FILE)
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

    fitted_preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["logreg"]

    numeric_pipeline = fitted_preprocessor.named_transformers_["numeric"]
    categorical_pipeline = fitted_preprocessor.named_transformers_["categorical"]

    scaler = numeric_pipeline.named_steps["scaler"]
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    onehot = categorical_pipeline.named_steps["onehot"]

    transformed_names = fitted_preprocessor.get_feature_names_out().tolist()
    transformed_train = fitted_preprocessor.transform(X_train)

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

    input_features = [c for c in X_train.columns if c not in ENGINEERED_FEATURES]
    input_numeric = [c for c in input_features if c in numeric_features]
    input_categorical = [c for c in input_features if c in categorical_features]

    export = {
        "classes": classifier.classes_.tolist(),
        "displayOrder": DISPLAY_ORDER,
        "classRank": CLASS_RANK,
        "numericFeatures": numeric_features,
        "categoricalFeatures": categorical_features,
        "inputNumeric": input_numeric,
        "inputCategorical": input_categorical,
        "inputFeatures": input_features,
        "engineeredFeatures": ENGINEERED_FEATURES,
        "supportMap": SUPPORT_MAP,
        "fieldGroups": [[name, fields] for name, fields in FIELD_GROUPS],
        "labels": FEATURE_LABELS,

        # Preprocessing constants, in the exact order scikit-learn applies them.
        "numericMedians": dict(zip(numeric_features,
                                   numeric_imputer.statistics_.tolist())),
        "categoricalModes": dict(zip(categorical_features,
                                     categorical_imputer.statistics_.tolist())),
        "scalerMean": dict(zip(numeric_features, scaler.mean_.tolist())),
        "scalerScale": dict(zip(numeric_features, scaler.scale_.tolist())),
        "categories": {
            feature: values.tolist()
            for feature, values in zip(categorical_features, onehot.categories_)
        },

        # The fitted model itself.
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),

        # Needed for the explanation.
        "transformedNames": transformed_names,
        "columnOrigin": column_origin,
        "transformedMeans": transformed_train.mean(axis=0).tolist(),

        # Validation rules and form defaults, taken from the training data only.
        "numericRules": {
            column: {
                "min": float(X_train[column].min()),
                "max": float(X_train[column].max()),
                "mean": float(X_train[column].mean()),
                "default": float(X_train[column].median()),
                "step": 1 if pd.api.types.is_integer_dtype(X_train[column]) else 0.5,
            }
            for column in input_numeric
        },
        "categoryRules": {
            column: {
                "values": sorted(X_train[column].dropna().unique().tolist()),
                "default": X_train[column].mode()[0],
                "share": {
                    value: float((X_train[column] == value).mean())
                    for value in sorted(X_train[column].dropna().unique().tolist())
                },
            }
            for column in input_categorical
        },

        # What-if settings, shared with the Flask app.
        "actionableNumeric": ACTIONABLE_NUMERIC,
        "actionableOrdinal": ACTIONABLE_ORDINAL,
        "combinedPlan": [[f, m, a] for f, m, a in COMBINED_PLAN],

        "metrics": {
            "accuracy": accuracy_score(y_test, y_pred) * 100,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred) * 100,
            "macro_f1": f1_score(y_test, y_pred, average="macro") * 100,
            "training_records": int(len(X_train)),
            "test_records": int(len(X_test)),
        },
        "bestParameters": {key: str(value) for key, value in BEST_PARAMETERS.items()},
    }

    return export, model, X_test, y_test


PAGE_SCRIPT = r"""
// ===================================================================
//  Logistic Regression, reproduced in the browser
//  Every number below comes from the trained scikit-learn pipeline;
//  this code only applies them in the same order.
// ===================================================================

const MODEL = JSON.parse(document.getElementById("model-data").textContent);

// --- Step 1: derive the two engineered features from the raw answers ----------
function addEngineeredFeatures(record) {
    const full = Object.assign({}, record);
    full["Support_Index"] =
        MODEL.supportMap[record["Parental_Involvement"]] +
        MODEL.supportMap[record["Access_to_Resources"]] +
        MODEL.supportMap[record["Teacher_Quality"]];
    full["Study_Consistency"] = record["Hours_Studied"] * record["Attendance"] / 100;
    return full;
}

// --- Step 2: impute, scale, and one-hot encode, exactly as ColumnTransformer --
function transform(record) {
    const full = addEngineeredFeatures(record);
    const vector = [];

    for (const feature of MODEL.numericFeatures) {
        let value = full[feature];
        if (value === undefined || value === null || Number.isNaN(value)) {
            value = MODEL.numericMedians[feature];       // SimpleImputer(median)
        }
        const mean = MODEL.scalerMean[feature];
        const scale = MODEL.scalerScale[feature];
        vector.push((value - mean) / scale);             // StandardScaler
    }

    for (const feature of MODEL.categoricalFeatures) {
        let value = full[feature];
        if (value === undefined || value === null || value === "") {
            value = MODEL.categoricalModes[feature];     // SimpleImputer(most_frequent)
        }
        for (const category of MODEL.categories[feature]) {
            vector.push(value === category ? 1 : 0);     // OneHotEncoder
        }
    }

    return vector;
}

// --- Step 3: score each class and turn the scores into probabilities ----------
function predict(record) {
    const vector = transform(record);

    const logits = MODEL.coef.map((weights, index) => {
        let total = MODEL.intercept[index];
        for (let j = 0; j < weights.length; j++) {
            total += weights[j] * vector[j];
        }
        return total;
    });

    // Softmax, written the same way scikit-learn does it: subtract the largest
    // value first so that Math.exp cannot overflow.
    const largest = Math.max(...logits);
    const exponentials = logits.map(value => Math.exp(value - largest));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    const probabilities = exponentials.map(value => value / total);

    let bestIndex = 0;
    for (let i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
    }

    const byClass = {};
    MODEL.classes.forEach((name, index) => { byClass[name] = probabilities[index]; });

    return { predicted: MODEL.classes[bestIndex], probabilities: byClass, vector: vector };
}

// --- Step 4: explain the result ----------------------------------------------
// The score that decides the winner is the predicted class's score minus the
// average of the others, so those are the weights used here. Each contribution is
// measured against the average student in the training data, which makes the
// numbers add up to a meaningful total.
function explain(record, predictedClass, vector) {
    const predictedIndex = MODEL.classes.indexOf(predictedClass);
    const otherIndices = MODEL.classes.map((_, i) => i).filter(i => i !== predictedIndex);

    const grouped = {};
    for (let j = 0; j < vector.length; j++) {
        let otherAverage = 0;
        for (const index of otherIndices) otherAverage += MODEL.coef[index][j];
        otherAverage /= otherIndices.length;

        const weight = MODEL.coef[predictedIndex][j] - otherAverage;
        const centred = vector[j] - MODEL.transformedMeans[j];
        const origin = MODEL.columnOrigin[j];

        grouped[origin] = (grouped[origin] || 0) + weight * centred;
    }

    const full = addEngineeredFeatures(record);
    const rows = Object.keys(grouped).map(feature => ({
        feature: feature,
        label: MODEL.labels[feature] || feature.replace(/_/g, " "),
        value: describeValue(feature, full),
        comparison: compareWithAverage(feature, full),
        contribution: grouped[feature]
    }));

    rows.sort((a, b) => b.contribution - a.contribution);

    const largest = Math.max(...rows.map(row => Math.abs(row.contribution)), 1e-9);
    rows.forEach(row => { row.width = Math.min(100, Math.abs(row.contribution) / largest * 100); });

    return {
        supporting: rows.filter(row => row.contribution > 0.01).slice(0, 6),
        opposing: rows.filter(row => row.contribution < -0.01)
                      .sort((a, b) => a.contribution - b.contribution).slice(0, 6)
    };
}

function describeValue(feature, full) {
    const value = full[feature];
    if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : value.toFixed(1);
    }
    return String(value);
}

function compareWithAverage(feature, full) {
    if (MODEL.numericRules[feature]) {
        const average = MODEL.numericRules[feature].mean;
        const difference = full[feature] - average;
        if (Math.abs(difference) < 0.05 * Math.max(Math.abs(average), 1)) {
            return "about average (" + average.toFixed(1) + ")";
        }
        return Math.abs(difference).toFixed(1) + (difference > 0 ? " above" : " below") +
               " the average of " + average.toFixed(1);
    }
    if (MODEL.categoryRules[feature]) {
        const share = MODEL.categoryRules[feature].share[full[feature]];
        if (share === undefined) return "";
        return Math.round(share * 100) + "% of students are in this group";
    }
    if (feature === "Support_Index") {
        return "combines parental involvement, resources, and teacher quality";
    }
    if (feature === "Study_Consistency") {
        return "combines hours studied with attendance";
    }
    return "";
}

// --- Step 5: what-if analysis -------------------------------------------------
function runWhatIf(record, predictedClass) {
    if (predictedClass === "High") return [];

    const targetClass = predictedClass === "Medium" ? "High" : "Medium";
    const baseProbability = predict(record).probabilities[targetClass];

    const candidates = [];

    for (const feature of Object.keys(MODEL.actionableNumeric)) {
        const rules = MODEL.numericRules[feature];
        for (const increment of MODEL.actionableNumeric[feature]) {
            const newValue = Math.min(record[feature] + increment, rules.max);
            if (newValue <= record[feature]) continue;
            candidates.push({
                key: feature,
                description: "Raise " + MODEL.labels[feature].toLowerCase() +
                             " from " + record[feature] + " to " + newValue,
                record: Object.assign({}, record, { [feature]: newValue })
            });
        }
    }

    for (const feature of Object.keys(MODEL.actionableOrdinal)) {
        const ladder = MODEL.actionableOrdinal[feature];
        const position = ladder.indexOf(record[feature]);
        if (position < 0) continue;
        for (const higher of ladder.slice(position + 1)) {
            candidates.push({
                key: feature,
                description: "Improve " + MODEL.labels[feature].toLowerCase() +
                             " from " + record[feature] + " to " + higher,
                record: Object.assign({}, record, { [feature]: higher })
            });
        }
    }

    const results = [];
    for (const candidate of candidates) {
        const outcome = predict(candidate.record);
        const gain = outcome.probabilities[targetClass] - baseProbability;
        if (gain <= 0.005) continue;
        results.push({
            key: candidate.key,
            description: candidate.description,
            gain: gain * 100,
            targetClass: targetClass,
            changesClass: MODEL.classRank[outcome.predicted] > MODEL.classRank[predictedClass],
            newClass: outcome.predicted
        });
    }

    results.sort((a, b) => b.gain - a.gain);

    const seen = new Set();
    const trimmed = [];
    for (const result of results) {
        if (seen.has(result.key)) continue;
        seen.add(result.key);
        trimmed.push(result);
        if (trimmed.length === 4) break;
    }

    if (trimmed.length > 0) return trimmed;

    // Nothing worked on its own, so apply the improvements together.
    let current = Object.assign({}, record);
    const steps = [];
    let last = null;

    for (const [feature, mode, amount] of MODEL.combinedPlan) {
        const oldValue = current[feature];
        let newValue;

        if (mode === "add") {
            newValue = Math.min(oldValue + amount, MODEL.numericRules[feature].max);
        } else {
            const ladder = MODEL.actionableOrdinal[feature] || [];
            if (ladder.indexOf(oldValue) < 0 ||
                ladder.indexOf(oldValue) >= ladder.indexOf(amount)) continue;
            newValue = amount;
        }
        if (newValue === oldValue) continue;

        steps.push(MODEL.labels[feature] + ": " + oldValue + " to " + newValue);
        current = Object.assign({}, current, { [feature]: newValue });

        const outcome = predict(current);
        const gain = outcome.probabilities[targetClass] - baseProbability;
        const flipped = MODEL.classRank[outcome.predicted] > MODEL.classRank[predictedClass];
        last = { outcome: outcome, gain: gain, flipped: flipped, steps: steps.slice() };

        if (flipped || gain > 0.01) {
            return [{
                description: "No single change is enough on its own. These changes made " +
                             "together would move the prediction:",
                steps: last.steps,
                gain: gain * 100,
                targetClass: targetClass,
                changesClass: flipped,
                newClass: outcome.predicted
            }];
        }
    }

    if (!last) return [];

    return [{
        description: "Even applying every realistic improvement together does not change " +
                     "the predicted class for this student:",
        steps: last.steps,
        gain: last.gain * 100,
        targetClass: targetClass,
        changesClass: false,
        newClass: last.outcome.predicted
    }];
}

// --- Step 6: validation, the same rules the Flask version applies -------------
function validate(form) {
    const record = {};
    const errors = {};

    for (const feature of MODEL.inputFeatures) {
        const field = form.elements[feature];
        const raw = field ? String(field.value).trim() : "";

        if (raw === "") {
            errors[feature] = "This field is required.";
            continue;
        }

        if (MODEL.numericRules[feature]) {
            const rules = MODEL.numericRules[feature];
            const number = Number(raw);
            if (!Number.isFinite(number)) {
                errors[feature] = "Please enter a number.";
                continue;
            }
            if (number < rules.min || number > rules.max) {
                errors[feature] = "Please enter a value between " + rules.min + " and " +
                    rules.max + ". This is the range covered by the training data, so the " +
                    "model cannot make a reliable prediction outside it.";
                continue;
            }
            record[feature] = number;
        } else {
            const allowed = MODEL.categoryRules[feature].values;
            if (!allowed.includes(raw)) {
                errors[feature] = "Please choose one of: " + allowed.join(", ") + ".";
                continue;
            }
            record[feature] = raw;
        }
    }

    return { record: record, errors: errors };
}

// --- Page rendering ----------------------------------------------------------
function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, character => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[character]));
}

function buildForm() {
    const container = document.getElementById("form-fields");
    let html = "";

    for (const [groupName, groupFields] of MODEL.fieldGroups) {
        html += '<fieldset><legend>' + escapeHtml(groupName) + '</legend><div class="grid">';

        for (const feature of groupFields) {
            const label = MODEL.labels[feature] || feature;
            html += '<div class="field" data-field="' + feature + '">';
            html += '<label for="' + feature + '">' + escapeHtml(label) + '</label>';

            if (MODEL.numericRules[feature]) {
                const rules = MODEL.numericRules[feature];
                html += '<input type="number" id="' + feature + '" name="' + feature +
                        '" value="' + rules.default + '" min="' + rules.min +
                        '" max="' + rules.max + '" step="' + rules.step + '">';
                html += '<span class="hint">Allowed range ' + rules.min + ' to ' + rules.max +
                        ' &middot; average ' + rules.mean.toFixed(1) + '</span>';
            } else {
                const rules = MODEL.categoryRules[feature];
                html += '<select id="' + feature + '" name="' + feature + '">';
                for (const option of rules.values) {
                    html += '<option value="' + escapeHtml(option) + '"' +
                            (option === rules.default ? " selected" : "") + '>' +
                            escapeHtml(option) + '</option>';
                }
                html += '</select>';
                html += '<span class="hint">' + rules.values.length + ' options</span>';
            }

            html += '<span class="error"></span></div>';
        }
        html += '</div></fieldset>';
    }

    container.innerHTML = html;
}

function showErrors(errors) {
    document.querySelectorAll(".field").forEach(field => {
        const feature = field.getAttribute("data-field");
        const message = errors[feature];
        field.classList.toggle("has-error", Boolean(message));
        field.querySelector(".error").textContent = message || "";
    });

    const notice = document.getElementById("notice");
    const count = Object.keys(errors).length;
    if (count > 0) {
        notice.textContent = "Please correct the highlighted fields and submit again.";
        notice.style.display = "block";
        const first = document.querySelector(".field.has-error");
        if (first) first.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
        notice.style.display = "none";
    }
}

function factorTable(rows, kind) {
    if (rows.length === 0) {
        return '<p class="lead">' + (kind === "positive"
            ? "No single factor stood out; the result came from many small effects."
            : "Nothing worked strongly against this result.") + '</p>';
    }

    let html = '<table class="factors"><tr><th>Factor</th><th>Strength</th><th></th></tr>';
    for (const row of rows) {
        html += '<tr><td><div class="factor-name">' + escapeHtml(row.label) + '</div>' +
                '<div class="factor-value">' + escapeHtml(row.value) +
                (row.comparison ? " &mdash; " + escapeHtml(row.comparison) : "") +
                '</div></td>' +
                '<td class="bar-cell"><div class="bar ' + kind +
                '" style="width: ' + row.width.toFixed(4) + '%"></div></td>' +
                '<td class="amount">' + (row.contribution > 0 ? "+" : "") +
                row.contribution.toFixed(2) + '</td></tr>';
    }
    return html + '</table>';
}

function renderResult(record, outcome, explanation, suggestions) {
    const ordered = MODEL.displayOrder.map(label => ({
        label: label,
        value: outcome.probabilities[label] * 100
    }));
    const sorted = ordered.slice().sort((a, b) => b.value - a.value);
    const confidence = sorted[0].value;
    const runnerUp = sorted[1];
    const predicted = outcome.predicted;

    const meaning = predicted === "Low"
        ? "A Low prediction means the model expects an exam score below 65, which is the group an early-intervention system would flag for extra support."
        : predicted === "Medium"
            ? "A Medium prediction means the model expects an exam score between 65 and 69."
            : "A High prediction means the model expects an exam score of 70 or above.";

    let html = '<div class="card"><h2>Prediction result</h2>';
    html += '<div class="verdict ' + predicted + '"><div>' +
            '<div class="class-name">' + predicted + '</div>' +
            '<div style="font-size:13px;color:var(--muted)">predicted performance</div></div>' +
            '<div class="detail">The model is <strong>' + confidence.toFixed(1) +
            '%</strong> confident in this class. The next most likely class is <strong>' +
            runnerUp.label + '</strong> at ' + runnerUp.value.toFixed(1) + '%. ' +
            meaning + '</div></div>';

    html += '<div class="probability">';
    for (const item of ordered) {
        html += '<div class="row"><span class="name">' + item.label + '</span>' +
                '<span class="track"><span class="fill ' + item.label +
                '" style="width: ' + item.value.toFixed(4) + '%"></span></span>' +
                '<span class="value">' + item.value.toFixed(1) + '%</span></div>';
    }
    html += '</div></div>';

    const full = addEngineeredFeatures(record);
    html += '<div class="card"><h2>Why did you get this result?</h2>' +
            '<p class="lead">Logistic Regression decides by adding up one weighted ' +
            'contribution per factor, so the result can be broken down exactly. Each number ' +
            'below shows how much that factor pushed the prediction towards <strong>' +
            predicted + '</strong> compared with an average student in the training data. ' +
            'A longer green bar means the factor helped more; a red bar means the factor ' +
            'pushed away from this class.</p>';

    html += '<div class="split"><div><h3>Factors that led to ' + predicted +
            '<span class="tag positive">pushed towards</span></h3>' +
            factorTable(explanation.supporting, "positive") + '</div>';
    html += '<div><h3>Factors that worked against it' +
            '<span class="tag negative">pushed away</span></h3>' +
            factorTable(explanation.opposing, "negative") + '</div></div>';

    html += '<div class="disclaimer"><strong>Calculated from your answers:</strong> ' +
            'Support Index = ' + full["Support_Index"] + ' out of 6 &middot; ' +
            'Study Consistency = ' + full["Study_Consistency"].toFixed(1) + '. ' +
            'These two engineered features are not entered directly &mdash; they are derived ' +
            'from the answers you gave, which is why they can appear in the table above.' +
            '</div></div>';

    if (suggestions.length > 0) {
        html += '<div class="card"><h2>What would change the result?</h2>' +
                '<p class="lead">Each change below was re-run through the model one at a ' +
                'time, keeping everything else the same, to see how much it would raise the ' +
                'chance of reaching <strong>' + suggestions[0].targetClass + '</strong>.</p>';

        for (const item of suggestions) {
            html += '<div class="suggestion' + (item.changesClass ? " flip" : "") + '">' +
                    escapeHtml(item.description);
            if (item.steps) {
                html += '<ul class="steps">';
                for (const step of item.steps) html += '<li>' + escapeHtml(step) + '</li>';
                html += '</ul>';
            }
            html += '<span class="gain">';
            if (item.gain > 0) {
                html += 'Chance of ' + item.targetClass + ' increases by ' +
                        item.gain.toFixed(1) + ' percentage points' +
                        (item.changesClass ? ', which is enough to change the prediction to <strong>' +
                        item.newClass + '</strong>' : '') + '.';
            } else {
                html += 'The predicted class stays at <strong>' + item.newClass +
                        '</strong>, which means this student\'s result is driven by factors ' +
                        'the model sees as difficult to move.';
            }
            html += '</span></div>';
        }

        html += '<div class="disclaimer">These are model estimates based on patterns in the ' +
                'training data, not guaranteed outcomes, and they show correlation rather ' +
                'than proven cause and effect.</div></div>';
    }

    html += '<div class="card"><h2>Your answers</h2><details>' +
            '<summary>Show the 19 values used for this prediction</summary>' +
            '<table class="answers">';
    for (const feature of MODEL.inputFeatures) {
        html += '<tr><td>' + escapeHtml(MODEL.labels[feature] || feature) + '</td><td>' +
                escapeHtml(record[feature]) + '</td></tr>';
    }
    html += '</table></details>' +
            '<div class="actions"><button class="button" type="button" id="again">' +
            'Make another prediction</button></div>' +
            '<div class="disclaimer">This is a coursework prototype trained on a public ' +
            'Kaggle dataset. It should not be used to make real decisions about a student.' +
            '</div></div>';

    const results = document.getElementById("results");
    results.innerHTML = html;
    results.style.display = "block";

    document.getElementById("again").addEventListener("click", () => {
        results.style.display = "none";
        window.scrollTo({ top: 0, behavior: "smooth" });
    });

    results.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- Wiring ------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    buildForm();

    document.getElementById("predict-form").addEventListener("submit", event => {
        event.preventDefault();

        const { record, errors } = validate(event.target);
        showErrors(errors);
        if (Object.keys(errors).length > 0) {
            document.getElementById("results").style.display = "none";
            return;
        }

        const outcome = predict(record);
        const explanation = explain(record, outcome.predicted, outcome.vector);
        const suggestions = runWhatIf(record, outcome.predicted);

        renderResult(record, outcome, explanation, suggestions);
    });

    document.getElementById("reset").addEventListener("click", () => {
        buildForm();
        document.getElementById("results").style.display = "none";
        document.getElementById("notice").style.display = "none";
    });
});
"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Academic Performance Prediction</title>
<style>
__STYLE__
</style>
</head>
<body>

<header class="masthead">
    <div class="page">
        <h1>Student Academic Performance Prediction</h1>
        <p>BMCS2203 Artificial Intelligence &middot; Multiclass Logistic Regression</p>
        <div class="badges">
            <span class="badge">Best model: Logistic Regression (C=__C__, solver=__SOLVER__)</span>
            <span class="badge">Test accuracy __ACCURACY__%</span>
            <span class="badge">Macro F1 __MACRO_F1__%</span>
            <span class="badge">__TEST_RECORDS__ held-out students</span>
        </div>
    </div>
</header>

<main class="page">

<div class="notice" id="notice" style="display:none"></div>

<form class="card" id="predict-form" novalidate>
    <h2>Enter the student details</h2>
    <p class="lead">
        Fill in the 19 factors below and the model will predict whether this student's
        academic performance is Low, Medium, or High, and explain which factors led to
        that result. Every field is required, and numeric values must stay inside the
        range covered by the training data.
    </p>

    <div id="form-fields"></div>

    <div class="actions">
        <button class="button" type="submit">Predict performance</button>
        <button class="button secondary" type="button" id="reset">Reset the form</button>
    </div>
</form>

<div id="results" style="display:none"></div>

<div class="card">
    <h2>How the prediction is made</h2>
    <p class="lead" style="margin-bottom:0">
        Your answers are processed exactly as they were during training: missing values are
        imputed, categorical answers are one-hot encoded, and numeric values are standardized.
        Two extra features are then calculated from your answers &mdash;
        <strong>Support Index</strong> (parental involvement + access to resources + teacher
        quality) and <strong>Study Consistency</strong> (hours studied &times; attendance).
        Logistic Regression scores each of the three classes and the highest score wins, so
        the result can be traced back to the exact contribution of each factor.
    </p>
</div>

<p class="footer">
    Trained on __TRAINING_RECORDS__ students from StudentPerformanceFactors.csv.
    This page contains the coefficients of the model selected in LogisticRegression.ipynb and
    calculates the prediction in the browser, so it needs no server and no installation.
</p>

</main>

<script id="model-data" type="application/json">__MODEL_JSON__</script>
<script>
__SCRIPT__
</script>
</body>
</html>
"""


def build_page(export):
    style = STYLE_FILE.read_text(encoding="utf-8")

    # The JSON sits inside a <script> tag, so the only sequence that could break out
    # of it must be neutralised.
    model_json = json.dumps(
        export, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")

    page = PAGE_TEMPLATE
    page = page.replace("__STYLE__", style)
    page = page.replace("__SCRIPT__", PAGE_SCRIPT)
    page = page.replace("__MODEL_JSON__", model_json)
    page = page.replace("__C__", export["bestParameters"]["C"])
    page = page.replace("__SOLVER__", export["bestParameters"]["solver"])
    page = page.replace("__ACCURACY__", f"{export['metrics']['accuracy']:.2f}")
    page = page.replace("__MACRO_F1__", f"{export['metrics']['macro_f1']:.2f}")
    page = page.replace("__TEST_RECORDS__", str(export["metrics"]["test_records"]))
    page = page.replace("__TRAINING_RECORDS__", str(export["metrics"]["training_records"]))
    return page


def main():
    print("Training the model and exporting its coefficients...")
    export, model, X_test, y_test = train_and_export()

    print(f"  Test accuracy      : {export['metrics']['accuracy']:.2f}%")
    print(f"  Transformed columns: {len(export['transformedNames'])}")
    print(f"  Classes            : {export['classes']}")

    page = build_page(export)
    OUTPUT_FILE.write_text(page, encoding="utf-8")

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nWritten {OUTPUT_FILE.name} ({size_kb:.0f} KB)")
    print("Double-click that file to use the predictor. No Python or server required.")

    # Keep a copy of the test data so verify_predictor.py can check the page against
    # scikit-learn without retraining.
    check = X_test.drop(columns=export["engineeredFeatures"]).copy()

    # JSON has no NaN, so missing values become null. The JavaScript treats null the
    # same way SimpleImputer does, which is exactly the behaviour being checked.
    check_records = []
    for _, row in check.iterrows():
        record = {}
        for column, value in row.items():
            if pd.isna(value):
                record[column] = None
            elif isinstance(value, np.integer):
                record[column] = int(value)
            elif isinstance(value, np.floating):
                record[column] = float(value)
            else:
                record[column] = value
        check_records.append(record)
    expected = model.predict(X_test).tolist()
    probabilities = model.predict_proba(X_test).tolist()

    verification = {
        "records": check_records,
        "expected": expected,
        "probabilities": probabilities,
        "classes": export["classes"],
    }
    verification_file = PROJECT_FOLDER / "predictor_check.json"
    verification_file.write_text(json.dumps(verification, allow_nan=False), encoding="utf-8")
    print(f"Written {verification_file.name} for verification "
          f"({len(check_records)} test students)")


if __name__ == "__main__":
    main()
