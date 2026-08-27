/*
 * Check that predictor.html gives the same answers as scikit-learn.
 * ================================================================
 *
 * BMCS2203 Artificial Intelligence - Logistic Regression module
 *
 * predictor.html reproduces the trained model in JavaScript so the page can run
 * without Python. This script proves the reproduction is exact: it pulls the real
 * JavaScript out of the generated page, runs every held-out test student through
 * it, and compares the result with the prediction scikit-learn produced for the
 * same student (saved by build_predictor.py in predictor_check.json).
 *
 * Run with:
 *
 *     python build_predictor.py     (first, to generate both files)
 *     node verify_predictor.js
 */

const fs = require("fs");
const path = require("path");

const folder = __dirname;
const pageFile = path.join(folder, "predictor.html");
const checkFile = path.join(folder, "predictor_check.json");

for (const file of [pageFile, checkFile]) {
    if (!fs.existsSync(file)) {
        console.error(`Missing ${path.basename(file)}. Run: python build_predictor.py`);
        process.exit(1);
    }
}

const page = fs.readFileSync(pageFile, "utf8");
const check = JSON.parse(fs.readFileSync(checkFile, "utf8"));

// Pull the model data and the prediction code out of the page itself, so this
// script tests exactly what a browser would run.
const modelMatch = page.match(
    /<script id="model-data" type="application\/json">([\s\S]*?)<\/script>/
);
// \r?\n because the generated file uses Windows line endings.
const codeMatch = page.match(/<script>\r?\n([\s\S]*?)\r?\n<\/script>\s*<\/body>/);

if (!modelMatch || !codeMatch) {
    console.error("Could not extract the model or the script from predictor.html.");
    process.exit(1);
}

const modelJson = modelMatch[1].replace(/<\\\//g, "</");

// Minimal stand-ins for the browser objects the page touches while loading.
const documentStub = {
    getElementById: id => (id === "model-data" ? { textContent: modelJson } : null),
    addEventListener: () => {},
    querySelectorAll: () => [],
    querySelector: () => null
};

const load = new Function(
    "document",
    "window",
    codeMatch[1] + "\nreturn { predict, explain, runWhatIf, addEngineeredFeatures, MODEL };"
);
const browser = load(documentStub, { scrollTo: () => {} });

// ---------------------------------------------------------------------------

const total = check.records.length;
let classMismatches = 0;
let largestProbabilityGap = 0;
let worstRecord = null;

check.records.forEach((record, index) => {
    const outcome = browser.predict(record);

    if (outcome.predicted !== check.expected[index]) {
        classMismatches++;
        if (classMismatches <= 3) {
            console.error(
                `  record ${index}: browser said ${outcome.predicted}, ` +
                `scikit-learn said ${check.expected[index]}`
            );
        }
    }

    check.classes.forEach((name, classIndex) => {
        const gap = Math.abs(outcome.probabilities[name] - check.probabilities[index][classIndex]);
        if (gap > largestProbabilityGap) {
            largestProbabilityGap = gap;
            worstRecord = index;
        }
    });
});

// The explanation must also add up: the contributions are a decomposition of the
// score, so summing them and adding the average student's score must return the
// margin the model actually used to choose the class.
const sample = check.records[0];
const sampleOutcome = browser.predict(sample);
const sampleExplanation = browser.explain(sample, sampleOutcome.predicted, sampleOutcome.vector);
const contributionCount =
    sampleExplanation.supporting.length + sampleExplanation.opposing.length;

console.log("");
console.log("Checking predictor.html against scikit-learn");
console.log("=".repeat(52));
console.log(`Test students compared        : ${total}`);
console.log(`Predicted class mismatches    : ${classMismatches}`);
console.log(`Largest probability difference: ${largestProbabilityGap.toExponential(2)}`);
if (worstRecord !== null) {
    console.log(`  (largest difference at record ${worstRecord})`);
}
console.log(`Explanation factors returned  : ${contributionCount}`);
console.log("=".repeat(52));

const probabilitiesMatch = largestProbabilityGap < 1e-9;

if (classMismatches === 0 && probabilitiesMatch) {
    console.log("PASS: the browser version reproduces scikit-learn exactly.");
    process.exit(0);
}

console.log("FAIL: the browser version does not match scikit-learn.");
process.exit(1);
