# ClimaGuard — Viva notes (DS-4491 ML Systems Design)

Short answers in simple English, each followed by a Banglish explanation (🇧🇩).
The live-demo scripts are in Part B. Run every command from the project root,
with the virtual environment active (`.\venv\Scripts\Activate.ps1`).

---

## Part A — Questions and answers

### 1. What does `dvc repro` do?
It runs the pipeline in `dvc.yaml` from start to end, but **only the stages whose
inputs changed**. For each stage it compares the current hashes of the stage's
`deps` (data, code) and `params` with the hashes saved in `dvc.lock`. If they
match, the stage is skipped ("didn't change, skipping"). If they don't, it
reruns that stage and every stage after it that uses its outputs. At the end it
updates `dvc.lock`.

🇧🇩 `dvc repro` holo "reproduce". Eta `dvc.yaml` dekhe pipeline chalay, kintu
shob stage abar chalay na — jegulor input (data, code ba param) bodlaise shudhu
shegulo ar tar porer stage gulo chalay. Baki gulo skip kore, karon `dvc.lock`
e ager hash mile gese.

### 2. Why DVC?
- Git is built for small text files. Our data (2.2 MB raw, ~20 MB features) and
  models (~60 MB) are binary and change on every run, so Git would bloat.
- DVC keeps the **big files outside Git** (in a cache and a remote) and puts a
  small pointer (hash) in Git. So one Git commit = one exact version of code +
  data + model.
- It also gives us a **pipeline**: each step's inputs and outputs are declared,
  so anyone can rebuild the results with one command and DVC only reruns what
  is needed.

🇧🇩 Git boro binary file (data, model) er jonno banano hoy nai. DVC boro file
gula alada jaygay (cache/remote) rakhe, ar Git e shudhu ekta chotto hash
pointer rakhe. Tai code + data + model er version ekshathe track hoy, ar
`dvc repro` diye je keu same result banate pare.

### 3. What is `dvc.yaml`?
The pipeline definition. It lists the **stages** (prepare → featurize →
select_features → split → train → evaluate → explain → drift → assets). For each
stage: `cmd` (the command to run), `deps` (input files and code), `params`
(which keys of `params.yaml` it uses), `outs` (files it creates), and
`metrics` / `plots` (small result files DVC can show and compare).

🇧🇩 `dvc.yaml` holo pipeline er naksha. Kon stage kon command chalabe, ki input
nibe (deps), kon param use korbe, ar ki output banabe (outs, metrics, plots) —
shob eikhane lekha.

### 4. What is `params.yaml`?
The single file with **every tunable setting**: seed, cleaning rules, rolling
windows and lags, feature-selection thresholds, the split dates, all model
hyperparameters, SHAP sample size, drift thresholds. The stage scripts contain
no hard-coded settings; they read this file. DVC tracks each stage's params, so
changing a value reruns exactly the stages that use it.

🇧🇩 `params.yaml` e shob setting ekjaygay — seed, split er date, model er
hyperparameter, drift threshold. Code e kono number hard-code kora nai. Ekta
value bodlale DVC bujhe kon stage abar chalate hobe.

### 5. How does DVC know which stage to rerun?
`dvc.lock` stores, for every stage, the **MD5 hash** of every dependency and
output file and the **values** of its params from the last successful run.
`dvc status` / `dvc repro` recompute the hashes now and compare. Any
difference → that stage is "changed" → it and everything downstream reruns.

🇧🇩 `dvc.lock` e protita stage er input/output file er MD5 hash ar param er
value save thake. DVC notun hash ber kore purano tar sathe milay. Na mille
stage ta "changed" dhore abar chalay.

### 6. What happens if the dataset changes?
The raw CSV's hash no longer matches `data/raw/...csv.dvc`. We run
`dvc add data/raw/<csv>` to record the new version (new hash in the `.dvc`
file), then `dvc repro`: `prepare` depends on the CSV, so **every stage
reruns**. We commit the new `.dvc` file + `dvc.lock` to Git and `dvc push` the
new data and models. The old version stays recoverable with `git checkout
<old commit>` + `dvc checkout`.

🇧🇩 Dataset bodlale tar hash bodlay. `dvc add` diye notun version record kori,
tarpor `dvc repro` — prothom stage (prepare) data er upor nirbhor kore, tai
puro pipeline abar chole. Purano version o `git checkout` + `dvc checkout` diye
fera jay.

### 7. What happens if a parameter changes?
Only the stages that list that param rerun, plus the stages after them.
Example: `train.rf.n_estimators` → `train` reruns, then `evaluate`, `explain`,
`drift`, `assets` (they use the models). `prepare`, `featurize`,
`select_features`, `split` are skipped. `dvc params diff` shows what changed and
`dvc metrics diff` shows the effect on the results.

🇧🇩 Param bodlale shudhu je stage oi param use kore ar tar porer stage gula
chole. Jemon RF er tree shonkha bodlale train theke shuru, kintu data er stage
gula skip hoy.

### 8. Git vs DVC
| | Git | DVC |
|---|---|---|
| Tracks | code, `dvc.yaml`, `dvc.lock`, `params.yaml`, `*.dvc` pointers, metrics JSON, README | raw data, parquet files, models (`.joblib`), SHAP outputs, dashboard artifacts |
| Stored in | GitHub | DVC cache (`.dvc/cache`) + DVC remote (DagsHub) |
| Size | small text | large / binary |
| Link | a Git commit holds the hashes… | …DVC uses them to fetch the exact files |

🇧🇩 Git code ar chotto text file rakhe (GitHub e). DVC boro data ar model rakhe
(remote storage e). Git e je hash thake, sheta diye DVC thik file ta niye ashe.

### 9. What do `dvc push` and `dvc pull` do?
- `dvc push`: uploads the files in the local DVC cache (data, models, outputs
  referenced by `dvc.lock` and `.dvc` files) to the **remote** (DagsHub). Like
  `git push`, but for data.
- `dvc pull`: downloads those files from the remote into the cache and puts them
  in the workspace (`dvc fetch` + `dvc checkout`).

🇧🇩 `dvc push` = data/model remote e upload. `dvc pull` = remote theke
download kore workspace e boshano. Git push/pull er motoi, kintu boro file er
jonno.

### 10. How does another person reproduce our experiment?
```bash
git clone https://github.com/taohidislamkhan/ClimaGuard.git
cd ClimaGuard
python -m venv venv && venv\Scripts\activate      # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
dvc pull            # data + models + outputs from the DVC remote
dvc status          # "Data and pipelines are up to date."
dvc repro           # nothing to do — or, after `dvc repro -f`, the same metrics
python app.py       # dashboard at http://127.0.0.1:5000
```
Same code + same data hash + same params + fixed seeds → same metrics.
Without remote access they can still run `dvc repro` if they have the raw CSV.

🇧🇩 Onno keu clone kore, `pip install`, `dvc pull` korle amader exact data ar
model pabe. `dvc repro` chalale "up to date" bolbe, karon shob hash mile.
Seed fixed, tai abar chalale o same result ashbe.

### 11. Drift: data drift vs concept drift, detection, response
- **Data drift** = the **inputs** change (the distribution of X). Example: GDP
  per capita keeps rising, so 2024–25 values sit higher than 2015–22.
  **Detection:** for each model feature we compare train (reference) vs test
  (current) with **PSI** (Population Stability Index; bins from the train
  quantiles; ≥ 0.2 = significant) and the **KS test** (p < 0.05). A feature
  counts as drifted only if both agree. We then check whether any of the top-15
  SHAP features drifted.
- **Concept drift** = the **relationship** between X and y changes, so the same
  inputs lead to different outcomes and the model gets worse.
  **Detection:** we score the model on each test quarter and compare it with
  the **same quarter of the validation year** (e.g. 2024 Q3 vs 2023 Q3, so
  seasonality is not mistaken for drift). Flag if macro-F1 drops > 0.05 or RMSE
  rises > 20%.
- **Response:** `drift_report.json` has `retrain_recommended`. If true, move
  `split.train_end` / `split.val_end` forward in `params.yaml` (train on newer
  weeks) and run `dvc repro`; compare with `dvc metrics diff`. Our current
  result: only GDP drifted (not a top feature) and no quarter was flagged, so
  retraining is not required — but we demonstrated the retrain path anyway.

🇧🇩 **Data drift** = input er distribution bodlano (jemon GDP bere jawa).
**Concept drift** = input ar output er shomporko bodlano, model er performance
kome. Data drift PSI + KS test diye dhori; concept drift protita quarter er
score ke ager bochorer oi quarter er sathe milai. Drift hole `params.yaml` e
split er date shamne niye `dvc repro` — model notun data diye abar train hoy.

### 12. Extra questions that may come up
- **Why a chronological split, not random?** Lag and rolling features copy
  values from neighbouring weeks. A random split would put a test week's values
  inside training rows' windows → leakage, inflated scores. Train ≤ 2022,
  validation 2023, test 2024–25.
  🇧🇩 Random split korle future er data training e dhuke jay (lag/rolling er
  maddhome). Tai date diye split.
- **Why are the rolling windows grouped by country?** So a window never mixes
  the last weeks of one country with the first weeks of the next.
- **How is the winner chosen?** On the **validation** split only (macro-F1 /
  RMSE). The test split is never used to choose.
- **Why tree models?** Non-linear effects (heat risk jumps above a temperature
  threshold) and interactions (temperature × PM2.5); no scaling needed; SHAP's
  TreeExplainer is exact for them.
- **Why don't our numbers equal the proposal (RF acc 0.617)?** The proposal
  numbers came before the final pipeline. The DVC pipeline reproduces the
  project's actual pipeline within ±0.002 for the winners (see README).
- **Known issue:** waterborne rolling features include the current week's
  count → that R² is optimistic. Documented, not hidden.

---

## Part B — Live demo script

Say the sentence in *italics* while the command runs.

### 0. Setup (before the viva)
```bash
.\venv\Scripts\Activate.ps1
dvc status          # should be clean
git status          # should be clean
```

### 1. `dvc init`
Show it in a scratch folder so the real repo is not touched:
```bash
mkdir C:\tmp\dvc-demo; cd C:\tmp\dvc-demo; git init; dvc init
```
**Expect:** "Initialized DVC repository." and a new `.dvc/` folder
(`config`, `.gitignore`) plus `.dvcignore`, all staged in Git.
*"`dvc init` turns a Git repo into a DVC repo. It creates `.dvc/` with the
config and the cache location. In our project we ran this once and committed
it."*

### 2. `dvc add`
```bash
cat data/raw/global_climate_health_impact_tracker_2015_2025.csv.dvc
```
**Expect:** `md5: ed50c623...`, `size: 2213853`, `path: ...csv`.
*"We ran `dvc add` on the raw CSV. The real file went into the DVC cache and
Git only stores this small pointer file with its MD5 hash. `.gitignore` stops
Git from committing the CSV itself."*

### 3. `dvc dag`
```bash
dvc dag
```
**Expect:** the graph raw.csv.dvc → prepare → featurize → select_features →
split → train → evaluate / explain → drift, and assets at the bottom.
*"This is our pipeline as a graph. Each box is a stage in `dvc.yaml`; arrows
come from the deps/outs. DVC builds this automatically from the file paths."*

### 4. `dvc status`
```bash
dvc status
```
**Expect:** `Data and pipelines are up to date.`
*"Every hash in `dvc.lock` matches the files on disk, so nothing needs to run."*

### 5. `dvc repro` (nothing changed)
```bash
dvc repro
```
**Expect:** `Stage 'prepare' didn't change, skipping` … for every stage.
*"Because nothing changed, DVC skips all nine stages — this is the caching."*

### 6. Change a parameter → partial rerun → `dvc metrics diff`
```bash
# edit params.yaml: train.rf.n_estimators: 200 -> 100
dvc status          # train: changed params: train.rf.n_estimators
dvc repro           # prepare..split skipped; train, evaluate, explain, drift, assets rerun
dvc params diff
dvc metrics diff
git checkout params.yaml   # revert
dvc repro                  # back to the committed results (or: git checkout dvc.lock; dvc checkout)
```
**Expect:** `dvc status` names only `train` (and downstream after it runs);
`dvc metrics diff` shows the Random Forest numbers moving slightly while the
winner (Logistic Regression) is unchanged.
*"Only stages that depend on `train` params rerun. The data stages are skipped.
`dvc metrics diff` compares the new metrics with the last commit."*
(`train` takes about 4 minutes; say this before starting.)

### 7. `dvc metrics show`
```bash
dvc metrics show
```
**Expect:** a table with `classifier_metrics.json` (winner.accuracy ≈ 0.820,
winner.macro_f1 ≈ 0.819, winner.roc_auc ≈ 0.944, rf.accuracy ≈ 0.809 …),
`regressor_metrics.json` (vector.winner.r2 ≈ 0.916, heat ≈ 0.832 …),
`drift_report.json`, `data_quality.json` (negative_aqi_fixed = 374), etc.
*"These JSON files are declared as metrics and tracked in Git, so every commit
has its results."*

### 8. `dvc plots show`
```bash
dvc plots show
start dvc_plots/index.html      # Mac: open, Linux: xdg-open
```
**Expect:** an HTML page with the confusion matrix, predicted-vs-actual
scatter for each disease, and the quarterly drift line chart.
*"DVC renders the plot CSVs from the evaluate and drift stages. `dvc plots
diff` can compare two commits."*

### 9. `dvc push`
```bash
dvc push
```
**Expect:** `N files pushed` or `Everything is up to date.`
*"This uploads the data and models from our cache to DagsHub. The credentials
are in `.dvc/config.local`, which is git-ignored."*

### 10. `dvc pull` (fresh clone)
```bash
cd C:\tmp; git clone https://github.com/taohidislamkhan/ClimaGuard.git fresh; cd fresh
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
dvc remote modify --local origin auth basic
dvc remote modify --local origin user <dagshub-user>
dvc remote modify --local origin password <dagshub-token>
dvc pull
dvc status          # Data and pipelines are up to date.
```
*"A fresh clone has only code and pointers. `dvc pull` downloads exactly the
data and models that this commit's `dvc.lock` points to, and `dvc status`
confirms everything matches."*

### 11. Drift demo (bonus)
```bash
cat reports/drift/drift_report.json      # retrain_recommended, data_drift, concept_drift
# retrain response: move the split forward
# params.yaml: split.train_end "2023-12-31", split.val_end "2024-06-30"
dvc repro
dvc metrics diff
git checkout params.yaml; git checkout dvc.lock; dvc checkout     # revert
```
*"Our response policy: when drift is found we retrain on newer weeks by moving
the split dates. DVC reruns split onward and `dvc metrics diff` shows the
effect."*
