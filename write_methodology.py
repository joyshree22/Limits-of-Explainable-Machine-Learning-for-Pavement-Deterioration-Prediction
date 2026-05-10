"""
Generate Methodology_new.docx — 7-section journal-ready version.

Structure (user-specified outline):
  §3.1  Overall Modelling Framework
  §3.2  Dataset Description          (incl. visit-level aggregation)
  §3.3  Feature Engineering          (incl. composites, leakage, Δ formulation)
  §3.4  Data Partitioning Strategy   (incl. GroupKFold + LOO)
  §3.5  Model Development            (incl. HPO, weighting, SHAP)
  §3.6  Evaluation Metrics           (incl. skill, Δ-scale, bootstrap CI)
  §3.7  Baseline Models

All exact values verified against Methodology.docx and resultandd copy.docx.
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor

doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)

def _r(run):
    run.font.name = "Times New Roman"
    run.font.color.rgb = RGBColor(0, 0, 0)

def h1(text):
    p = doc.add_heading(text, level=1)
    r = p.runs[0]; r.font.size = Pt(14); r.font.bold = True; _r(r)
    p.paragraph_format.space_before = Pt(18); p.paragraph_format.space_after = Pt(6)

def h2(text):
    p = doc.add_heading(text, level=2)
    r = p.runs[0]; r.font.size = Pt(13); r.font.bold = True; _r(r)
    p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(4)

def h3(text):
    p = doc.add_heading(text, level=3)
    r = p.runs[0]; r.font.size = Pt(12); r.font.bold = True; _r(r)
    p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(2)

def body(text):
    p = doc.add_paragraph(text)
    p.style = doc.styles["Normal"]
    p.paragraph_format.first_line_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(6)

def eq(label, text):
    p = doc.add_paragraph()
    p.style = doc.styles["Normal"]
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.left_indent = Inches(0.75)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(f"{text}    ({label})")
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)

def tbl_cap(text):
    p = doc.add_paragraph(text)
    p.style = doc.styles["Normal"]
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(3)
    p.runs[0].font.italic = True; p.runs[0].font.size = Pt(11)

def add_table(headers, rows, caption=None):
    if caption:
        tbl_cap(caption)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = h
        for r in t.rows[0].cells[i].paragraphs[0].runs:
            r.font.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(11)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            t.rows[ri + 1].cells[ci].text = val
            for r in t.rows[ri + 1].cells[ci].paragraphs[0].runs:
                r.font.name = "Times New Roman"; r.font.size = Pt(11)
    doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
h1("3.  Methodology Framework")
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
h2("3.1  Overall Modelling Framework")
# ════════════════════════════════════════════════════════════════════════════

body(
    "This study adopts a structured, sequential modelling pipeline designed to evaluate "
    "the predictive limits and explainability of tree-based ensemble machine learning for "
    "pavement deterioration prediction using the LTPP longitudinal database. The pipeline "
    "comprises five interdependent stages executed in strict sequential order: "
    "(i) dataset construction and visit-level aggregation; "
    "(ii) feature engineering and leakage prevention; "
    "(iii) section-stratified data partitioning and cross-validation design; "
    "(iv) ensemble model training, hyperparameter optimisation, and explainability "
    "analysis; and (v) multi-dimensional evaluation against pre-specified benchmarks "
    "and consistency criteria. Fig. 1 illustrates the end-to-end workflow."
)

body(
    "The pipeline distinguishes two prediction tasks: a design-stage task in which only "
    "information available at pavement design time is used (structural geometry, material "
    "properties, traffic loading, climate exposure, and age), and a monitoring-stage task "
    "that additionally incorporates the most recently observed condition value as a "
    "temporal lag predictor. Twelve models are trained in total: two tree-based "
    "architectures (XGBoost and Random Forest) × three deterioration targets (IRI, "
    "rut depth, cracking index) × two prediction tasks (design, monitoring)."
)

body(
    "All methodological decisions — feature inclusion criteria, leakage rules, "
    "partitioning scheme, benchmark specifications, evaluation metrics, and pre-specified "
    "consistency checks — were finalised before any ensemble model was evaluated on the "
    "held-out test set. All preprocessing parameters are estimated exclusively on "
    "training-set observations and applied without re-estimation to validation and test "
    "sections. The entire pipeline is implemented as a deterministic workflow to ensure "
    "full computational reproducibility."
)

# ════════════════════════════════════════════════════════════════════════════
h2("3.2  Dataset Description")
# ════════════════════════════════════════════════════════════════════════════

body(
    "The empirical foundation of this study is the Long-Term Pavement Performance (LTPP) "
    "database maintained by the Federal Highway Administration (FHWA). LTPP is a "
    "longitudinal observational programme that collects pavement structural, traffic, "
    "climate, material, and performance measurements from instrumented pavement sections "
    "distributed across North America. The extract used in this study contains 4,397 raw "
    "rows at 432 source variables spanning the period 1989–2021."
)

body(
    "The analysis focuses exclusively on flexible pavement sections from the LTPP General "
    "Pavement Studies (GPS-1) programme. A total of 48 sections were selected across four "
    "climatically distinct regions: Arizona (22 sections), Georgia (12), Ohio (7), and "
    "Ontario, Canada (7). These regions span a 210× freeze-index gradient from the "
    "hot-arid desert conditions of Arizona to the severe freeze–thaw continental climate "
    "of Ontario. Annual freeze index (FI), computed as cumulative degree-days below 0 °C, "
    "characterises the regional climate exposure: Arizona (mean FI = 5.9), Georgia "
    "(mean FI = 10.9), Ohio (mean FI = 316.6), and Ontario (mean FI = 834.9). The "
    "regional grouping was selected to evaluate whether ensemble models trained under "
    "pooled conditions remain transferable across substantially different climatic and "
    "operational environments."
)

body(
    "Three pavement deterioration indicators serve as independent prediction targets: "
    "International Roughness Index (IRI, m/km), representing longitudinal surface "
    "roughness; mean rut depth (mm), representing permanent deformation under repeated "
    "loading; and HPMS composite cracking index (% lane area), representing surface "
    "distress extent. Each row in the LTPP extract represents a single site-visit "
    "observation, with multiple observations sharing a common pavement-section identifier. "
    "This hierarchical structure — repeated temporal observations nested within sections — "
    "governs all partitioning, cross-validation, and uncertainty-quantification "
    "procedures throughout the study."
)

body(
    "IRI measurements in LTPP are collected at approximately five independent passes per "
    "site visit (left and right wheel paths, centre lane, and repeated runs), all recorded "
    "on the same observation date. The raw IRI dataset therefore contains 3,233 records "
    "representing 650 unique (section, date) combinations across 48 sections (mean "
    "4.97 raw records per visit). For all modelling, the IRI dataset is aggregated to "
    "visit level: for each (section, date) pair, the mean IRI across all same-day runs "
    "is taken as the representative observation. This reduces the effective IRI sample "
    "from 3,233 to 650 visit-level observations, which is the correct denominator for "
    "all capacity, bootstrap, and confidence-interval calculations in this study. "
    "Rutting (566 records) and distress (378 records) are collected once per visit and "
    "are used as recorded without further aggregation."
)

body(
    "Pavement age is derived from two near-complete LTPP date fields and used as a "
    "continuous predictor and as the basis for the age-decay benchmark:"
)
eq("1", "AGE_YEARS  =  (visit_date − construction_date) / 365.25")

body(
    "All 48 sections belong to one of three structural pavement families used as "
    "imputation strata: ACUB (asphalt concrete over unbound granular base), "
    "ACATB (asphalt concrete over asphalt-treated base), and ACTB (asphalt concrete "
    "over cement-treated base). Table 3.1 summarises the regional dataset composition."
)

add_table(
    ["Region", "Sections", "Raw IRI records", "Effective IRI obs.*",
     "Rutting obs.", "Year range", "Mean FI"],
    [
        ["Arizona", "22", "1,469", "290", "254", "1989–2021", "5.9"],
        ["Georgia", "12",   "477",  "95", "106", "1990–2009", "10.9"],
        ["Ohio",     "7",   "740", "162", "114", "1996–2016", "316.6"],
        ["Ontario",  "7",   "547", "103",  "92", "1989–2016", "834.9"],
        ["Total",   "48", "3,233", "650", "566", "1989–2021", "207.4"],
    ],
    caption=(
        "Table 3.1. Dataset summary by region. "
        "* Effective IRI observations after visit-level aggregation to unique "
        "(section, date) means."
    ),
)

# ════════════════════════════════════════════════════════════════════════════
h2("3.3  Feature Engineering")
# ════════════════════════════════════════════════════════════════════════════

body(
    "Feature engineering was designed to preserve temporal honesty and prevent "
    "information leakage. A predictor was admitted to the design-stage feature set only "
    "if it could plausibly be known at or before the time of pavement design or "
    "rehabilitation, prior to observing the deterioration target at the prediction visit."
)

h3("3.3.1  Selected and Removed Variables")

body(
    "After removing target variables, administrative identifiers, date fields, and "
    "sub-indicator duplicates from the 432 source columns, approximately 388 candidate "
    "predictor columns remain across four engineering domains. A 50% missingness "
    "threshold is applied independently per target dataset at the feature level — no "
    "observations are discarded at this stage. After threshold application, approximately "
    "200–220 features are retained per target. Table 3.2 summarises the domain "
    "structure after this step."
)

add_table(
    ["Domain", "Approx. features retained", "Representative variables"],
    [
        ["Pavement structure", "~42",
         "AC layer count, total AC thickness, bulk specific gravity, air voids"],
        ["Traffic", "~45",
         "AADTT; 38 MEPDG axle-load distribution factors (one cluster representative "
         "retained after collinearity reduction); vehicle-class composition"],
        ["Climate", "~22",
         "Freeze index, freeze-thaw cycle count, precipitation, mean temperature "
         "(annual and seasonal decompositions)"],
        ["Materials and operations", "~27",
         "Subgrade resilient modulus, base gradation parameters, Atterberg limits"],
    ],
    caption=(
        "Table 3.2. Feature domain structure after missingness threshold. "
        "Feature counts are approximate prior to collinearity reduction."
    ),
)

body(
    "Two explicit exclusion rules govern the design-stage feature set. First, same-visit "
    "condition variables — any LTPP field recording pavement condition observed during "
    "the same inspection as the prediction target — are excluded from all design models. "
    "These variables are unavailable at construction time and constitute direct leakage: "
    "earlier pipeline iterations retaining same-visit distress indicators produced "
    "near-perfect distress R², which collapsed to 0.278 upon their removal, confirming "
    "that the previous performance inflation originated from leakage rather than genuine "
    "predictive capability. Second, geographic identifiers (latitude, longitude, state "
    "code, climate-zone label) are excluded to prevent regional proxy encoding that would "
    "suppress LOO transfer failure. It is acknowledged that traffic-loading distributions "
    "and layer-material descriptors may still embed regional construction practices "
    "indirectly; this residual confounding is evaluated analytically through the "
    "leave-one-region-out framework described in Section 3.4."
)

body(
    "Remaining missing values are imputed using group-median imputation stratified by "
    "pavement family (ACUB, ACATB, ACTB). Imputation medians are computed exclusively "
    "on training-section observations and applied without re-estimation to validation "
    "and test sections. All continuous features are then standardised using StandardScaler "
    "(zero mean, unit variance), fitted on training sections only. Min-max scaling is "
    "explicitly not used: Ontario's freeze index values (mean 834.9, maximum 1,559) "
    "lie an order of magnitude above Arizona's (mean 5.9, maximum 107). Under min-max "
    "normalisation, all non-Ontario sections would compress to a narrow region near zero "
    "while Ontario sections approach 1.0 — effectively encoding regional identity into "
    "the scale transformation and contradicting the purpose of climate-transfer testing. "
    "StandardScaler eliminates this artefact without distorting relative distances."
)

body(
    "Spearman rank correlation is used for collinearity detection because "
    "pavement-performance relationships are nonlinear and do not satisfy Pearson "
    "assumptions. A hierarchical clustering procedure is applied to training-set "
    "observations only: pairwise Spearman matrix → distance matrix (1 − |ρ|) → "
    "agglomerative clustering (average linkage) → dendrogram cut at |ρ| = 0.85 → one "
    "representative retained per cluster (priority order: physics composite > directly "
    "measured variable > aggregate > fewest missing, with target-correlation tiebreaker) "
    "→ selection applied to validation and test sets without re-estimation. This "
    "procedure reduces the feature set to approximately 114–118 retained features "
    "per target."
)

h3("3.3.2  Physics-Based Composite Features")

body(
    "Nine composite features encoding pavement-deterioration mechanisms are constructed "
    "from retained variables before any data split and applied identically to all "
    "partitions (Table 3.3). The structural freeze-insulation composite uses the form "
    "THICK_AC × 1/(1 + CLIM_FI) rather than the ratio THICK_AC / CLIM_FI to avoid "
    "undefined values at FI = 0, which affects 32.3% of all records (Arizona and Georgia "
    "sections with genuinely zero freezing degree-days)."
)

add_table(
    ["Composite feature", "Formula", "Physical mechanism encoded"],
    [
        ["Wet-freeze compound",
         "CLIM_FT × CLIM_PRECIP",
         "Freeze cycles amplified by moisture availability"],
        ["Structural freeze insulation",
         "THICK_AC × 1/(1 + CLIM_FI)",
         "Thicker AC buffers freeze-induced base damage; defined at FI = 0"],
        ["Material freeze susceptibility",
         "CLIM_FT × AC_BSG",
         "Denser mix resists freeze-thaw deterioration"],
        ["Cumulative wet-freeze damage",
         "CLIM_FI × CLIM_PRECIP × AGE_YEARS",
         "Integrated climate exposure accumulated over service life"],
        ["Freeze stress per layer",
         "CLIM_FI / LAYER_COUNT_AC",
         "Freeze loading distributed across structural layer count"],
        ["Traffic-climate interaction",
         "AADTT × CLIM_FI",
         "Heavy traffic compounded by freeze severity"],
        ["Structural adequacy index",
         "THICK_AC / UB_MR",
         "AC depth relative to subgrade stiffness"],
        ["Thermal gradient compound",
         "(TEMP_SUM − TEMP_WIN) × CLIM_FI",
         "Seasonal thermal amplitude under freeze exposure"],
        ["Age-climate compound",
         "AGE_YEARS × CLIM_FI",
         "Cumulative service age under freeze conditions"],
    ],
    caption=(
        "Table 3.3. Physics-based composite features. CLIM_FI = freeze index; "
        "CLIM_FT = freeze-thaw cycle count; THICK_AC = total AC thickness; "
        "UB_MR = subgrade resilient modulus; AC_BSG = AC bulk specific gravity."
    ),
)

h3("3.3.3  Delta Formulation for Monitoring Models")

body(
    "The study distinguishes two prediction tasks. The design task uses only features "
    "known at or before construction with no observed condition information. The "
    "monitoring task adds one temporal lag: the most recently observed target value "
    "within a preceding lag window, constructed date-aware rather than by row offset "
    "because LTPP inspection intervals are irregular. Lag windows are derived from the "
    "empirical inter-visit gap distributions, selecting the smallest window retaining "
    "at least 60% of consecutive visit pairs. The inter-visit gap distribution for IRI "
    "has a median of 393 days, mean 519 days, and 75th percentile of 704 days. Table 3.4 "
    "reports the target-specific lag windows adopted."
)

add_table(
    ["Target", "Lag window", "Basis"],
    [
        ["IRI",
         "730 days (2 years)",
         "Median gap 393 days; 365-day window loses 58% of transitions"],
        ["Rutting",
         "730 days (2 years)",
         "Median gap 450 days; 365-day window loses 64% of transitions"],
        ["Distress",
         "1,095 days (3 years)",
         "Median gap 748 days; 365-day window loses 78% of transitions"],
    ],
    caption="Table 3.4. Target-specific lag windows for monitoring models.",
)

body(
    "The monitoring model predicts the deterioration increment (delta) rather than the "
    "absolute level, separating genuine structural learning from lag reconstruction:"
)
eq("2", "ΔIRI_t  =  IRI_t  −  IRI_{t−1}")
body(
    "Delta predictions are remapped to the absolute scale for comparison with "
    "persistence and for reporting alongside design-stage results:"
)
eq("3", "IRI_hat_t  =  IRI_{t−1}  +  ΔIRI_hat_t")
body(
    "Performance is reported on both the absolute IRI scale and the delta scale (Eq. 2). "
    "High absolute R² combined with low delta-scale skill indicates that monitoring "
    "performance arises primarily from autocorrelation of the lag input rather than "
    "from structural learning. Both scales are therefore reported for full transparency."
)

# ════════════════════════════════════════════════════════════════════════════
h2("3.4  Data Partitioning Strategy")
# ════════════════════════════════════════════════════════════════════════════

body(
    "All partitioning is performed at the pavement-section level: every observation "
    "from a given section is assigned to exactly one of training, validation, or test. "
    "Row-level splitting is explicitly avoided because repeated observations from the "
    "same section share common structural history and climate exposure and are not "
    "statistically independent. A row-level random split would allow earlier visits "
    "from a section to appear in training while later visits from the same section "
    "appear in test, effectively providing the model with the deterioration trajectory "
    "of test sections during training."
)

body(
    "The 48 sections are stratified by region and allocated approximately 70/15/15 at "
    "the section level (Table 3.5). The test set is sealed at the moment of "
    "partitioning and is not accessed until all feature, hyperparameter, and "
    "architecture decisions are finalised."
)

add_table(
    ["Partition", "Sections", "IRI obs.", "Rutting obs.", "Distress obs."],
    [
        ["Training",   "34  (16 AZ, 8 GA, 5 OH, 5 ON)", "455", "410", "272"],
        ["Validation", " 7  (3 AZ, 2 GA, 1 OH, 1 ON)",  "109", " 82", " 61"],
        ["Test",       " 7  (3 AZ, 2 GA, 1 OH, 1 ON)",  " 86", " 74", " 45"],
        ["Total",      "48",                             "650", "566", "378"],
    ],
    caption=(
        "Table 3.5. Section-level data partition allocation and effective observation "
        "counts per target after preprocessing."
    ),
)

h3("3.4.1  GroupKFold Cross-Validation")

body(
    "GroupKFold cross-validation, grouped by section ID, is enforced during "
    "hyperparameter optimisation to ensure no section spans both training and validation "
    "folds within any single iteration. This preserves the same statistical independence "
    "guarantee as the outer section-level split: within each fold, all observations from "
    "a section are either entirely in the fold training set or entirely withheld. "
    "Five-fold GroupKFold is applied across the 41 training-plus-validation sections "
    "during Optuna hyperparameter search."
)

h3("3.4.2  Leave-One-Region-Out Validation")

body(
    "Leave-one-region-out (LOO) validation is applied to the design model only and "
    "constitutes the primary transferability test of the study. Four iterations are "
    "performed, each withholding all sections from one region and retraining on the "
    "remaining three. LOO evaluates a substantially stronger condition than conventional "
    "train/test splitting because the withheld region represents a genuinely unseen "
    "climatic and operational environment."
)

body(
    "Hyperparameters selected from full-dataset training are held fixed across all LOO "
    "iterations to isolate transferability from retuning ability. Each LOO iteration "
    "independently refits group-median imputation statistics and StandardScaler on its "
    "three-region training set: no information from the withheld region — not for "
    "imputation, not for scaling, not for collinearity reduction — enters the training "
    "pipeline. Using globally fitted preprocessing parameters in LOO would constitute "
    "information leakage from the withheld region and would invalidate the "
    "transferability test. Table 3.6 characterises the transfer difficulty for each "
    "withheld region across the 210× freeze-index gradient."
)

add_table(
    ["Withheld region", "Mean FI", "Transfer scenario", "Transfer difficulty"],
    [
        ["Arizona",  "5.9",
         "Warm-arid extrapolation",
         "Elevated — model trained predominantly on freeze-influenced behaviour"],
        ["Georgia",  "10.9",
         "Wet-subtropical extrapolation",
         "Elevated — wet-subtropical regime under-represented in training"],
        ["Ohio",     "316.6",
         "Temperate interpolation",
         "Expected modest — FI lies within the training envelope"],
        ["Ontario",  "834.9",
         "Severe-freeze extrapolation",
         "Elevated — extreme freeze-thaw subgrade deformation unseen in training"],
    ],
    caption=(
        "Table 3.6. LOO transfer scenarios by withheld region. FI = annual freeze index "
        "(cumulative degree-days below 0 °C)."
    ),
)

body(
    "Negative LOO R² values are interpreted as substantive findings rather than "
    "anomalies: they indicate that the pooled model performs worse than a regional mean "
    "baseline when transferred to the unseen region."
)

# ════════════════════════════════════════════════════════════════════════════
h2("3.5  Model Development")
# ════════════════════════════════════════════════════════════════════════════

body(
    "Two tree-based ensemble algorithms are trained and evaluated: eXtreme Gradient "
    "Boosting (XGBoost) and Random Forest (RF). XGBoost constructs an additive sequence "
    "of shallow decision trees by iteratively minimising a regularised second-order "
    "Taylor expansion of the residual loss:"
)
eq("4", "ŷ_i^(t)  =  ŷ_i^(t−1)  +  f_t(x_i),    f_t ∈ F")
body(
    "L1 and L2 regularisation terms control tree complexity and reduce overfitting on "
    "the small pavement dataset. Random Forest constructs a large ensemble of decorrelated "
    "trees via bootstrap aggregation and randomised feature subsets, with predictions "
    "averaged across all trees:"
)
eq("5", "ŷ_RF  =  (1/B) Σ_{b=1}^{B} f_b(x)")
body(
    "Both architectures model nonlinear interactions among traffic, structural, climatic, "
    "and material variables without explicit interaction-term specification. Using two "
    "architectures additionally enables the pre-specified cross-model SHAP consistency "
    "check described in Section 3.5.3."
)

h3("3.5.1  Hyperparameter Optimisation")

body(
    "Twelve models are trained in total (2 architectures × 3 targets × 2 tasks). "
    "Hyperparameters are tuned using Optuna's tree-structured Parzen estimator (TPE) "
    "with 5-fold GroupKFold cross-validation grouped by section ID, minimising "
    "validation-set RMSE. Trial budgets are 150 for IRI and 100 for rutting and "
    "distress, reflecting the larger effective IRI sample. Following selection, final "
    "models are retrained on the combined training and validation set (41 sections) "
    "with the selected hyperparameters unchanged. Table 3.7 reports the hyperparameter "
    "search spaces."
)

add_table(
    ["Parameter", "XGBoost search range", "Random Forest search range"],
    [
        ["n_estimators",                  "50–500",           "50–500"],
        ["max_depth",                     "3–12",             "3–12"],
        ["learning_rate",                 "0.01–0.3 (log)",   "—"],
        ["subsample / colsample_bytree",  "0.5–1.0 each",     "—"],
        ["reg_alpha, reg_lambda",         "10⁻⁸–10 (log)",   "—"],
        ["min_samples_split",             "—",                "2–20"],
        ["min_samples_leaf",              "—",                "1–10"],
        ["max_features",                  "—",                "sqrt, log2, 0.3–0.9"],
    ],
    caption="Table 3.7. Hyperparameter search spaces for XGBoost and Random Forest.",
)

body(
    "Arizona comprises 22 of 48 sections (45.7% of IRI observations). To prevent "
    "implicit optimisation for arid deterioration behaviour, inverse-frequency sample "
    "weights proportional to the reciprocal of each region's observation count are "
    "applied during training and normalised to a mean of 1.0."
)

h3("3.5.2  Pre-Specified Consistency Checks")

body(
    "Four consistency checks were pre-specified before final model interpretation to "
    "evaluate whether apparently strong predictive performance remains methodologically "
    "valid under corrected evaluation conditions. These checks target four principal "
    "failure modes: information leakage, persistence dominance, regional "
    "non-transferability, and unstable explanation behaviour."
)

add_table(
    ["Check", "Evidence criterion", "Failure mode targeted"],
    [
        ["Leakage sentinel",
         "Design-distress R² after same-visit variable removal collapses from "
         "near-perfect to 0.278",
         "Direct information leakage"],
        ["Persistence benchmark",
         "Monitoring vs. persistence evaluated on identical 251-observation IRI "
         "common set",
         "Optimistic forecast-skill claims"],
        ["Regional transfer",
         "LOO R² examined for negative values across withheld regions",
         "Non-transferable pooled relationships"],
        ["Explanation stability",
         "SHAP Spearman ρ ≥ 0.750 across XGBoost and RF importance vectors "
         "(pre-specified threshold)",
         "Unstable mechanistic interpretation"],
    ],
    caption=(
        "Table 3.8. Pre-specified consistency checks applied before test-set "
        "interpretation."
    ),
)

h3("3.5.3  SHAP Explainability Framework")

body(
    "Model explainability is evaluated using SHapley Additive exPlanations (SHAP) "
    "computed via the TreeExplainer algorithm, which exploits the internal tree structure "
    "of XGBoost to compute exact Shapley values without approximation. SHAP decomposes "
    "each prediction into additive feature contributions relative to the expected model "
    "output over the training data. Global feature importance is summarised as the mean "
    "absolute SHAP value across test observations."
)

body(
    "SHAP values describe associations within model behaviour on this specific dataset; "
    "they do not establish causal relationships between features and deterioration "
    "outcomes, and all attribution language is restricted accordingly. The following "
    "SHAP outputs are computed: (i) global mean |SHAP| importance bar chart; "
    "(ii) beeswarm distribution coloured by feature quantile; (iii) one regional "
    "waterfall plot per climate zone; and (iv) regional SHAP stratification, with "
    "importance recalculated on the per-climate-zone test subset."
)

body(
    "Cross-model explanation stability is evaluated using Spearman rank correlation "
    "between the XGBoost and RF global SHAP importance vectors. The threshold ρ > 0.750 "
    "is pre-specified before any SHAP results are examined. If ρ ≤ 0.750, attribution "
    "claims are restricted to the highest-ranking features shared consistently across "
    "both architectures. Partial dependence plots (PDPs) are computed for five key "
    "predictors and two feature pairs using quantile-spaced grids (50 points for "
    "one-way; 30 × 30 for two-way) and are interpreted as internal model response "
    "surfaces rather than universal deterioration relationships."
)

# ════════════════════════════════════════════════════════════════════════════
h2("3.6  Evaluation Metrics")
# ════════════════════════════════════════════════════════════════════════════

body(
    "Model performance is assessed using three complementary metrics computed on the "
    "held-out test set. The coefficient of determination (R²) quantifies the proportion "
    "of variance explained relative to a mean predictor:"
)
eq("6", "R²  =  1  −  Σ(y_i − ŷ_i)²  /  Σ(y_i − ȳ)²")
body("Root mean squared error and mean absolute error measure absolute prediction error:")
eq("7", "RMSE  =  √[ (1/n) Σ(y_i − ŷ_i)² ]")
eq("8", "MAE   =  (1/n) Σ|y_i − ŷ_i|")

body(
    "All three metrics are computed at the observation level on the test partition. "
    "R² is additionally computed on the delta scale (Eq. 2) for monitoring models, "
    "providing a direct measure of deterioration-increment forecasting skill independent "
    "of lag-input autocorrelation."
)

h3("3.6.1  Skill Score vs. Persistence")

body(
    "For monitoring models, forecast skill over the persistence benchmark is quantified "
    "as the proportional reduction in RMSE relative to the persistence baseline:"
)
eq("9", "Skill  =  1  −  RMSE_model / RMSE_persistence")
body(
    "A common evaluation set is enforced for all monitoring-versus-persistence "
    "comparisons: all monitoring models and all persistence benchmarks are evaluated "
    "on identical (section, date) rows — those possessing a confirmed prior observation "
    "within the most restrictive persistence window. For IRI this common set comprises "
    "251 visit-observations (those with a prior observation within 365 days). Using "
    "different evaluation subsets for the trained model and the naive baseline would "
    "allow an optimistic comparison in which the monitoring model is assessed on a "
    "richer subset than the persistence model it is evaluated against."
)

h3("3.6.2  Bootstrap Confidence Intervals")

body(
    "With only seven test sections per target, point estimates of R² carry substantial "
    "sampling uncertainty. Confidence intervals are computed by resampling at the section "
    "level over 2,000 bootstrap iterations: each iteration resamples seven sections with "
    "replacement, retaining all observations belonging to each sampled section as a "
    "block. R² is recomputed on the concatenated observations of the sampled sections; "
    "the 2.5th–97.5th percentile interval forms the 95% CI. Section-level resampling "
    "preserves within-section temporal dependence; row-level resampling would "
    "underestimate uncertainty by treating correlated observations as independent draws. "
    "All R² values in the main results table are reported as point estimate "
    "[lower CI, upper CI]; wide intervals are interpreted substantively as evidence "
    "of limited certainty rather than dismissed as artefacts."
)

# ════════════════════════════════════════════════════════════════════════════
h2("3.7  Baseline Models")
# ════════════════════════════════════════════════════════════════════════════

body(
    "Four baseline models are computed before any ensemble model is trained, "
    "establishing the minimum performance threshold that ensemble models must exceed "
    "to demonstrate meaningful predictive skill. All baselines are evaluated on the "
    "same held-out test partition as the ensemble models."
)

body(
    "(i) Mean predictor. Every test observation receives the training-set mean of the "
    "target variable. This is the absolute performance floor; R² = 0 by construction. "
    "Any model failing to outperform the mean predictor has no predictive content "
    "beyond the marginal distribution of the training target."
)

body(
    "(ii) Age-only empirical baseline. An exponential function "
    "IRI = a · exp(b · AGE_YEARS) is fitted by nonlinear least squares on training "
    "observations, with parameters a and b estimated directly from the LTPP data rather "
    "than from published AASHTO look-up tables. This baseline evaluates whether "
    "deterioration can be explained primarily by time in service, without structural, "
    "traffic, or climate inputs."
)

body(
    "(iii) Ridge regression. Fitted across all 114–118 retained features with "
    "regularisation parameter α tuned on the validation set over a log-spaced grid from "
    "10⁻⁴ to 10⁴, evaluating whether linear combinations of the full retained feature "
    "set generalise to unseen sections."
)

body(
    "(iv) Persistence at k = 1, 2, and 3 years. The prediction is the most recently "
    "observed target value within k years prior to the prediction date; any observation "
    "lacking a qualifying prior within k years is excluded from that k-year evaluation. "
    "Three persistence horizons are reported to characterise how forecast skill varies "
    "with lag recency: the 1-year persistence benchmark, which exploits the most recent "
    "observation, represents the most demanding competitor for monitoring models because "
    "pavement deterioration is highly autocorrelated over short intervals."
)

body(
    "All analyses were implemented in Python 3.11 using pandas, NumPy, scikit-learn, "
    "XGBoost (v2.0), SHAP (v0.43), Optuna (v3.4), SciPy, and matplotlib. Fitted models, "
    "preprocessing parameters, and all intermediate artefacts are serialised alongside "
    "results to support full computational reproducibility."
)

doc.save("Methodology_new.docx")
print("Saved: Methodology_new.docx")
