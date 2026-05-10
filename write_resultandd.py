"""
Generate resultandd.docx from the corrected primary-analysis outputs.

The old resultandd document was a full Results and Discussion chapter with
styled tables, many figures, and section-by-section narrative. This generator
keeps that chapter shape, but uses the repaired pipeline outputs and avoids the
earlier leakage-driven or post-hoc claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).parent
RES = ROOT / "results"
FIG = ROOT / "figures"
OUT = ROOT / "resultandd.docx"

TARGET_LABELS = {
    "iri": "IRI",
    "rutting": "Rutting",
    "distress": "Distress",
}


def read_csv(name: str, required: bool = True) -> pd.DataFrame:
    path = RES / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required result file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_parquet(name: str, required: bool = False) -> pd.DataFrame:
    path = RES / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required result file: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def first_metric(df: pd.DataFrame, model: str, column: str) -> float | None:
    if df.empty or "model" not in df.columns or column not in df.columns:
        return None
    rows = df[df["model"] == model]
    if rows.empty:
        return None
    value = rows.iloc[0][column]
    return None if pd.isna(value) else float(value)


def model_parts(model: str) -> tuple[str, str, str]:
    parts = str(model).split("_")
    arch = parts[0].upper() if parts else ""
    task = " ".join(parts[1:-1]).replace("delta", "Delta").title()
    target = TARGET_LABELS.get(parts[-1], parts[-1].upper() if parts else "")
    return arch, task, target


def friendly_model(model: str) -> str:
    arch, task, target = model_parts(model)
    return f"{arch} {task} {target}".replace("  ", " ").strip()


def add_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text: str, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(8.5)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(11)

    for style_name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.color.rgb = RGBColor(31, 78, 121)

    return doc


def heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(8 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)


def para(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.line_spacing = 1.08
    p.paragraph_format.space_after = Pt(5)


def bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(8)


def table_title(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(10)


def add_table(
    doc: Document,
    title: str,
    df: pd.DataFrame,
    columns: Iterable[str],
    labels: dict[str, str] | None = None,
    digits: int = 3,
) -> None:
    labels = labels or {}
    columns = [c for c in columns if c in df.columns]
    table_title(doc, title)
    if df.empty or not columns:
        para(doc, "No rows were available for this table.")
        return

    table = doc.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for idx, col in enumerate(columns):
        cell = table.rows[0].cells[idx]
        add_cell_shading(cell, "1F4E79")
        set_cell_text(cell, labels.get(col, col), bold=True, color="FFFFFF")

    for row_idx, (_, row) in enumerate(df.iterrows()):
        cells = table.add_row().cells
        if row_idx % 2:
            for cell in cells:
                add_cell_shading(cell, "EAF2F8")
        for col_idx, col in enumerate(columns):
            value = row.get(col)
            text = fmt(value, digits=digits)
            if col == "model":
                text = friendly_model(text)
            set_cell_text(cells[col_idx], text)

    doc.add_paragraph()


def add_figure(doc: Document, filename: str, caption_text: str, width: float = 6.4) -> None:
    path = FIG / filename
    if not path.exists():
        para(doc, f"[Figure missing: {filename}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    caption(doc, caption_text)


def compact_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "model" in out.columns:
        out = out.assign(model=out["model"].map(friendly_model))
    return out


def target_counts() -> pd.DataFrame:
    rows = []
    for target in TARGET_LABELS:
        features = read_parquet(f"dataset_{target}_features.parquet")
        visit = read_parquet(f"dataset_{target}_visit.parquet")
        selected_train = read_parquet(f"selected_{target}_train.parquet")
        selected_val = read_parquet(f"selected_{target}_val.parquet")
        selected_test = read_parquet(f"selected_{target}_test.parquet")
        rows.append(
            {
                "Target": TARGET_LABELS[target],
                "Feature rows": len(features),
                "Visit rows": len(visit),
                "Train rows": len(selected_train),
                "Validation rows": len(selected_val),
                "Test rows": len(selected_test),
            }
        )
    return pd.DataFrame(rows)


def split_summary() -> pd.DataFrame:
    assignments = read_csv("section_assignment.csv", required=False)
    if assignments.empty:
        return pd.DataFrame()
    summary = (
        assignments.groupby(["region", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["train", "val", "test"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["total_sections"] = summary[["train", "val", "test"]].sum(axis=1)
    return summary[["region", "train", "val", "test", "total_sections"]]


def benchmark_view(bench: pd.DataFrame) -> pd.DataFrame:
    return bench.copy()


def test_view(test: pd.DataFrame) -> pd.DataFrame:
    out = test.copy()
    if "model" in out.columns:
        out["Target"] = out["model"].map(lambda m: model_parts(m)[2])
        out["Task"] = out["model"].map(lambda m: model_parts(m)[1])
        out["Model"] = out["model"].map(lambda m: model_parts(m)[0])
    return out[
        ["Target", "Task", "Model", "n_obs", "n_sections", "R2", "R2_CI_lower", "R2_CI_upper", "RMSE", "MAE"]
    ]


def consistency_checks(test: pd.DataFrame, common: pd.DataFrame, loo: pd.DataFrame, shap: pd.DataFrame) -> pd.DataFrame:
    distress_r2 = first_metric(test, "xgb_design_distress", "R2")
    monitoring_r2 = first_metric(common, "xgb_monitoring_iri_common", "R2")
    persistence_r2 = first_metric(common, "persist_k1_common_iri", "R2")
    ontario = loo[(loo.get("withheld_region") == "Ontario") & (loo.get("arch") == "xgb")]
    ontario_r2 = None if ontario.empty else float(ontario.iloc[0]["R2"])
    rho = None if shap.empty else float(shap.iloc[0]["spearman_rho"])
    threshold = None if shap.empty else float(shap.iloc[0]["threshold"])

    rows = [
        {
            "Check": "Leakage sentinel",
            "Evidence": f"XGB design distress R2 = {fmt(distress_r2)}",
            "Interpretation": "Near-perfect distress accuracy disappeared after same-visit distress indicators were removed.",
        },
        {
            "Check": "Persistence benchmark",
            "Evidence": f"Common-set XGB IRI R2 = {fmt(monitoring_r2)}; persistence R2 = {fmt(persistence_r2)}",
            "Interpretation": "Monitoring performance is not automatically useful unless it beats persistence on identical rows.",
        },
        {
            "Check": "Regional transfer",
            "Evidence": f"Ontario XGB LOO R2 = {fmt(ontario_r2)}",
            "Interpretation": "The corrected analysis does not support the old positive Ontario-transfer claim.",
        },
        {
            "Check": "Explanation stability",
            "Evidence": f"SHAP rank rho = {fmt(rho)}; threshold = {fmt(threshold)}",
            "Interpretation": "Interpretability is useful descriptively, but the feature ranking is below the pre-specified stability threshold.",
        },
    ]
    return pd.DataFrame(rows)


def top_shap_features(shap_global: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if shap_global.empty:
        return pd.DataFrame()
    xgb = shap_global[
        (shap_global["arch"].str.lower() == "xgb")
        & (shap_global["target"].str.lower() == "iri")
        & (shap_global["task"].str.lower() == "design")
    ].copy()
    if xgb.empty:
        xgb = shap_global.copy()
    return (
        xgb.sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .assign(rank=lambda d: range(1, len(d) + 1))
        [["rank", "feature", "mean_abs_shap"]]
    )


def section_4_1(doc: Document) -> None:
    heading(doc, "4.1  Dataset and Preprocessing Summary (§3.1-§3.7)", 2)
    audit = read_csv("audit_raw.csv", required=False)
    total_rows = audit.loc[audit["Unnamed: 0"] == "total_rows", "value"].iloc[0] if not audit.empty else "--"
    date_min = audit.loc[audit["Unnamed: 0"] == "date_min", "value"].iloc[0] if not audit.empty else "--"
    date_max = audit.loc[audit["Unnamed: 0"] == "date_max", "value"].iloc[0] if not audit.empty else "--"
    para(
        doc,
        f"The processed LTPP extract contains {total_rows} raw rows spanning {date_min} to {date_max}. "
        "After target-specific filtering, thresholding, feature screening, and section-based splitting, "
        "the design datasets remain small and regionally imbalanced. This matters because the article's "
        "central question is not only whether ensembles fit pooled data, but whether their fitted "
        "relationships survive climate and regional transfer."
    )
    add_table(
        doc,
        "Table 4.1. Target-specific rows retained after preprocessing and splitting.",
        target_counts(),
        ["Target", "Feature rows", "Visit rows", "Train rows", "Validation rows", "Test rows"],
        digits=0,
    )
    add_table(
        doc,
        "Table 4.2. Section-level train/validation/test allocation by region.",
        split_summary(),
        ["region", "train", "val", "test", "total_sections"],
        labels={"region": "Region", "train": "Train", "val": "Validation", "test": "Test", "total_sections": "Total"},
        digits=0,
    )


def section_4_2(doc: Document, test: pd.DataFrame, common: pd.DataFrame, loo: pd.DataFrame, shap: pd.DataFrame) -> None:
    heading(doc, "4.2  Pre-Specified Consistency Checks (§3.13)", 2)
    para(
        doc,
        "The repaired pipeline is evaluated with explicit consistency checks rather than with the most "
        "optimistic model alone. These checks target the exact failure modes identified in the analysis: "
        "same-visit leakage, persistence dominance, climate transfer failure, and unstable explanations."
    )
    add_table(
        doc,
        "Table 4.3. Corrected consistency checks.",
        consistency_checks(test, common, loo, shap),
        ["Check", "Evidence", "Interpretation"],
    )


def section_4_3(doc: Document, bench: pd.DataFrame) -> None:
    heading(doc, "4.3  Benchmark Model Performance (§3.9)", 2)
    para(
        doc,
        "The benchmark models define the minimum bar for the machine-learning models. Mean and age-only "
        "baselines test whether a target can be explained by simple central tendency or time-in-service, "
        "while Ridge tests whether a linear regularized model already captures the useful signal. "
        "Persistence benchmarks are reported separately because they are monitoring-task baselines."
    )
    add_table(
        doc,
        "Table 4.4. Benchmark performance on held-out test rows.",
        benchmark_view(bench),
        ["model", "n", "R2", "RMSE", "MAE"],
        labels={"model": "Benchmark", "n": "N"},
    )


def section_4_4(doc: Document, test: pd.DataFrame) -> None:
    heading(doc, "4.4  Ensemble Model Test-Set Performance (§3.10-§3.11.2)", 2)
    design_distress = first_metric(test, "xgb_design_distress", "R2")
    design_iri = first_metric(test, "xgb_design_iri", "R2")
    monitoring_iri = first_metric(test, "xgb_monitoring_iri", "R2")
    para(
        doc,
        f"After removing same-visit condition features and geographic proxies, ensemble performance is "
        f"much less inflated. XGBoost design IRI reaches R2 = {fmt(design_iri)}, while XGBoost design "
        f"distress reaches R2 = {fmt(design_distress)} rather than the near-perfect result in the old "
        f"document. XGBoost monitoring IRI reaches R2 = {fmt(monitoring_iri)}, but this is expected to "
        "benefit from lagged condition information and must be judged against persistence."
    )
    add_table(
        doc,
        "Table 4.5. Corrected ensemble test-set performance with section-bootstrap intervals.",
        test_view(test),
        ["Target", "Task", "Model", "n_obs", "n_sections", "R2", "R2_CI_lower", "R2_CI_upper", "RMSE", "MAE"],
        labels={
            "n_obs": "Rows",
            "n_sections": "Sections",
            "R2_CI_lower": "R2 CI lower",
            "R2_CI_upper": "R2 CI upper",
        },
    )
    heading(doc, "4.4.1  Regime-Split Rutting Models", 3)
    para(
        doc,
        "The old chapter treated warm/freeze rutting splits as a fault-specific repair. The corrected "
        "pipeline no longer elevates that post-hoc split to a primary result because the article's main "
        "methodology emphasizes pre-specified target, task, and regional-transfer evaluations. Any future "
        "regime split should be reintroduced only as a pre-registered sensitivity analysis with held-out "
        "section counts and uncertainty intervals."
    )


def section_4_5(doc: Document, common: pd.DataFrame) -> None:
    heading(doc, "4.5  Monitoring Task vs. Persistence Benchmarks (§3.9 Common Set)", 2)
    xgb = first_metric(common, "xgb_monitoring_iri_common", "skill_vs_persist_k1")
    delta = first_metric(common, "xgb_monitoring_delta_iri_common", "skill_vs_persist_k1")
    para(
        doc,
        "Monitoring models are compared with persistence on the same section/date rows. This prevents an "
        "optimistic comparison where a trained model and a naive baseline are evaluated on different data. "
        f"On the common IRI set, absolute XGBoost monitoring has skill_vs_persist_k1 = {fmt(xgb)}, while "
        f"the delta-IRI sensitivity model has skill_vs_persist_k1 = {fmt(delta)}."
    )
    add_table(
        doc,
        "Table 4.6. Common-set monitoring and persistence comparison.",
        common,
        ["model", "n", "R2", "RMSE", "MAE", "skill_vs_persist_k1"],
        labels={"model": "Model", "n": "N", "skill_vs_persist_k1": "Skill vs k=1 persistence"},
    )
    heading(doc, "4.5.1  Delta IRI Monitoring - Forecast Skill over Persistence", 3)
    para(
        doc,
        "The delta formulation predicts deterioration increments and then maps the prediction back to the "
        "absolute IRI scale using the lagged condition. This is a useful sensitivity because it separates "
        "forecasting deterioration from simply reconstructing the lag. It should still be interpreted as a "
        "monitoring result, not a design-stage transfer result."
    )


def section_4_6(doc: Document, loo: pd.DataFrame, ontario: pd.DataFrame) -> None:
    heading(doc, "4.6  Leave-One-Region-Out Generalisation (§3.11.3)", 2)
    ont_xgb = loo[(loo["withheld_region"] == "Ontario") & (loo["arch"] == "xgb")]
    georgia_xgb = loo[(loo["withheld_region"] == "Georgia") & (loo["arch"] == "xgb")]
    ont_r2 = None if ont_xgb.empty else float(ont_xgb.iloc[0]["R2"])
    ga_r2 = None if georgia_xgb.empty else float(georgia_xgb.iloc[0]["R2"])
    para(
        doc,
        f"Leave-one-region-out validation is the clearest stress test for the article's transferability "
        f"argument. The corrected Ontario XGBoost LOO R2 is {fmt(ont_r2)}, and Georgia XGBoost LOO R2 is "
        f"{fmt(ga_r2)}. These failures are not cosmetic; they show that pooled accuracy does not guarantee "
        "transportable pavement-deterioration relationships."
    )
    add_table(
        doc,
        "Table 4.7. Leave-one-region-out IRI design validation.",
        loo,
        ["withheld_region", "arch", "target", "n_obs", "R2", "RMSE", "MAE"],
        labels={"withheld_region": "Withheld region", "arch": "Model", "target": "Target", "n_obs": "Rows"},
    )
    add_figure(doc, "loo_vs_climate_gradient.png", "Figure 4.1. LOO R2 across the climate gradient.")
    add_figure(doc, "loo_scatter_4panel.png", "Figure 4.2. Observed versus predicted IRI under LOO validation.")
    heading(doc, "4.6.1  Ontario Section-Level Sensitivity (§3.11.3)", 3)
    para(
        doc,
        "Ontario section-level results are retained as exploratory diagnostics only. Most sections have "
        "fewer than 30 observations, so these rows should identify failure patterns rather than support "
        "strong inferential claims."
    )
    add_table(
        doc,
        "Table 4.8. Ontario section-level LOO sensitivity.",
        ontario,
        ["section_key", "fi_group", "mean_fi", "n_obs", "arch", "R2", "note"],
        labels={
            "section_key": "Section",
            "fi_group": "Freeze-index group",
            "mean_fi": "Mean FI",
            "n_obs": "Rows",
            "arch": "Model",
            "note": "Note",
        },
    )
    add_figure(doc, "ontario_sensitivity_scatter.png", "Figure 4.3. Ontario section-level LOO sensitivity.")


def section_4_7(doc: Document) -> None:
    heading(doc, "4.7  Residual Diagnostics (§3.11.4)", 2)
    para(
        doc,
        "Residual diagnostics were regenerated after the leakage and bootstrap corrections. They are used "
        "to assess heteroscedasticity, regional bias, and systematic misspecification in the corrected "
        "XGBoost IRI design model. The figure should be read alongside the weak LOO results: residual "
        "structure is another warning that pooled test performance is not enough."
    )
    add_figure(doc, "residual_diagnostics.png", "Figure 4.4. Residual diagnostics for corrected XGBoost IRI design predictions.")


def section_4_8(doc: Document, shap_consistency: pd.DataFrame, shap_global: pd.DataFrame) -> None:
    heading(doc, "4.8  SHAP TreeExplainer (§3.12.1)", 2)
    rho = None if shap_consistency.empty else float(shap_consistency.iloc[0]["spearman_rho"])
    threshold = None if shap_consistency.empty else float(shap_consistency.iloc[0]["threshold"])
    para(
        doc,
        f"SHAP explanations were regenerated for the corrected design model. The cross-model rank "
        f"correlation is rho = {fmt(rho)}, below the pre-specified threshold of {fmt(threshold)}. "
        "Consequently, SHAP is reported as model-behavior evidence and not as causal attribution."
    )
    heading(doc, "4.8.1  Global Feature Importance", 3)
    add_table(
        doc,
        "Table 4.9. Top corrected XGBoost IRI design SHAP features.",
        top_shap_features(shap_global),
        ["rank", "feature", "mean_abs_shap"],
        labels={"rank": "Rank", "feature": "Feature", "mean_abs_shap": "Mean |SHAP|"},
    )
    add_figure(doc, "shap_bar_iri_design.png", "Figure 4.5. Global SHAP bar plot for corrected XGBoost IRI design.", width=5.5)
    add_figure(doc, "beeswarm_iri_design.png", "Figure 4.6. SHAP beeswarm for corrected XGBoost IRI design.", width=5.8)
    heading(doc, "4.8.2  Waterfall Decomposition by Region (§3.12.1)", 3)
    para(
        doc,
        "The waterfall plots illustrate individual regional predictions. They are useful for auditing how "
        "a model assembles one prediction, but they should not be generalized beyond the local examples."
    )
    add_figure(doc, "waterfall_4panel.png", "Figure 4.7. Regional SHAP waterfall examples.")
    heading(doc, "4.8.3  Regional SHAP Stratification (§3.12.1)", 3)
    para(
        doc,
        "Regional SHAP summaries expose changes in model reliance across climates and agencies. These "
        "patterns are descriptive and must be reconciled with the poor LOO transfer before being used in "
        "engineering interpretation."
    )
    add_figure(doc, "regional_shap.png", "Figure 4.8. Regional SHAP summary.")
    add_figure(doc, "shap_category_by_region.png", "Figure 4.9. SHAP category contributions by region.")
    add_figure(doc, "regional_top_features.png", "Figure 4.10. Region-specific top SHAP features.")
    heading(doc, "4.8.4  Cross-Model SHAP Consistency (§3.12.2)", 3)
    add_table(
        doc,
        "Table 4.10. Cross-model SHAP rank consistency.",
        shap_consistency,
        ["n_features", "spearman_rho", "p_value", "threshold", "status"],
        labels={
            "n_features": "Features",
            "spearman_rho": "Spearman rho",
            "p_value": "p-value",
            "threshold": "Threshold",
            "status": "Status",
        },
    )


def section_4_9(doc: Document) -> None:
    heading(doc, "4.9  Partial Dependence Analysis (§3.12.3)", 2)
    para(
        doc,
        "Partial dependence plots summarize fitted response surfaces for selected predictors. Because the "
        "model does not transfer reliably across regions, these plots are interpreted as internal model "
        "diagnostics rather than universal pavement-performance curves."
    )
    add_figure(doc, "pdp_oneway.png", "Figure 4.11. One-way partial dependence diagnostics.", width=6.2)
    add_figure(doc, "pdp_twoway.png", "Figure 4.12. Two-way partial dependence diagnostics.", width=6.2)


def section_4_10(doc: Document) -> None:
    heading(doc, "4.10  Regional Climate, Traffic, and Deterioration Profile", 2)
    para(
        doc,
        "The regional profile contextualizes why a pooled model may fail under LOO validation. Differences "
        "in freeze index, precipitation, traffic loading, and deterioration distributions create a transfer "
        "problem that cannot be solved by in-sample explanation alone."
    )
    add_figure(doc, "regional_climate_traffic_profile.png", "Figure 4.13. Regional climate, traffic, and deterioration profile.")


def section_4_11(doc: Document) -> None:
    heading(doc, "4.11  Synthesis: Limits of Explainable ML for Pavement Prediction (§3.14)", 2)
    para(
        doc,
        "The corrected results support the article's main argument more clearly than the old, overfit "
        "chapter. Apparent accuracy can be produced by same-visit condition variables, regional proxies, "
        "lag persistence, and bootstrap procedures that do not respect section dependence. Once those "
        "shortcuts are removed, design-stage prediction is modest, regional transfer is weak, monitoring "
        "must be benchmarked against persistence, and explanations become conditional on model validity."
    )
    bullet(doc, "Report design-stage and monitoring tasks separately; do not use monitoring accuracy as evidence of design transfer.")
    bullet(doc, "Keep persistence on the same section/date rows as monitoring models whenever forecast skill is claimed.")
    bullet(doc, "Treat SHAP and PDP as diagnostics unless the model passes transfer and stability checks.")
    bullet(doc, "Frame negative LOO results as substantive evidence about limited generalizability, not as a failed side experiment.")


def build_doc() -> Document:
    bench = read_csv("benchmark_metrics.csv")
    test = read_csv("test_metrics.csv")
    common = read_csv("common_set_comparison.csv")
    loo = read_csv("loo_summary.csv")
    ontario = read_csv("ontario_section_loo.csv", required=False)
    shap_consistency = read_csv("shap_consistency.csv")
    shap_global = read_csv("shap_global.csv", required=False)

    doc = setup_document()
    heading(doc, "4. Results and Discussion", 1)
    para(
        doc,
        "This section reports the corrected results after repairing the pipeline issues identified in "
        "analysis.md: same-visit condition leakage was removed, geographic and climate-zone proxy features "
        "were excluded from the primary design models, section-level bootstrapping was corrected, and "
        "monitoring-vs-persistence comparisons were recomputed on identical rows. The findings therefore "
        "emphasize transferability, benchmark skill, and explanation stability rather than optimistic fit."
    )

    section_4_1(doc)
    section_4_2(doc, test, common, loo, shap_consistency)
    section_4_3(doc, bench)
    section_4_4(doc, test)
    section_4_5(doc, common)
    section_4_6(doc, loo, ontario)
    section_4_7(doc)
    section_4_8(doc, shap_consistency, shap_global)
    section_4_9(doc)
    section_4_10(doc)
    section_4_11(doc)

    return doc


def main() -> None:
    doc = build_doc()
    doc.save(OUT)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
