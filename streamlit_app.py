"""Streamlit interface for the Spectra AI anomaly detection stack."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

st.set_page_config(page_title="Spectra AI Prompt Anomaly Guard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bayesian_analysis import BayesianAnomalyAnalyzer  # noqa: E402
from mahalanobis_detector import MahalanobisDetector  # noqa: E402
from text_detectors import detect_all_text_anomalies  # noqa: E402

MODELS_DIR = BASE_DIR / "models"
MODEL_LABELS = {
    "one_class": "One-Class Mahalanobis",
    "two_class": "Two-Class Mahalanobis",
    "text": "Text Pattern Detectors",
    "ensemble": "Hybrid Ensemble",
}


@st.cache_resource(show_spinner=False)
def load_sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_mahalanobis_detectors() -> Dict[str, Any]:
    files = {
        "one_class": MODELS_DIR / "mahalanobis_detector_oneclass.pkl",
        "two_class_normal": MODELS_DIR / "mahalanobis_detector_normal.pkl",
        "two_class_anomalous": MODELS_DIR / "mahalanobis_detector_anomalous.pkl",
    }
    detectors: Dict[str, Any] = {}
    for key, path in files.items():
        if path.exists():
            with path.open("rb") as fh:
                detectors[key] = pickle.load(fh)
        else:
            detectors[key] = None
    return detectors


@st.cache_resource(show_spinner=False)
def load_model_metadata() -> Dict[str, Any]:
    path = MODELS_DIR / "model_info.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    return {}


def compute_embedding(prompt: str, model) -> np.ndarray:
    return model.encode(prompt, convert_to_numpy=True)


def evaluate_one_class(embedding: np.ndarray, detectors: Dict[str, Any]) -> Dict[str, Any]:
    detector: MahalanobisDetector | None = detectors.get("one_class")
    if detector is None:
        raise RuntimeError("One-class Mahalanobis model is not available.")

    embedding_matrix = embedding.reshape(1, -1)
    distance = float(detector.compute_mahalanobis_distance(embedding))
    probability = float(detector.compute_chi2_probabilities(embedding_matrix)[0])
    threshold_sq = detector.chi2_threshold if detector.chi2_threshold is not None else None
    threshold = float(np.sqrt(threshold_sq)) if threshold_sq else None
    flagged = bool(detector.predict(embedding_matrix)[0])

    metrics = {
        "Mahalanobis distance": f"{distance:.3f}",
        "Chi-square probability": f"{probability:.4f}",
    }
    if threshold is not None:
        metrics["Distance threshold"] = f"{threshold:.3f}"

    return {
        "flagged": flagged,
        "metrics": metrics,
        "raw": {
            "distance": distance,
            "probability": probability,
            "threshold_distance": threshold,
            "significance_level": detector.significance_level,
        },
    }


def evaluate_two_class(embedding: np.ndarray, detectors: Dict[str, Any]) -> Dict[str, Any]:
    normal_detector: MahalanobisDetector | None = detectors.get("two_class_normal")
    anomalous_detector: MahalanobisDetector | None = detectors.get("two_class_anomalous")
    if normal_detector is None or anomalous_detector is None:
        raise RuntimeError("Two-class Mahalanobis models are not available.")

    distance_normal = float(normal_detector.compute_mahalanobis_distance(embedding))
    distance_anomalous = float(anomalous_detector.compute_mahalanobis_distance(embedding))
    margin = distance_normal - distance_anomalous
    flagged = distance_anomalous < distance_normal

    metrics = {
        "Distance to normal": f"{distance_normal:.3f}",
        "Distance to anomalous": f"{distance_anomalous:.3f}",
        "Decision margin": f"{margin:.3f}",
    }

    return {
        "flagged": flagged,
        "metrics": metrics,
        "raw": {
            "distance_normal": distance_normal,
            "distance_anomalous": distance_anomalous,
            "margin": margin,
        },
    }


def evaluate_text(prompt: str, sentence_model) -> Dict[str, Any]:
    analysis = detect_all_text_anomalies(prompt, model=sentence_model)
    metrics = {
        "Triggered rules": str(int(analysis.get("rules_and_roleplay", False))),
        "Keyword hits": str(len(analysis.get("keywords_found", []))),
        "Pattern hits": str(len(analysis.get("patterns_found", []))),
        "Anomaly score": str(analysis.get("anomaly_score", 0)),
    }

    return {
        "flagged": bool(analysis.get("is_anomalous", False)),
        "metrics": metrics,
        "raw": analysis,
    }


def evaluate_ensemble(
    embedding: np.ndarray,
    prompt: str,
    detectors: Dict[str, Any],
    sentence_model,
) -> Dict[str, Any]:
    two_class = evaluate_two_class(embedding, detectors)
    text_based = evaluate_text(prompt, sentence_model)
    flagged = two_class["flagged"] or text_based["flagged"]

    metrics = {
        "Two-class vote": "Anomalous" if two_class["flagged"] else "Normal",
        "Text detector vote": "Anomalous" if text_based["flagged"] else "Normal",
        "Decision": "Anomalous" if flagged else "Normal",
    }

    return {
        "flagged": flagged,
        "metrics": metrics,
        "raw": {
            "two_class": two_class,
            "text": text_based,
        },
    }


def build_result(
    prompt: str,
    model_choice: str,
    detectors: Dict[str, Any],
    sentence_model,
) -> Dict[str, Any]:
    embedding = compute_embedding(prompt, sentence_model)
    if model_choice == MODEL_LABELS["one_class"]:
        result = evaluate_one_class(embedding, detectors)
    elif model_choice == MODEL_LABELS["two_class"]:
        result = evaluate_two_class(embedding, detectors)
    elif model_choice == MODEL_LABELS["text"]:
        result = evaluate_text(prompt, sentence_model)
    else:
        result = evaluate_ensemble(embedding, prompt, detectors, sentence_model)

    result["model_choice"] = model_choice
    return result


def compute_bayesian(flagged: bool, prior: float, tpr: float, fpr: float) -> Tuple[float, Dict[str, Any]]:
    analyzer = BayesianAnomalyAnalyzer(
        prior_anomaly_rate=prior,
        true_positive_rate=tpr,
        false_positive_rate=fpr,
    )
    posterior_if_flagged = analyzer.compute_posterior_anomaly(flagged=True)
    posterior_if_not_flagged = analyzer.compute_posterior_anomaly(flagged=False)
    posterior = posterior_if_flagged if flagged else posterior_if_not_flagged
    summary = analyzer.get_summary()
    summary["posterior_selected"] = posterior
    summary["posterior_if_not_flagged"] = posterior_if_not_flagged
    return posterior, summary


def render_sidebar(metadata: Dict[str, Any]) -> Tuple[str, float, float, float]:
    st.sidebar.header("Detection Settings")

    detectors = load_mahalanobis_detectors()
    options = []
    if detectors.get("one_class") is not None:
        options.append(MODEL_LABELS["one_class"])
    if detectors.get("two_class_normal") is not None and detectors.get("two_class_anomalous") is not None:
        options.append(MODEL_LABELS["two_class"])
    options.append(MODEL_LABELS["text"])
    if MODEL_LABELS["two_class"] in options:
        options.append(MODEL_LABELS["ensemble"])

    if not options:
        st.sidebar.error("No detector artifacts were found in models/.")
        st.stop()

    default_index = options.index(MODEL_LABELS["ensemble"]) if MODEL_LABELS["ensemble"] in options else 0
    model_choice = st.sidebar.radio("Model selection", options, index=default_index)

    prior = st.sidebar.slider("Prior anomaly rate", 0.01, 0.50, value=0.10, step=0.01)
    tpr = st.sidebar.slider("Assumed detector TPR", 0.50, 1.00, value=0.95, step=0.01)
    fpr = st.sidebar.slider("Assumed detector FPR", 0.00, 0.30, value=0.05, step=0.01)

    if metadata:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Model snapshot**")
        st.sidebar.markdown(f"Embeddings: {metadata.get('embedding_model_name', 'n/a')}")
        st.sidebar.markdown(
            f"Trained samples: normal={metadata.get('trained_on_normal_samples', 'n/a')}, "
            f"anomalous={metadata.get('trained_on_anomalous_samples', 'n/a')}"
        )
        st.sidebar.markdown(f"One-class accuracy: {metadata.get('oneclass_accuracy', 'n/a')}")
        st.sidebar.markdown(f"Two-class accuracy: {metadata.get('twoclass_accuracy', 'n/a')}")
        st.sidebar.markdown(f"Training date: {metadata.get('training_date', 'n/a')}")

    return model_choice, prior, tpr, fpr


def render_result(result: Dict[str, Any], bayes_summary: Dict[str, Any]) -> None:
    flagged = result["flagged"]
    headline = "Anomalous" if flagged else "Normal"
    if flagged:
        st.error(f"Detection Verdict: {headline}")
    else:
        st.success(f"Detection Verdict: {headline}")

    posterior = bayes_summary.get("posterior_selected", 0.0)
    posterior_if_flagged = bayes_summary.get("posterior_if_flagged", posterior)
    posterior_if_not_flagged = bayes_summary.get("posterior_if_not_flagged", posterior)

    cols = st.columns(3)
    cols[0].metric("Posterior P(anomaly)", f"{posterior*100:.1f}%")
    cols[1].metric("Assumed TPR", f"{bayes_summary.get('true_positive_rate', 0.0)*100:.0f}%")
    cols[2].metric("Assumed FPR", f"{bayes_summary.get('false_positive_rate', 0.0)*100:.0f}%")

    st.progress(min(max(posterior, 0.0), 1.0))

    st.markdown("### Model Metrics")
    metric_cols = st.columns(len(result["metrics"]) or 1)
    for col, (label, value) in zip(metric_cols, result["metrics"].items()):
        col.metric(label, value)

    with st.expander("Bayesian details"):
        bayes_table = {
            "Posterior if flagged": f"{posterior_if_flagged*100:.1f}%",
            "Posterior if not flagged": f"{posterior_if_not_flagged*100:.1f}%",
            "Likelihood ratio": f"{bayes_summary.get('likelihood_ratio', 0.0):.2f}",
            "Bayes factor": f"{bayes_summary.get('bayes_factor', 0.0):.2f}",
            "Interpretation": bayes_summary.get('interpretation', 'n/a'),
        }
        st.table([[k, v] for k, v in bayes_table.items()])

    raw = result.get("raw", {})
    if result["model_choice"] == MODEL_LABELS["text"] and raw:
        with st.expander("Text detector signals"):
            st.write({k: v for k, v in raw.items() if k not in ("anomaly_score", "is_anomalous")})
    elif result["model_choice"] == MODEL_LABELS["ensemble"] and raw:
        with st.expander("Component outputs"):
            st.write({
                "Two-class": raw.get("two_class", {}).get("metrics", {}),
                "Text": raw.get("text", {}).get("metrics", {}),
            })


def main() -> None:
    st.markdown("# Spectra AI Prompt Anomaly Guard")
    st.markdown(
        "This console allows analysts to vet prompts against the Mahalanobis-based "
        "and rule-driven detectors refined in the Spectra AI anomaly detection notebook."
    )

    metadata = load_model_metadata()
    model_choice, prior, tpr, fpr = render_sidebar(metadata)

    sentence_model = load_sentence_model()
    detectors = load_mahalanobis_detectors()

    with st.form("prompt_form"):
        prompt = st.text_area(
            "Enter the prompt to evaluate",
            height=200,
            placeholder="Paste or type the prompt you wish to audit...",
        )
        submitted = st.form_submit_button("Analyze Prompt")

    if submitted:
        if not prompt.strip():
            st.warning("Please provide a prompt before running detection.")
            return
        with st.spinner("Running detectors..."):
            try:
                result = build_result(prompt, model_choice, detectors, sentence_model)
            except RuntimeError as err:
                st.error(str(err))
                return
        posterior, bayes_summary = compute_bayesian(result["flagged"], prior, tpr, fpr)
        bayes_summary["posterior_selected"] = posterior
        render_result(result, bayes_summary)

    st.markdown("---")
    st.markdown(
        "Need batch scoring or retraining? Refer to the notebook in `notebooks/` for full data pipelines."
    )


if __name__ == "__main__":
    main()
