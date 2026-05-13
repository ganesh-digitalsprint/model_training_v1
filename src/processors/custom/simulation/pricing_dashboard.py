"""
pricing_dashboard.py
Logs pricing strategy comparison metrics and charts to MLflow.
Replaces: src/simulation/pricing_dashboard.py
"""
import mlflow
import matplotlib.pyplot as plt
from utils.constants import (
    DEFAULT_TRACKING_URI, DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION,
    MLFLOW_EXPERIMENT_PRICING, MLFLOW_ARTIFACT_PRICING_CHART,
    DASHBOARD_LABEL_CURRENT, DASHBOARD_LABEL_RULE, DASHBOARD_LABEL_AI,
    SUMMARY_CURRENT_REVENUE, SUMMARY_NEW_REVENUE, SUMMARY_RULE_REVENUE,
    SUMMARY_AI_REVENUE, SUMMARY_REVENUE_LIFT, SUMMARY_LIFT_PCT,
    SUMMARY_REVENUE_CHANGE, SUMMARY_REVENUE_CHANGE_PCT,
    SUMMARY_CHAMPION_REVENUE, SUMMARY_CHALLENGER_REVENUE,
)


def log_pricing_dashboard(rule_summary, ai_rule_summary, current_ai_summary,
                           cc_summary, model_name=DEFAULT_MODEL_NAME,
                           model_version=DEFAULT_MODEL_VERSION,
                           tracking_uri: str = DEFAULT_TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PRICING)

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=f"pricing_dashboard_{model_name}_{model_version}"):

        cr  = rule_summary.get(SUMMARY_CURRENT_REVENUE, 0)
        rr  = rule_summary.get(SUMMARY_NEW_REVENUE, 0)
        mlflow.log_metrics({"current_revenue": cr, "rule_revenue": rr,
                            "rule_vs_current_change": rr - cr,
                            "rule_vs_current_pct": ((rr - cr) / cr * 100) if cr else 0})

        if ai_rule_summary:
            air = ai_rule_summary.get(SUMMARY_AI_REVENUE, 0)
            mlflow.log_metrics({"ai_revenue": air,
                                "ai_vs_rule_lift": ai_rule_summary.get(SUMMARY_REVENUE_LIFT, 0),
                                "ai_vs_rule_pct":  ai_rule_summary.get(SUMMARY_LIFT_PCT, 0)})

        if current_ai_summary:
            mlflow.log_metrics({
                "ai_vs_current_lift": current_ai_summary.get(SUMMARY_REVENUE_CHANGE, 0),
                "ai_vs_current_pct":  current_ai_summary.get(SUMMARY_REVENUE_CHANGE_PCT, 0)})

        mlflow.log_params({"model_name": model_name, "model_version": model_version})

        labels = [DASHBOARD_LABEL_CURRENT, DASHBOARD_LABEL_RULE, DASHBOARD_LABEL_AI]
        values = [cr, rr, air if ai_rule_summary else 0]
        plt.figure(); plt.bar(labels, values)
        plt.title("Pricing Strategy Comparison"); plt.tight_layout()
        plt.savefig(MLFLOW_ARTIFACT_PRICING_CHART)
        mlflow.log_artifact(MLFLOW_ARTIFACT_PRICING_CHART)

        if cc_summary:
            mlflow.log_metrics({"champion_revenue":    cc_summary.get(SUMMARY_CHAMPION_REVENUE, 0),
                                "challenger_revenue":  cc_summary.get(SUMMARY_CHALLENGER_REVENUE, 0),
                                "challenger_lift":     cc_summary.get(SUMMARY_REVENUE_LIFT, 0),
                                "challenger_lift_pct": cc_summary.get(SUMMARY_LIFT_PCT, 0)})
        print("Pricing dashboard logged to MLflow")