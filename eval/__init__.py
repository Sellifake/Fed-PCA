"""
Evaluation Module for Fed-ProFiLA-AD
"""

from .inference import (
    evaluate_client,
    evaluate_all_clients,
    compute_anomaly_scores_batch,
    predict_anomaly,
    evaluate_with_different_thresholds,
    save_evaluation_results,
    load_evaluation_results
)
from .metrics import (
    calculate_auc,
    calculate_pr_auc,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    find_best_threshold,
    calculate_confusion_matrix,
    calculate_all_metrics,
    print_metrics,
    calculate_average_metrics
)

__all__ = [
    'evaluate_client',
    'evaluate_all_clients',
    'compute_anomaly_scores_batch',
    'predict_anomaly',
    'evaluate_with_different_thresholds',
    'save_evaluation_results',
    'load_evaluation_results',
    'calculate_auc',
    'calculate_pr_auc',
    'calculate_f1_score',
    'calculate_precision',
    'calculate_recall',
    'find_best_threshold',
    'calculate_confusion_matrix',
    'calculate_all_metrics',
    'print_metrics',
    'calculate_average_metrics'
]
