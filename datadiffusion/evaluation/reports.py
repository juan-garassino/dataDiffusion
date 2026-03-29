"""Human-readable quality report — printed to console and saved to file."""

import os
from .composite import QualityMetrics


def _status_icon(status: str) -> str:
    return {"PASS": "✓ PASS", "WARN": "⚠ WARN", "FAIL": "✗ FAIL"}.get(status, status)


def _section_status(value: float, good: float, bad: float) -> str:
    if value <= good:
        return "✓ PASS"
    elif value <= bad:
        return "⚠ WARN"
    return "✗ FAIL"


def format_quality_report(metrics: QualityMetrics, iteration: int = 0, threshold: float = 0.80) -> str:
    w = 62
    lines = []
    a = lines.append

    a("╔" + "═" * w + "╗")
    a(f"║{'QUALITY REPORT — Iteration ' + str(iteration):^{w}}║")
    a("╠" + "═" * w + "╣")

    status_str = "PASSED ✓" if metrics.composite_score >= threshold else "NEEDS IMPROVEMENT"
    a(f"║  Composite Score: {metrics.composite_score:.2f} / 1.00  (threshold: {threshold:.2f}){' ' * (w - 52)}║")
    a(f"║  Status: {status_str}{' ' * (w - 11 - len(status_str))}║")
    a("╠" + "═" * w + "╣")

    # 1. Distribution match
    ks_status = _section_status(metrics.avg_ks_statistic, 0.15, 0.30)
    a(f"║{' ' * w}║")
    a(f"║  1. DISTRIBUTION MATCH (KS avg: {metrics.avg_ks_statistic:.2f}){' ' * (w - 48)}{ks_status}  ║")

    for fr in metrics.feature_reports:
        icon = _status_icon(fr.status)
        line = f"     {fr.feature_name:12s}  KS={fr.ks_statistic:.2f}  p={fr.ks_pvalue:.3f}"
        padding = w - len(line) - len(icon) - 2
        a(f"║{line}{' ' * max(padding, 1)}{icon}  ║")

    a(f"║{' ' * w}║")

    # 2. Correlation structure
    corr_status = _section_status(metrics.corr_diff_norm, 0.3, 0.5)
    a(f"║  2. CORRELATION STRUCTURE (Frobenius: {metrics.corr_diff_norm:.2f}){' ' * (w - 52)}{corr_status}  ║")
    a(f"║{' ' * w}║")

    # 3. ML utility
    ml_status = _section_status(1.0 - metrics.ml_utility_ratio, 0.2, 0.5)
    a(f"║  3. ML UTILITY (R² ratio: {metrics.ml_utility_ratio:.2f}){' ' * (w - 43)}{ml_status}  ║")
    a(f"║     Real-trained R²:      {metrics.r2_real:.2f}{' ' * (w - 32)}║")
    a(f"║     Synthetic-trained R²: {metrics.r2_synthetic:.2f}{' ' * (w - 32)}║")

    if metrics.r2_real > 0:
        gap_pct = (1.0 - metrics.ml_utility_ratio) * 100
        a(f"║     Gap: synthetic underperforms by {gap_pct:.0f}%{' ' * (w - 42)}║")

    a(f"║{' ' * w}║")

    # Diagnosis
    a("╠" + "═" * w + "╣")
    a(f"║  DIAGNOSIS:{' ' * (w - 13)}║")

    failed = metrics.failed_features
    if failed:
        idx_str = ",".join(str(f.feature_index) for f in failed)
        a(f"║  • Features {idx_str} have high KS → more capacity/epochs{' ' * max(0, w - 53 - len(idx_str))}║")
    if metrics.corr_diff_norm > 0.5:
        a(f"║  • Correlation structure weak → increase model depth{' ' * (w - 54)}║")
    if metrics.ml_utility_ratio < 0.5:
        a(f"║  • ML utility low → more capacity + training{' ' * (w - 48)}║")
    if not failed and metrics.corr_diff_norm <= 0.5 and metrics.ml_utility_ratio >= 0.5:
        a(f"║  • No critical issues detected{' ' * (w - 32)}║")

    a(f"║{' ' * w}║")
    a("╚" + "═" * w + "╝")

    return "\n".join(lines)


def print_quality_report(metrics: QualityMetrics, iteration: int = 0, threshold: float = 0.80):
    print(format_quality_report(metrics, iteration, threshold))


def save_quality_report(metrics: QualityMetrics, save_dir: str, iteration: int = 0, threshold: float = 0.80):
    os.makedirs(save_dir, exist_ok=True)
    report = format_quality_report(metrics, iteration, threshold)
    path = os.path.join(save_dir, "quality_report.txt")
    with open(path, "w") as f:
        f.write(report)
    return path
