"""
Export Helpers Module
Utilities to export attendance logs, engagement metrics, and session intelligence reports.
"""

import os
from datetime import datetime
import pandas as pd

def export_attendance_csv(attendance_df, output_path=None):
    """Saves attendance DataFrame to CSV."""
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/sessions", exist_ok=True)
        output_path = f"data/sessions/attendance_{ts}.csv"
    attendance_df.to_csv(output_path, index=False)
    return output_path

def export_timeline_csv(timeline_df, output_path=None):
    """Saves granular multimodal timeline DataFrame to CSV."""
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/sessions", exist_ok=True)
        output_path = f"data/sessions/timeline_{ts}.csv"
    timeline_df.to_csv(output_path, index=False)
    return output_path

def save_session_report(report_markdown, output_path=None):
    """Saves GenAI session report to a markdown file."""
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/sessions", exist_ok=True)
        output_path = f"data/sessions/report_{ts}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)
    return output_path
