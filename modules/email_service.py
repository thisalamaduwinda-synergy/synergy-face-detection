"""
email_service.py
─────────────────────────────────────────────────────────────
Sends the daily attendance report via SMTP (stdlib only).

Config keys (from config.yaml  →  email: section):
  enabled       bool   – master switch
  smtp_host     str    – e.g. "smtp.gmail.com"
  smtp_port     int    – 587 (STARTTLS) or 465 (SSL)
  use_tls       bool   – STARTTLS on port 587
  sender        str    – From address
  recipients    list   – To addresses
  subject_prefix str   – prepended to the email subject
  attach_excel  bool   – attach .xlsx in addition to .csv

Password comes from the environment variable EMAIL_PASSWORD.
For Gmail: generate an App Password in Google Account → Security.
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import date
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self, cfg: Dict) -> None:
        email_cfg = cfg.get("email", {})
        self.enabled       = bool(email_cfg.get("enabled", False))
        self.smtp_host     = email_cfg.get("smtp_host", "smtp.gmail.com")
        self.smtp_port     = int(email_cfg.get("smtp_port", 587))
        self.use_tls       = bool(email_cfg.get("use_tls", True))
        self.sender        = email_cfg.get("sender", "")
        self.recipients: List[str] = email_cfg.get("recipients", [])
        self.subject_prefix = email_cfg.get("subject_prefix", "Attendance Report")
        self.attach_excel  = bool(email_cfg.get("attach_excel", True))
        self.password      = os.environ.get("EMAIL_PASSWORD", "")

    # ── Public API ──────────────────────────────────────────

    def send_daily_report(
        self,
        target_date: date,
        records: List[Dict],
        absent: List[Dict],
        csv_bytes: bytes,
        excel_bytes: Optional[bytes] = None,
    ) -> bool:
        """
        Send the daily attendance email.
        Returns True on success, False on failure.
        """
        if not self.enabled:
            return False
        if not self.recipients:
            logger.warning("Email enabled but no recipients configured.")
            return False
        if not self.password:
            logger.warning("EMAIL_PASSWORD not set – skipping email.")
            return False

        msg = self._build_message(target_date, records, absent, csv_bytes, excel_bytes)
        return self._send(msg)

    # ── Message building ─────────────────────────────────────

    def _build_message(
        self,
        target_date: date,
        records: List[Dict],
        absent: List[Dict],
        csv_bytes: bytes,
        excel_bytes: Optional[bytes],
    ) -> MIMEMultipart:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"{self.subject_prefix} – {target_date}"
        msg["From"]    = self.sender
        msg["To"]      = ", ".join(self.recipients)

        # HTML body
        html = self._build_html(target_date, records, absent)
        msg.attach(MIMEText(html, "html", "utf-8"))

        # CSV attachment
        csv_part = MIMEApplication(csv_bytes, Name=f"attendance_{target_date}.csv")
        csv_part["Content-Disposition"] = f'attachment; filename="attendance_{target_date}.csv"'
        msg.attach(csv_part)

        # Excel attachment (optional)
        if self.attach_excel and excel_bytes:
            xl_part = MIMEApplication(
                excel_bytes,
                Name=f"attendance_{target_date}.xlsx",
            )
            xl_part["Content-Disposition"] = (
                f'attachment; filename="attendance_{target_date}.xlsx"'
            )
            msg.attach(xl_part)

        return msg

    def _build_html(
        self,
        target_date: date,
        records: List[Dict],
        absent: List[Dict],
    ) -> str:
        present_count = len(records)
        absent_count  = len(absent)
        total         = present_count + absent_count
        rate          = f"{present_count / total * 100:.1f}%" if total else "–"

        late_count = sum(1 for r in records if r.get("is_late"))

        absent_rows = "".join(
            f"<tr><td>{a.get('employee_id','')}</td>"
            f"<td>{a.get('name','')}</td>"
            f"<td>{a.get('department','')}</td></tr>"
            for a in absent
        ) or "<tr><td colspan='3' style='color:#16a34a'>All employees present ✓</td></tr>"

        present_rows = "".join(
            f"<tr>"
            f"<td>{r.get('employee_id','')}</td>"
            f"<td>{r.get('employee_name','')}</td>"
            f"<td>{r.get('department','')}</td>"
            f"<td>{(r.get('first_seen') or '')[:19].replace('T',' ')}</td>"
            f"<td>{(r.get('last_seen')  or '')[:19].replace('T',' ')}</td>"
            f"<td>{'Late' if r.get('is_late') else 'On Time'}</td>"
            f"</tr>"
            for r in records
        )

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; color: #1e293b; background: #f8fafc; margin:0; padding:20px; }}
  .card {{ background:#fff; border-radius:12px; padding:28px; max-width:720px; margin:0 auto;
           box-shadow:0 2px 12px rgba(0,0,0,.08); }}
  h2   {{ color:#1e40af; margin:0 0 4px; }}
  .sub {{ color:#64748b; font-size:13px; margin-bottom:20px; }}
  .stats {{ display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }}
  .stat {{ background:#f1f5f9; border-radius:8px; padding:14px 20px; text-align:center; min-width:100px; }}
  .stat-n {{ font-size:28px; font-weight:800; color:#1e40af; }}
  .stat-l {{ font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.5px; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-bottom:24px; }}
  th {{ background:#1e40af; color:#fff; padding:9px 12px; text-align:left; font-size:11px;
        text-transform:uppercase; letter-spacing:.4px; }}
  td {{ padding:8px 12px; border-bottom:1px solid #e2e8f0; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:nth-child(even) {{ background:#f8fafc; }}
  .late {{ color:#dc2626; font-weight:600; }}
  .ontime {{ color:#16a34a; font-weight:600; }}
  h3 {{ color:#334155; font-size:14px; margin:0 0 8px; }}
</style></head>
<body><div class="card">
  <h2>Daily Attendance Report</h2>
  <div class="sub">{target_date.strftime("%A, %d %B %Y")}</div>

  <div class="stats">
    <div class="stat"><div class="stat-n">{present_count}</div><div class="stat-l">Present</div></div>
    <div class="stat"><div class="stat-n" style="color:#dc2626">{absent_count}</div><div class="stat-l">Absent</div></div>
    <div class="stat"><div class="stat-n" style="color:#d97706">{late_count}</div><div class="stat-l">Late</div></div>
    <div class="stat"><div class="stat-n">{rate}</div><div class="stat-l">Rate</div></div>
  </div>

  <h3>Absent Employees ({absent_count})</h3>
  <table>
    <thead><tr><th>ID</th><th>Name</th><th>Department</th></tr></thead>
    <tbody>{absent_rows}</tbody>
  </table>

  <h3>Present Employees ({present_count})</h3>
  <table>
    <thead><tr><th>ID</th><th>Name</th><th>Department</th><th>First Seen</th><th>Last Seen</th><th>Status</th></tr></thead>
    <tbody>{present_rows}</tbody>
  </table>

  <p style="font-size:11px;color:#94a3b8;text-align:center;margin-top:8px">
    Generated by Employee Face Recognition System
  </p>
</div></body></html>"""

    # ── SMTP send ────────────────────────────────────────────

    def _send(self, msg: MIMEMultipart) -> bool:
        try:
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30)
                server.ehlo()
                server.starttls()
                server.ehlo()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30)
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.recipients, msg.as_bytes())
            server.quit()
            logger.info(
                "Attendance email sent to %s", ", ".join(self.recipients)
            )
            return True
        except Exception as exc:
            logger.error("Failed to send attendance email: %s", exc)
            return False
