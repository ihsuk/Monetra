"""
utils/frontend_patch.py — Patch the Monetra HTML to call the real backend
==========================================================================
Run this script once to inject the API integration layer into monetra.html.
It adds a thin JS adapter that replaces the static runPrediction() function
with one that actually POSTs to the FastAPI backend.

Usage:
  python utils/frontend_patch.py path/to/monetra.html

Creates monetra_live.html in the same directory.
"""

import sys
import re
import os

BACKEND_URL = "https://monetra-backend-nngy.onrender.com"

# JavaScript to inject — replaces / wraps key frontend functions
JS_INJECT = f"""
<script>
// ── Monetra Live API Integration ─────────────────────────────────────────────
const API = '{BACKEND_URL}';

// Override runPrediction() to call real backend
async function runPrediction() {{
  const loan   = +document.getElementById('fi-loan').value   || 500000;
  const income = +document.getElementById('fi-income').value || 800000;
  const credit = +document.getElementById('fi-credit').value || 720;
  const age    = +document.getElementById('fi-age').value    || 32;

  // Show loading state
  const btn = document.querySelector('[onclick="runPrediction()"]');
  if (btn) {{ btn.textContent = '⏳ Analysing…'; btn.disabled = true; }}

  try {{
    const res  = await fetch(API + '/predict', {{
      method:  'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        loan_amount:   loan,
        annual_income: income,
        credit_score:  credit,
        applicant_age: age,
        loan_tenure:   36,
        employment_type: 'salaried'
      }})
    }});
    const data = await res.json();

    const approved = data.prediction === 'APPROVED';
    document.getElementById('pred-res-val').textContent = data.prediction;
    document.getElementById('pred-res-val').style.color = approved ? 'var(--gn)' : 'var(--rd)';
    document.getElementById('pred-conf-val').textContent = data.confidence_pct;
    document.getElementById('pred-conf-bar').style.width = (data.confidence * 100) + '%';
    document.getElementById('pred-conf-bar').style.background = approved ? 'var(--gn)' : 'var(--rd)';

    const rt = document.getElementById('pred-risk-tag');
    const rl = data.risk_level;
    rt.textContent = rl === 'LOW' ? '🟢 LOW RISK' : rl === 'MEDIUM' ? '🟡 MEDIUM RISK' : '🔴 HIGH RISK';
    rt.className   = 'tag ' + (rl === 'LOW' ? 'tg' : rl === 'MEDIUM' ? 'ta' : 'tr');

    document.getElementById('pred-result').classList.add('show');
    console.log('[Monetra] Prediction ID:', data.prediction_id, '| Version:', data.model_version);
  }} catch (err) {{
    console.error('[Monetra] Predict API error:', err);
    alert('Backend unreachable. Is the FastAPI server running on port 8000?');
  }} finally {{
    if (btn) {{ btn.textContent = '▶ Run Prediction'; btn.disabled = false; }}
  }}
}}

// Auto-refresh health badge in sidebar on load
async function refreshHealthBadge() {{
  try {{
    const res  = await fetch(API + '/health');
    const data = await res.json();
    // Update sidebar health indicator if present
    const el = document.querySelector('.sb-foot .hi');
    if (el) el.textContent = data.status.toUpperCase();
  }} catch (_) {{}}
}}

// Poll drift summary every 60 s and update the overview PSI bar
async function pollDrift() {{
  try {{
    const res  = await fetch(API + '/drift?save=false&window=100');
    const data = await res.json();
    console.log('[Monetra] Drift avg PSI:', data.avg_psi, '| Features drifted:', data.features_drifted);
  }} catch (_) {{}}
}}

window.addEventListener('load', () => {{
  refreshHealthBadge();
  setInterval(pollDrift, 60_000);
}});
</script>
"""


def patch(src_path: str):
    with open(src_path, "r", encoding="utf-8") as f:
        html = f.read()

    dst_path = os.path.join(
        os.path.dirname(src_path),
        os.path.splitext(os.path.basename(src_path))[0] + "_live.html"
    )

    # If it is the advanced monetra_final.html, simply update the API constant
    if "let API = 'http://localhost:8000';" in html:
        print("Detected advanced monetra_final.html. Patching API constant dynamically...")
        html = html.replace("let API = 'http://localhost:8000';", f"let API = '{BACKEND_URL}';")
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Patched HTML written to: {dst_path}")
        return

    # Fallback to injecting the JS adapter block for older monetra.html
    if JS_INJECT.strip() in html:
        print("Already patched. Skipping.")
        return

    html = html.replace("</body>", JS_INJECT + "\n</body>")

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Patched HTML written to: {dst_path}")



if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "monetra.html"
    patch(path)
