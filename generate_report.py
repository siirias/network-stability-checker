#!/usr/bin/env python3
"""
generate_report.py — Build HTML report from network monitor data.

Runs:
  - make_plots.py
  - net_stats.py

Outputs:
  report/report.html
"""

import subprocess
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
REPORT = BASE / "report"

PLOTS_SCRIPT = BASE / "make_plots.py"
STATS_SCRIPT = BASE / "net_stats.py"

TARGET_DESCRIPTIONS = {
    "172.25.208.1": "Local router/gateway — verifies Wi-Fi and local network stability.",
    "193.166.4.1": "ISP-side server — tests the provider’s access network performance.",
    "1.1.1.1": "Cloudflare public DNS — tests general internet connectivity and routing.",
    "8.8.8.8": "Google public DNS — second independent global endpoint for reliability comparison.",
}

def run_plots():
    print("Generating plots...")
    subprocess.run(["python", str(PLOTS_SCRIPT)], check=True)


def run_stats():
    print("Collecting statistics...")
    result = subprocess.run(
        ["python", str(STATS_SCRIPT)],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout



def build_html(stats_text: str):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    target_lines = "<ul>"
    for i in TARGET_DESCRIPTIONS:
        target_lines += f"<li><b>{i}</b> — {TARGET_DESCRIPTIONS[i]}</li>\n"
    target_lines +="</ul>\n"
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Internet reliability report</title>

<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 900px;
    margin: auto;
    padding: 20px;
    background: #fafafa;
}}

h1, h2 {{
    color: #333;
}}

pre {{
    background: #f0f0f0;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
}}

img {{
    width: 100%;
    margin: 20px 0;
    border: 1px solid #ddd;
}}

.section {{
    margin-bottom: 40px;
}}

.small {{
    color: #666;
    font-size: 0.9em;
}}
</style>
</head>

<body>

<h1>Internet Reliability Report</h1>
<p class="small">Generated {now}</p>

<div class="section">
<h2>Summary</h2>
<p>
This report summarizes measured network reliability based on continuous ping monitoring.
Multiple external targets were tested once per second.
Servers used for this test:
<p>
{target_lines}
</p>
Local router pings are analysed separately to distinguish
Wi-Fi/router issues from upstream ISP connectivity problems.
Loss above ~20–30% typically renders interactive applications unusable.

</p>
<pre>{stats_text}</pre>
</div>

<div class="section">
<h2>Latency and Packet Loss Timeline</h2>
<img src="timeseries_rtt_loss.png">
</div>

<div class="section">
<h2>Daily Packet Loss Heatmap</h2>
<img src="loss_heatmap.png">
</div>

<div class="section">
<h2>Loss Events Timeline</h2>
<img src="event_timeline.png">
</div>

<div class="section">
<h2>Loss by hour</h2>
<img src="loss_by_hour.png">
</div>


<div class="section">
<h2>Diurnal Quality Pattern</h2>
<img src="diurnal_quality.png">
</div>

<div class="section">
<h2>Methodology</h2>
<ul>
<li>ICMP ping tests executed every second</li>
<li>Multiple external hosts used to avoid single-route bias</li>
<li>Latency capped visually for clarity</li>
<li>Packet loss aggregated into time bins for reliability analysis</li>
</ul>
</div>

</body>
</html>
"""

    out = REPORT / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report written to {out.resolve()}")


def main():
    REPORT.mkdir(exist_ok=True)

    run_plots()
    stats = run_stats()
    build_html(stats)


if __name__ == "__main__":
    main()
