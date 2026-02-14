# Network Monitor & Reliability Reporter

This project logs continuous ping measurements and generates visual and statistical reports of internet connection stability.

The tool was created to diagnose intermittent packet loss and congestion issues that are not visible from simple speed tests.

---

## Features

- Continuous ping logging to multiple targets
- Live RTT + packet loss visualization
- Historical plots and heatmaps
- Statistical reliability report
- HTML/PDF report generation

---

## Folder structure

```
ping_logger.py        # collects ping data
live_plot.py          # live visualizer
make_plots.py         # generates report figures
net_stats.py          # computes reliability statistics
generate_report.py    # builds HTML/PDF report

data/                 # CSV logs (ignored in git)
report/               # generated figures + report (ignored)
```

---

## Requirements

Python 3.9+

Required packages:

```
pandas
numpy
matplotlib
```

Install with pip:

```
pip install pandas numpy matplotlib
```

or with conda:

```
conda install pandas numpy matplotlib
```

---

## Usage

### Start logging

```
./run_logger.sh
```

### View live network behaviour

```
python live_plot.py --minutes 120
```

### Generate plots

```
python make_plots.py
```

### Generate full report

```
python generate_report.py
```

Report will be written to:

```
report/report.html
```

---

## Measurement logic

The logger pings:

- Local router → detects local network issues
- Public DNS services → measures real internet reliability

Packet loss, latency and outages are derived from these measurements.

---

## Interpretation

This tool is meant to detect:

- congestion patterns
- evening peak instability
- intermittent packet loss
- routing issues

It is **not** a speed test.

---

## License

Personal diagnostic tool. Free to use and modify.
