# UIUC Statistics Datathon 2025 — Competition Brief
## Sponsored by Synchrony (Private-Label Credit Cards)

## Task
Build a model that predicts **intraday call center metrics in 30-minute intervals** for an entire month (August 2025), across **4 portfolios** (A, B, C, D).

## Forecast Targets (3 core metrics, 4 metrics in submission)
- **Call Volume (CV)** — Total incoming calls (answered + abandoned)
- **Abandoned Calls** — Callers who hung up before answered
- **Abandon Rate (ABD)** — % of callers who abandoned
- **CCT (Customer Care Time)** — Average handle time per call (seconds)

## Data Provided
- **Daily data:** 2 years (Jan 2024 - Dec 2025), 4 portfolios
- **Interval data:** 3 months (Apr-Jun 2024), 30-min granularity, 4 portfolios
- **Daily Staffing:** Full year 2025, agents per portfolio per day

## Constraints
- No negative forecast values
- Must use approved Docker images: PyTorch, TensorFlow, or sklearn/XGBoost/Prophet
- No external packages beyond what's in the Docker images
- Must use MLFlow for experiment tracking
- Account for: seasonality, DST, holidays, intraday/intraweek/intramonth patterns

## Submission (3 deliverables)
1. **Model output CSV** — `forecast_v##.csv` uploaded to AWS portal with Team ID
2. **Code** — Zipped, uploaded to Box folder with README
3. **Presentation + 7-min video** — Uploaded to Box

## Scoring
- Composite **cost-sensitive asymmetric loss**
- **Understaffing penalized MORE than overstaffing**
- Combines errors on volume, CCT, and abandon rate

## Timeline
- **Fri 3/27 6PM CST** — Datathon starts
- **Sat 3/28** — Office hours (AWS + Synchrony)
- **Sun 3/29 12PM CST** — Hard deadline
- **Wed 4/1 4PM CST** — Finalists present, winner announced
