# Chart Pattern Recognition and Trading

This repository contains a chart-pattern recognition system capable of detecting patterns across multiple time granularities, along with automated trading logic and backtesting functions.

---

## 1. Usage

Run the detector from the command line:

```bash
python detect_patterns.py --input data/close.csv --granularity 5min --output results/patterns.csv
```

### 1.1 Data Format Requirements
close.csv
-- Note that the close.csv must contain a column named Date. 
-- All remaining columns represent tickers. 
-- Each row corresponds to the close price of each ticker on that date

tickers.csv
-- Must contain a single column listing all ticker names used by the detector

### 1.2 Granularity
-- The Date field logically represents the timestamp of the observation
-- Though named “Date”, the system is fully scalable to different time resolutions such as seconds, minutes, hours, days, or weeks, provided your input data matches that resolution

## 2. Sample Images — Price Skeletons
## 3. Sample Images — Detected Patterns
## 4. Backtesting Results — 220 Liquid Indian Stocks (5 Years)
