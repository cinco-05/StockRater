# StockRater

StockRater is a command-line fundamental equity valuation tool. It retrieves market and financial statement data, estimates economic fair value and a margin-of-safety buy-below price, and reports whether a stock appears overvalued or undervalued.

The model includes FCFF/WACC and FCFE valuations, reverse DCF analysis, relative multiples, Piotroski F-Score, Altman Z-Score, Beneish M-Score, financial statement integrity checks, and optional historical backtesting.

> **Disclaimer:** This project is an educational quantitative screen. It is not financial advice. Market data can be delayed, incomplete, or incorrect; independently verify any result before making an investment decision.

## Requirements

- Python 3.10 or newer
- Internet access for live stock analysis

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/cinco-05/StockRater.git
cd StockRater
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Analyze a stock:

```bash
python stock_analyzer.py --ticker AAPL
```

Add peer comparisons or request a detailed report:

```bash
python stock_analyzer.py --ticker AAPL --peers MSFT,GOOGL --detail detailed
```

Other useful options:

```bash
python stock_analyzer.py --ticker AAPL --json
python stock_analyzer.py --ticker AAPL --mos 0.30
python stock_analyzer.py --self-test
python stock_analyzer.py --help
```

## Improved V6

`stock_analyzer_v6.py` is the recommended version. It adds explicit reverse-DCF bounds, strict benchmark matching in backtests, a versioned self-healing cache, validated model configuration, improved error diagnostics, automated regression tests, and GitHub Actions CI.

```bash
python stock_analyzer_v6.py --ticker AAPL
python stock_analyzer_v6.py --ticker AAPL --debug
python stock_analyzer_v6.py --config-json model_config.example.json --ticker AAPL
python stock_analyzer_v6.py --self-test
```

Install development dependencies and run the regression suite with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Copy `model_config.example.json` before editing model assumptions. V6 validates supported keys, numeric ranges, family weights, and sector-profile shapes before analysis begins.

Running the script without `--ticker` starts an interactive prompt.

## Backtesting

Save point-in-time results for later evaluation:

```bash
python stock_analyzer.py --ticker AAPL --snapshot-csv history.csv
```

Evaluate completed observations:

```bash
python stock_analyzer.py --backtest-csv history.csv
```

Use `--calibrate-out weights.json` with a backtest to generate candidate category weights, then apply validated weights with `--weights-json weights.json`.

## Data and cache

Live data is provided through `yfinance`. Responses are cached in the operating system's temporary directory for six hours by default. Use `--no-cache` to bypass the cache or `--clear-cache` to remove it.

The following environment variables customize V5 cache behavior:

- `SAV5_CACHE`: cache directory
- `SAV5_CACHE_TTL`: cache lifetime in seconds

V6 uses the corresponding `SAV6_CACHE` and `SAV6_CACHE_TTL` variables and automatically invalidates incompatible cache schemas.

## License

Licensed under the [MIT License](LICENSE).
