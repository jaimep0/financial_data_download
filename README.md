# Financial Data Extraction

Extract and consolidate historical financial and market data from Yahoo Finance using Python and `yfinance`.

## Data Collected

- **Financial Statements:** Quarterly income statements, balance sheets, and cash flow statements.
- **Revenue:** Historical revenue and related financial metrics.
- **Debt:** Total liabilities and reported debt balances.
- **Liquidity:** Current assets, current liabilities, cash, and cash equivalents.
- **Profitability:** Gross profit, operating income, and net income.
- **Market Data:** Historical stock prices, trading volume, and dividends.
- **Company Information:** Sector, industry, country, and shares outstanding.

## Output

A single CSV dataset (`data.csv`) containing financial information for multiple companies, organized by ticker and reporting period.

## Technologies

- Python
- pandas
- yfinance
- tqdm

## Usage

```python
df = extract_data(tickers)
```

## Project Scope

This repository focuses exclusively on **data extraction and consolidation**.

Financial ratios, valuation metrics, feature engineering, statistical analysis, and machine learning models are developed separately in the **Financial Analytics** project.
