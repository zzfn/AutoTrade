# Replace YFinance with Alpaca

## Goal

Remove the dependency on `yfinance` and establish `alpaca-py` as the primary data provider for the AutoTrade system. This prepares the system for future integration with professional data sources like Polygon.io and removes reliance on scraped/unofficial APIs.

## Context

Currently, `DataProviderFactory` falls back to `yfinance` if Alpaca is unavailable. `yfinance` is based on scraping Yahoo Finance and is not reliable for production trading systems. The user explicitly requested removing `yfinance` and using `alpaca-py`.

## Scope

- Remove `yfinance` from `pyproject.toml`.
- Remove `YFinanceDataProvider` class from `autotrade/research/data/providers.py`.
- Update `DataProviderFactory` to enforce `AlpacaDataProvider`.
- Ensure `AlpacaDataProvider` is robust.

## Trade-offs

- **Pros**: More reliable data source (Alpaca), cleaner dependency tree, professional API usage.
- **Cons**: Requires valid Alpaca API keys to work (no free fallback for unauthorized users).
