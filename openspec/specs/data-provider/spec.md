# data-provider Specification

## Purpose
TBD - created by archiving change replace-yfinance-with-alpaca. Update Purpose after archive.
## Requirements
### Requirement: Exclusive Data Source

The system MUST explicitly and exclusively use Alpaca as the data source.

#### Scenario: Exclusive Alpaca Usage

- **Given** a request for historical data
- **When** `DataProviderFactory.get_provider()` is called
- **Then** it MUST return an instance of `AlpacaDataProvider`
- **And** it MUST NOT attempt to import or use `yfinance`

### Requirement: Data Format Standardization

Data returned by the provider MUST adhere to a strict schema for downstream compatibility.

#### Scenario: Data Consistency

- **Given** `AlpacaDataProvider` fetches data
- **When** the data is returned
- **Then** it MUST be a pandas DataFrame with MultiIndex `(timestamp, symbol)`
- **And** columns MUST include `open`, `high`, `low`, `close`, `volume` (lowercase)

