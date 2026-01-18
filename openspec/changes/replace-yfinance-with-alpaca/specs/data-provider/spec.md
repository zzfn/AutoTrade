# Data Provider Specifications

## REMOVED Requirements

### Requirement: Fallback Mechanism

The system should no longer support fallback mechanisms to secondary data providers when the primary one fails.

#### Scenario: Fallback to YFinance

- **Given** Alpaca credentials are missing or invalid
- **When** the system requests a data provider
- **Then** the system SHOULD NOT fall back to YFinance
- **And** the system SHOULD raise a clear error indicating missing configuration

## ADDED Requirements

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
