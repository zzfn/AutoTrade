# Design

## Architecture

The `BaseDataProvider` abstract base class remains the interface contract.
`AlpacaDataProvider` becomes the sole implementation for now.
`DataProviderFactory` simplifies to returning `AlpacaDataProvider` or raising a configuration error.

## Future Extensibility

The structure retains `BaseDataProvider`, allowing future addition of `PolygonDataProvider` or others without breaking consumers like `QlibDataAdapter`.

## Verification

- Unit tests should mock `AlpacaDataProvider`.
- Integration tests will require `ALPACA_API_KEY` and `ALPACA_API_SECRET`.
