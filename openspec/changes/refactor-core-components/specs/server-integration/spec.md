# Server Integration Requirements

## ADDED Requirements

### Requirement: Centralized Trade Lifecycle

The web server module MUST directly manage the trading strategy lifecycle and state without intermediate manager classes to reduce architectural complexity.

#### Scenario: Server Startup Strategy Initialization

- **Given** the application is starting
- **When** the server `lifespan` startup event occurs
- **Then** the trading strategy initialization logic MUST be invoked directly from the server module
- **And** no separate `TradeManager` singleton should be instantiated
