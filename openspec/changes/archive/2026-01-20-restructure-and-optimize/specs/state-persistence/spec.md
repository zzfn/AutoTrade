# State Persistence

## ADDED Requirements

### Requirement: Automated Backup

Critical trading variables MUST be persisted automatically to prevent state loss.

#### Scenario: Automatic Variable Backup

- **Given** a Lumibot Strategy running live
- **When** `on_trading_iteration` completes
- **Then** the contents of `self.vars` must be serialized to the database defined by `DB_CONNECTION_STR`

### Requirement: State Recovery

Strategies MUST be able to resume their state seamlesssly after a restart.

#### Scenario: Crash Recovery

- **Given** a strategy that crashed and is restarted
- **When** `initialize` runs
- **Then** `self.vars` must be automatically populated with the last saved state from the DB

### Requirement: Default Configuration

The system MUST provide sensible defaults for persistence to ensure ease of use.

#### Scenario: DB Configuration

- **Given** no external database
- **When** the app starts
- **Then** it should default to a local SQLite file (e.g. `trading_state.db`) if not configured otherwise
