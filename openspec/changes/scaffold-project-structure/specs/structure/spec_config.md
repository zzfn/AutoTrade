# Spec: Configuration System

## ADDED Requirements

### Requirement: YAML-Based Configuration

The system MUST support configuration via YAML files, specifically for defining the trading universe (symbols).

#### Scenario: Define Trading Universe

GIVEN a `universe.yaml` (or `config.yaml`) file exists with a list of symbols
WHEN the application starts (either Research or Execution mode)
THEN it SHOULD load this list of symbols as the active trading universe
AND both Qlib (for data filtering) and LumiBot (for market data subscription) SHOULD use this same source of truth.

#### Scenario: Shared Access

GIVEN the configuration is loaded
WHEN code in `autotrade.execution` needs the symbol list
THEN it SHOULD access it via `autotrade.shared.config`
AND NOT read the file directly to ensure consistency.
