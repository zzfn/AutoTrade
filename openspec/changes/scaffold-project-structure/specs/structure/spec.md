# Spec: Directory Structure Refactor

## MODIFIED Requirements

### Requirement: Root-Level Package Separation

The `autotrade` source code MUST be organized into `research`, `execution`, and `shared` submodules to enforce strict separation of concerns.

#### Scenario: Research Code Isolation

GIVEN the project is structured
WHEN I inspect `autotrade/research`
THEN it SHOULD contain Qlib-related code, factor definitions, and model training scripts
AND it SHOULD NOT import from `autotrade/execution`.

#### Scenario: Execution Code Isolation

GIVEN the project is structured
WHEN I inspect `autotrade/execution`
THEN it SHOULD contain LumiBot strategies and broker adapters
AND it SHOULD depend on `autotrade/shared`.

### Requirement: Shared Infrastructure

Common utilities MUST be located in `autotrade/shared`.

#### Scenario: Config Access

GIVEN a strategy in `execution` needs configuration
WHEN it imports config
THEN it SHOULD import from `autotrade.shared.config`, not `autotrade.config` directly.
