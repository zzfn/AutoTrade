## MODIFIED Requirements
### Requirement: Unified Model Training

The system SHALL provide a unified interface for training ML models, capable of handling both initial historical training and rolling updates with recent data. When training is triggered, it MUST ensure sufficient training data exists; if data is missing or below a configurable threshold, the system MUST automatically synchronize data before proceeding.

#### Scenario: User initiates training

- **WHEN** user clicks "Train Model" in the UI
- **AND** confirms configuration (symbols, parameters)
- **THEN** system starts a background training task
- **AND** saves the result as a new model version upon completion

#### Scenario: Background execution

- **WHEN** training is triggered
- **THEN** the process runs asynchronously
- **AND** UI displays progress similar to previous rolling update
- **AND** user is notified upon success or failure

#### Scenario: Training with missing data

- **WHEN** training is triggered
- **AND** no training data is available
- **THEN** the system synchronizes the required data first
- **AND** training starts after synchronization completes

#### Scenario: Training with insufficient data

- **WHEN** training is triggered
- **AND** available data is below the configured threshold
- **THEN** the system synchronizes the required data first
- **AND** training starts after synchronization completes
