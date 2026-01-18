## ADDED Requirements

### Requirement: Unified Model Training

The system SHALL provide a unified interface for training ML models, capable of handling both initial historical training and rolling updates with recent data.

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

## REMOVED Requirements

### Requirement: 模型训练

**Reason**: Merged into Unified Model Training. The distinction between offline/script training and UI training is removed in favor of a unified capability.
**Migration**: Use Unified Model Training interface.

### Requirement: Rolling 模型更新

**Reason**: Merged into Unified Model Training. Rolling update is just a training run with latest data.
**Migration**: Use Unified Model Training interface.
