# Project Structure Refactoring

## ADDED Requirements

### Requirement: Standardized Directory Structure

The project MUST follow a clean, functional layered architecture to improve navigability and separation of concerns.

#### Scenario: Directory Layout

- **Given** the current mixed structure of `execution` and `research`
- **When** the project is refactored
- **Then** the root must contain `core/`, `strategies/`, `ml/`, `data/`, and `web/`
- **And** `main.py` should import from these new locations without error

### Requirement: Code Organization

Shared utilities MUST be centralized to avoid code duplication and circular dependencies.

#### Scenario: Code Reusability

- **Given** shared utilities used by both Strategy and Web Server
- **When** implemented
- **Then** they must be located in `core/` or `autotrade/utils/` to prevent circular imports
