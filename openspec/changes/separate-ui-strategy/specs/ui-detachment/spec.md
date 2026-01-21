## MODIFIED Requirements

### Requirement: Server Threading Model

The server threading model SHALL be inverted to run the strategy on the main thread and the UI on a background thread.

#### Scenario: Server Startup

Given the user runs `python autotrade/web/server.py`
Then the FastAPI server should start in a background thread
And the Trading Strategy should start in the main thread
And the UI should display the strategy status (updated from main thread)

#### Scenario: Lifespan decoupling

Given the FastAPI application starts
Then it should NOT automatically spawn a strategy thread
And it should only serve API endpoints and static files

#### Scenario: Signal Handling

Given the application is running
When the user sends a SIGINT (Ctrl+C)
Then the Strategy (Main Thread) should handle it (LumiBot handles gracefull shutdown)
And the UI server should shutdown gracefully
