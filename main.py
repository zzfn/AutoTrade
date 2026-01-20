import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="AutoTrade CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        help="Command to run: 'run' to start web server",
    )
    args = parser.parse_args()

    if args.command == "run":
        print("Starting AutoTrade Web Server (FastAPI + React)...")
        # Using reload=True for dev experience as requested
        uvicorn.run("autotrade.web.server:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print(f"Unknown command: {args.command}")
        print("Available commands: run")


if __name__ == "__main__":
    main()
