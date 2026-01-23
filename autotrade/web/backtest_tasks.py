import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from lumibot.backtesting import AlpacaBacktesting

from autotrade.core.config import config
from autotrade.ml import ModelManager
from autotrade.qlib_ml_strategy import QlibMLStrategy


_DB_FILENAME = "backtest_tasks.sqlite"
_TASK_TABLE = "backtest_tasks"
_LOG_TABLE = "backtest_task_logs"


def _db_path() -> Path:
    return config.DATA_DIR / _DB_FILENAME


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    config.ensure_directories()
    with _connect_db() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TASK_TABLE} (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                params_json TEXT NOT NULL,
                result_json TEXT,
                error_message TEXT
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_LOG_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES {_TASK_TABLE}(task_id)
            )
            """
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_logs_task_id ON {_LOG_TABLE}(task_id)")


def create_task(params: dict[str, Any]) -> dict[str, Any]:
    init_db()
    task_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    payload = json.dumps(params, ensure_ascii=True)
    with _connect_db() as conn:
        conn.execute(
            f"""
            INSERT INTO {_TASK_TABLE}
            (task_id, status, created_at, updated_at, params_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, "queued", now, now, payload),
        )
    return {"task_id": task_id, "status": "queued", "created_at": now}


def append_log(task_id: str, message: str) -> None:
    init_db()
    timestamp = datetime.now().strftime("%H:%M:%S")
    with _connect_db() as conn:
        conn.execute(
            f"""
            INSERT INTO {_LOG_TABLE} (task_id, timestamp, message)
            VALUES (?, ?, ?)
            """,
            (task_id, timestamp, message),
        )


def update_task(
    task_id: str,
    *,
    status: str | None = None,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    init_db()
    now = datetime.now().isoformat()
    updates = ["updated_at = ?"]
    params: list[Any] = [now]
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if result is not None:
        updates.append("result_json = ?")
        params.append(json.dumps(result, ensure_ascii=True))
    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)
    params.append(task_id)
    with _connect_db() as conn:
        conn.execute(
            f"UPDATE {_TASK_TABLE} SET {', '.join(updates)} WHERE task_id = ?",
            params,
        )


def get_task(task_id: str) -> dict[str, Any] | None:
    init_db()
    with _connect_db() as conn:
        task_row = conn.execute(
            f"SELECT * FROM {_TASK_TABLE} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if not task_row:
            return None
        logs = conn.execute(
            f"SELECT timestamp, message FROM {_LOG_TABLE} WHERE task_id = ? ORDER BY id ASC",
            (task_id,),
        ).fetchall()
    result = json.loads(task_row["result_json"]) if task_row["result_json"] else None
    return {
        "task_id": task_row["task_id"],
        "status": task_row["status"],
        "created_at": task_row["created_at"],
        "updated_at": task_row["updated_at"],
        "params": json.loads(task_row["params_json"]),
        "result": result,
        "error_message": task_row["error_message"],
        "logs": [f"[{row['timestamp']}] {row['message']}" for row in logs],
    }


def _claim_next_task() -> dict[str, Any] | None:
    init_db()
    with _connect_db() as conn:
        row = conn.execute(
            f"""
            SELECT task_id, params_json
            FROM {_TASK_TABLE}
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        now = datetime.now().isoformat()
        updated = conn.execute(
            f"""
            UPDATE {_TASK_TABLE}
            SET status = 'running', updated_at = ?
            WHERE task_id = ? AND status = 'queued'
            """,
            (now, row["task_id"]),
        )
        if updated.rowcount == 0:
            return None
        params = json.loads(row["params_json"])
        return {"task_id": row["task_id"], "params": params}


def _execute_backtest(params: dict[str, Any], log_fn: Callable[[str], None]) -> dict[str, Any]:
    log_fn("Backtest started.")

    backtesting_start = datetime.strptime(
        params.get("start_date", "2023-01-01"), "%Y-%m-%d"
    )
    backtesting_end = datetime.strptime(
        params.get("end_date", "2023-01-31"), "%Y-%m-%d"
    )

    symbol_input = params.get("symbol", "SPY")
    symbols = [
        s.strip().replace('"', "").replace("'", "")
        for s in symbol_input.split(",")
        if s.strip()
    ]
    if not symbols:
        symbols = ["SPY"]

    interval = "5min"
    log_fn(
        f"Backtesting {symbols} from {backtesting_start} to {backtesting_end} (Interval: {interval})"
    )

    strategy_class = QlibMLStrategy
    model_manager = ModelManager()

    lumibot_interval = "minute"

    model_name = params.get("model_name")
    if not model_name:
        model_name = model_manager.get_current_model()

    backtest_params = {
        "symbols": symbols,
        "model_name": model_name,
        "top_k": params.get("top_k", 3),
        "sleeptime": "0S",
        "timestep": "5M",
    }

    benchmark = "SPY" if len(symbols) > 1 or symbols[0] != "SPY" else symbols[0]

    start_time = datetime.now()

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    paper_trading = os.getenv("ALPACA_PAPER", "True").lower() == "true"

    if not api_key or not api_secret:
        raise ValueError("缺少 Alpaca API 凭证，请设置 ALPACA_API_KEY 和 ALPACA_API_SECRET 环境变量")

    alpaca_config = {
        "API_KEY": api_key,
        "API_SECRET": api_secret,
        "PAPER": paper_trading,
    }

    stop_event = threading.Event()
    progress_start = datetime.now()

    def progress_loop() -> None:
        while not stop_event.wait(10):
            elapsed = int((datetime.now() - progress_start).total_seconds())
            log_fn(f"Backtest running... elapsed {elapsed}s")

    progress_thread = threading.Thread(target=progress_loop, daemon=True)
    progress_thread.start()
    try:
        strategy_class.backtest(
            AlpacaBacktesting,
            backtesting_start,
            backtesting_end,
            config=alpaca_config,
            benchmark_asset=benchmark,
            parameters=backtest_params,
            time_unit=lumibot_interval,
            api_key=api_key,
            api_secret=api_secret,
            show_progress_bar=True,
        )
    finally:
        stop_event.set()
        progress_thread.join(timeout=1)

    import glob

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "logs")
    html_files = glob.glob(os.path.join(logs_dir, "*.html"))
    new_reports = []
    for report in html_files:
        if os.path.getmtime(report) >= start_time.timestamp() - 5:
            new_reports.append(os.path.basename(report))

    tearsheet = next((f for f in new_reports if "tearsheet" in f), None)
    trades_report = next((f for f in new_reports if "trades" in f), None)

    result = {
        "tearsheet": f"/reports/{tearsheet}" if tearsheet else None,
        "trades": f"/reports/{trades_report}" if trades_report else None,
        "timestamp": datetime.now().isoformat(),
    }
    if tearsheet or trades_report:
        log_fn(f"Backtest reports generated: {new_reports}")

    log_fn("Backtest finished successfully.")
    return result


def run_worker_loop(poll_interval: float = 1.0) -> None:
    init_db()
    while True:
        task = _claim_next_task()
        if not task:
            time.sleep(poll_interval)
            continue
        task_id = task["task_id"]
        params = task["params"]
        append_log(task_id, "Starting backtest task.")
        try:
            result = _execute_backtest(params, lambda msg: append_log(task_id, msg))
            update_task(task_id, status="completed", result=result, error_message=None)
        except Exception as exc:
            append_log(task_id, f"Backtest execution failed: {exc}")
            update_task(task_id, status="failed", error_message=str(exc))


if __name__ == "__main__":
    run_worker_loop()
