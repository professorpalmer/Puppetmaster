"""Additive CLI for named cells (status / inspect / tick)."""
from __future__ import annotations

import json
from typing import Any

from puppetmaster.cell import (
    CellNotFoundError,
    CellRegistry,
    InvalidCellIdError,
    interned_poll,
)


def _print_status(payload: dict[str, Any], *, json_mode: bool) -> int:
    if json_mode:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if isinstance(payload, list):
        if not payload:
            print("cells: (none)")
            return 0
        for item in payload:
            _print_one(item)
        return 0
    _print_one(payload)
    return 0


def _print_one(payload: dict[str, Any]) -> None:
    print(f"cell {payload.get('cell_id')}:")
    print(f"  path:         {payload.get('path')}")
    print(f"  inbox_depth:  {payload.get('inbox_depth')}")
    print(f"  hibernating:  {payload.get('hibernating')}")
    print(f"  next_alarm:   {payload.get('next_alarm') or '-'}")


def _registry(store) -> CellRegistry:
    return CellRegistry(store.root)


def _run_cell_status_command(args, store) -> int:
    registry = _registry(store)
    json_mode = bool(getattr(args, "json", False))
    cell_id = getattr(args, "cell_id", None)
    try:
        if cell_id:
            payload = registry.status(cell_id)
        else:
            payload = registry.list_cells()
    except (CellNotFoundError, InvalidCellIdError) as exc:
        print(f"error: {exc}")
        return 1
    return _print_status(payload, json_mode=json_mode)


def _run_cell_inspect_command(args, store) -> int:
    registry = _registry(store)
    json_mode = bool(getattr(args, "json", False))
    cell_id = getattr(args, "cell_id", None)
    if not cell_id:
        print("error: cell-inspect requires a cell id")
        return 1
    try:
        payload = registry.inspect(cell_id)
    except (CellNotFoundError, InvalidCellIdError) as exc:
        print(f"error: {exc}")
        return 1
    if json_mode:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    _print_one(payload)
    inbox = payload.get("inbox") or []
    print(f"  inbox_rows:   {len(inbox)}")
    for row in inbox:
        print(
            f"    #{row['id']} {row['status']} {row['kind']} "
            f"at {row['enqueued_at']}"
        )
    return 0


def _run_cell_tick_command(args, store) -> int:
    woken = interned_poll(store)
    payload = {"woken": len(woken), "cells": woken}
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"cell-tick: woke {payload['woken']} cell(s)")
    for item in woken:
        _print_one(item)
    return 0
