from __future__ import annotations

import argparse
import unittest

from puppetmaster.cli.helpers import (
    allowed_model_ids_list_from_mapping,
    append_allowed_models_cli_flags,
    routing_payload_from_args,
)


class BlankAllowlistFailsClosedTests(unittest.TestCase):
    def test_blank_allowed_models_keeps_empty_allowlist(self) -> None:
        args = argparse.Namespace(auto_route=True, allowed_models=["  "])
        payload = routing_payload_from_args(args, adapter="cursor")
        self.assertEqual(payload.get("allowed_model_ids"), [])

    def test_blank_allowed_models_without_auto_route_still_routes(self) -> None:
        args = argparse.Namespace(auto_route=False, allowed_models=[""])
        payload = routing_payload_from_args(args, adapter="cursor")
        self.assertTrue(payload.get("auto_route"))
        self.assertEqual(payload.get("allowed_model_ids"), [])

    def test_flag_absent_has_no_allowlist(self) -> None:
        args = argparse.Namespace(auto_route=True, allowed_models=[])
        payload = routing_payload_from_args(args, adapter="cursor")
        self.assertEqual(
            payload, {"auto_route": True, "allowed_adapters": ["cursor"]}
        )

    def test_no_flags_gives_empty_payload(self) -> None:
        args = argparse.Namespace(auto_route=False, allowed_models=[])
        self.assertEqual(routing_payload_from_args(args, adapter="cursor"), {})

    def test_real_values_keep_blank_filtered_list(self) -> None:
        args = argparse.Namespace(auto_route=True, allowed_models=["a", " ", "b"])
        payload = routing_payload_from_args(args, adapter="cursor")
        self.assertEqual(payload.get("allowed_model_ids"), ["a", "b"])


class AppendAllowedModelsFlagTests(unittest.TestCase):
    def test_mcp_empty_list_forwards_empty_flag(self) -> None:
        command: list[str] = []
        append_allowed_models_cli_flags(
            command, allowed_model_ids_list_from_mapping({"allowed_model_ids": []})
        )
        self.assertEqual(command, ["--allowed-models", ""])

    def test_forwarded_empty_flag_round_trips_to_empty_allowlist(self) -> None:
        command: list[str] = []
        append_allowed_models_cli_flags(command, [])
        args = argparse.Namespace(auto_route=True, allowed_models=command[1::2])
        payload = routing_payload_from_args(args, adapter="cursor")
        self.assertEqual(payload.get("allowed_model_ids"), [])

    def test_absent_key_appends_nothing(self) -> None:
        command: list[str] = []
        append_allowed_models_cli_flags(
            command, allowed_model_ids_list_from_mapping({})
        )
        self.assertEqual(command, [])

    def test_values_append_one_flag_each(self) -> None:
        command: list[str] = []
        append_allowed_models_cli_flags(command, ["a", "b"])
        self.assertEqual(command, ["--allowed-models", "a", "--allowed-models", "b"])


if __name__ == "__main__":
    unittest.main()
