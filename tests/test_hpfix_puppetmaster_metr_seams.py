from __future__ import annotations

import unittest

from puppetmaster.metr_seams import is_coordination_protocol_payload


class ProtocolRoleSeparatorTests(unittest.TestCase):
    def test_hyphen_mailbox_protocol_role_is_detected(self) -> None:
        value = {"role": "mailbox-protocol", "instruction": "do x"}
        self.assertTrue(is_coordination_protocol_payload(value))

    def test_underscore_peer_recruit_role_is_detected(self) -> None:
        value = {"role": "peer_recruit", "instruction": "do x"}
        self.assertTrue(is_coordination_protocol_payload(value))

    def test_underscore_mailbox_protocol_role_stays_detected(self) -> None:
        value = {"role": "mailbox_protocol", "instruction": "do x"}
        self.assertTrue(is_coordination_protocol_payload(value))

    def test_neutral_hyphen_role_stays_not_detected(self) -> None:
        value = {"role": "code-review", "instruction": "do x"}
        self.assertFalse(is_coordination_protocol_payload(value))


if __name__ == "__main__":
    unittest.main()
