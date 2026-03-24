"""
Unit tests for Byzantine agent subtypes.
Owner: C (Byzantine & Comms)
Run: pytest tests/test_byzantine.py -v

CRITICAL: Byzantine subtypes must corrupt only communication, not movement.
These tests verify that invariant before Experiment 1 runs.
"""

import pytest


@pytest.mark.skip(reason="BYZ-03 not yet implemented")
def test_random_noise_corrupts_message():
    """Random noise Byzantine sends a message different from ground truth."""
    from agents.byzantine import RandomNoiseByzantine
    agent = RandomNoiseByzantine(agent_id=0, seed=0)
    true_pos = (5, 5)
    msg = agent.get_message(true_hider_pos=true_pos)
    assert msg["believed_hider_pos"] != true_pos


@pytest.mark.skip(reason="BYZ-03 not yet implemented")
def test_silence_sends_null():
    """Silent Byzantine sends null/empty message."""
    from agents.byzantine import SilentByzantine
    agent = SilentByzantine(agent_id=0)
    msg = agent.get_message(true_hider_pos=(3, 7))
    assert msg is None or msg["believed_hider_pos"] is None


@pytest.mark.skip(reason="BYZ-03 not yet implemented")
def test_byzantine_moves_honestly():
    """Byzantine agents take the same actions as honest agents given same obs."""
    from agents.byzantine import RandomNoiseByzantine
    from agents.ppo import PPOAgent
    obs = {"agent_pos": (2, 2), "hider_pos": None, "step": 10}
    honest = PPOAgent(agent_id=0)
    byz = RandomNoiseByzantine(agent_id=0)
    assert honest.act(obs) == byz.act(obs)
