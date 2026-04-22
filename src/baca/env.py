"""Gymnasium environment wrapping one Usiec Cepra run over JSON-RPC.

One env instance owns one ``runId`` on the engine server. ``reset`` disposes
the old run and creates a new one so subsequent episodes are independent.
"""

from __future__ import annotations

import contextlib
from typing import Any, Literal

import numpy as np
from gymnasium import Env, spaces

from baca.encoder import MAX_ACTIONS, encode, observation_space
from baca.rpc_client import RpcClient

EngineObservation = dict[str, Any]
RewardShape = Literal["none", "floor"]


class UsiecCepraEnv(Env[dict[str, np.ndarray], int]):
    """One-run gymnasium environment backed by a shared :class:`RpcClient`."""

    metadata = {"render_modes": []}  # noqa: RUF012 — gym base class declares non-ClassVar

    def __init__(
        self,
        rpc: RpcClient,
        character_id: str = "jedrek",
        difficulty: str = "normal",
        maryna_enabled: bool = True,
        reveal_all_piles: bool = False,
        base_seed: int | None = None,
        max_episode_steps: int = 1000,
        reward_shape: RewardShape = "none",
    ) -> None:
        super().__init__()
        self._rpc = rpc
        self._character_id = character_id
        self._difficulty = difficulty
        self._maryna_enabled = maryna_enabled
        self._reveal_all_piles = reveal_all_piles
        self._episode_counter = 0
        self._base_seed = base_seed
        self._max_episode_steps = max_episode_steps
        self._reward_shape: RewardShape = reward_shape

        self.observation_space = observation_space()
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._run_id: str | None = None
        self._last_obs: EngineObservation | None = None
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _ = options  # gym API contract — unused
        if self._run_id is not None:
            with contextlib.suppress(Exception):
                self._rpc.call("engine.dispose", {"runId": self._run_id})
            self._run_id = None

        effective_seed = self._resolve_seed(seed)
        create_params: dict[str, Any] = {
            "characterId": self._character_id,
            "difficulty": self._difficulty,
            "marynaEnabled": self._maryna_enabled,
            "rules": {
                "revealAllPiles": self._reveal_all_piles,
                "observationMode": "agent",
            },
        }
        if effective_seed is not None:
            create_params["seed"] = effective_seed

        run = self._rpc.call("engine.create", create_params)
        self._run_id = run["runId"]
        start = self._rpc.call("engine.startRun", {"runId": self._run_id})
        self._last_obs = start["observation"]
        self._episode_counter += 1
        self._step_count = 0
        return encode(self._last_obs), {"runId": self._run_id, "seed": effective_seed}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._run_id is None or self._last_obs is None:
            raise RuntimeError("step() called before reset()")

        legal = self._last_obs.get("legalActions") or []
        if not legal:
            raise RuntimeError("No legal actions available; episode should have terminated")
        if action < 0 or action >= len(legal):
            action = 0  # Out-of-mask indices collapse to the first legal action.

        result = self._rpc.call(
            "engine.applyAction",
            {"runId": self._run_id, "action": legal[action]},
        )
        self._last_obs = result["observation"]
        self._step_count += 1
        done = bool(self._last_obs.get("done"))
        truncated = (not done) and self._step_count >= self._max_episode_steps
        outcome = self._last_obs.get("outcome")
        reward = self._compute_reward(done, outcome, self._last_obs)

        info: dict[str, Any] = {}
        if done or truncated:
            info["outcome"] = outcome if done else "truncated"
            inline_summary = result.get("summary")
            if inline_summary is not None:
                info["summary"] = inline_summary
            else:
                with contextlib.suppress(Exception):
                    fallback = self._rpc.call("engine.getRunSummary", {"runId": self._run_id})
                    info["summary"] = fallback.get("summary")

        return encode(self._last_obs), reward, done, truncated, info

    def _compute_reward(
        self,
        done: bool,
        outcome: str | None,
        obs: EngineObservation,
    ) -> float:
        if not done:
            return 0.0
        is_win = 1.0 if outcome == "player_win" else 0.0
        if self._reward_shape == "none":
            return is_win
        floor = float(obs.get("floor", 0) or 0)
        floor_progress = min(1.0, max(0.0, floor / 15.0))
        return 0.1 * floor_progress + 0.9 * is_win

    def action_masks(self) -> np.ndarray:
        """Return current legal-action mask for sb3-contrib MaskablePPO."""
        if self._last_obs is None:
            return np.zeros(MAX_ACTIONS, dtype=bool)
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        legal = self._last_obs.get("legalActions") or []
        mask[: min(len(legal), MAX_ACTIONS)] = True
        return mask

    def close(self) -> None:
        if self._run_id is not None:
            with contextlib.suppress(Exception):
                self._rpc.call("engine.dispose", {"runId": self._run_id})
            self._run_id = None

    def _resolve_seed(self, external_seed: int | None) -> int | None:
        if external_seed is not None:
            return external_seed
        if self._base_seed is None:
            return None
        return (self._base_seed + self._episode_counter) & 0xFFFFFFFF
