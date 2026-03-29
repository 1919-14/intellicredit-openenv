# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""IntelliCredit-CreditAppraisal-v1 Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import IntelliCreditAction, IntelliCreditObservation


class IntelliCreditClient(
    EnvClient[IntelliCreditAction, IntelliCreditObservation, State]
):
    """
    Client for the IntelliCredit Credit Appraisal Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with IntelliCreditClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.application_summary.company_name)
        ...
        ...     result = client.step(IntelliCreditAction(decision=1))
        ...     print(result.observation.reward)
    """

    def _step_payload(self, action: IntelliCreditAction) -> Dict:
        return {
            "decision": action.decision,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[IntelliCreditObservation]:
        obs_data = payload.get("observation", {})
        observation = IntelliCreditObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
