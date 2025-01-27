from __future__ import annotations
from typing import Tuple

import numpy as np
import tensordict
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from torchrl.data import (
    # Binary,
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs import EnvBase

class AIGEnv(EnvBase):
    def __init__(
        self,
        embedding_size: int,
        const_node: bool = True,
        reward_type: str = "shaped",
        negative_reward: float = -1.0,
        reset_dict: TensorDict | None = None,
    ) -> None:
        super(AIGEnv, self).__init__()
        self.batch_size: torch.Size = torch.Size()
        self.embedding_size: int = embedding_size
        self.n_pos: torch.Tensor = torch.tensor(1, dtype=torch.int32)
        self.const_node: bool = const_node
        self.reward_type: str = reward_type
        self.negative_reward: float = negative_reward

        self.action_spec: UnboundedDiscreteTensorSpec = UnboundedDiscreteTensorSpec(
            shape=torch.Size([1])
        )
        self.full_observation_spec: CompositeSpec = CompositeSpec(
            nodes=DiscreteTensorSpec(
                embedding_size, shape=(-1, embedding_size), dtype=torch.bool
            ),
            target=DiscreteTensorSpec(
                self.embedding_size, shape=(-1, embedding_size), dtype=torch.bool
            ),
            edge_type=DiscreteTensorSpec(1, dtype=torch.int64),
            left=DiscreteTensorSpec(1, dtype=torch.int64),
            right=DiscreteTensorSpec(1, dtype=torch.int64),
            action_mask=DiscreteTensorSpec(1, dtype=torch.bool),
            num_inputs=UnboundedDiscreteTensorSpec(1, dtype=torch.int32),
            num_outputs=UnboundedDiscreteTensorSpec(1, dtype=torch.int32),
            # reward=UnboundedContinuousTensorSpec(shape=torch.Size([1])),
        )
        self.state_spec: CompositeSpec = self.observation_spec.clone()

        self.reward_spec: CompositeSpec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(shape=torch.Size([1]))
        )
        self.full_done_spec: CompositeSpec = CompositeSpec(
            done=DiscreteTensorSpec(2, shape=torch.Size([1]), dtype=torch.bool)
        )
        self.full_done_spec["terminated"] = self.full_done_spec["done"].clone()
        self.full_done_spec["truncated"] = self.full_done_spec["done"].clone()

        if reset_dict is not None:
            self._reset(reset_dict)

        self.state: TensorDict = TensorDict(
            {
                "nodes": torch.empty(
                    (0, self.embedding_size), dtype=torch.bool, requires_grad=False
                ),
                "target": torch.empty(
                    (1, self.embedding_size), dtype=torch.bool, requires_grad=False
                ),
                "edge_type": torch.empty((0), dtype=torch.int64, requires_grad=False),
                "left": torch.empty((0), dtype=torch.int64, requires_grad=False),
                "right": torch.empty((0), dtype=torch.int64, requires_grad=False),
                # "done": torch.zeros((1), dtype=torch.bool, requires_grad=False),
                # "terminated": torch.zeros((1), dtype=torch.bool, requires_grad=False),
                # "truncated": torch.zeros((1), dtype=torch.bool, requires_grad=False),
                # "reward": torch.zeros((1), dtype=torch.float32, requires_grad=False),
                "num_inputs": torch.zeros((1), dtype=torch.int32, requires_grad=False),
                "num_outputs": torch.ones((1), dtype=torch.int32, requires_grad=False),
                "action_mask": torch.empty((0), dtype=torch.bool, requires_grad=False),
            },
            batch_size=torch.Size([]),
        )

    def _reset(self, reset_td: TensorDict) -> TensorDict:
        if reset_td is not None:
            # shape = reset_td.shape if reset_td is not None else ()
            # state = self.state_spec.zero(shape)
            if reset_td["num_inputs"] != self.state["num_inputs"]:
                self.state["num_inputs"][0] = reset_td["num_inputs"][0]
                self.state["nodes"] = self._construct_inputs()
            else:
                if self.const_node:
                    self.state.set(
                        "nodes", self.state["nodes"][: self.state["num_inputs"] + 1, :]
                    )
                else:
                    self.state.set(
                        "nodes", self.state["nodes"][: self.state["num_inputs"], :]
                    )
            target_repeat = 1
            if reset_td["target"].shape[-1] != self.embedding_size:
                target_repeat = self.embedding_size // reset_td["target"].shape[-1]

            self.state.set_("target", reset_td["target"].repeat(1, target_repeat))
            self.state.set(
                "edge_type",
                torch.empty(
                    (0), requires_grad=False, device=self.device, dtype=torch.int64
                ),
            )
            self.state.set(
                "left",
                torch.empty(
                    (0), requires_grad=False, device=self.device, dtype=torch.int64
                ),
            )
            self.state.set(
                "right",
                torch.empty(
                    (0), requires_grad=False, device=self.device, dtype=torch.int64
                ),
            )

        self.state.set("action_mask", self.action_mask(self.state))

        # Reset the done spec to False
        self.state.update(self.full_done_spec.zero(self.state.shape))

        return self.state

    def _step(self, action_td: TensorDict) -> TensorDict:
        action_td = action_td.clone(False)

        edge_type, left_id, right_id = self.unravel_index(
            action_td["action"], action_td["nodes"].shape[-2]
        )

        # left = self.state["nodes"][left_id, :].view(-1)
        # left = torch.index_select(self.state["nodes"], -2, torch.ones(1).to(torch.int64)).view(-1)
        # print("Left ID", left_id, left_id.shape)
        # vmap_index = torch.vmap(torch.index_select, in_dims=(0, None, 0))
        # left = vmap_index(self.state["nodes"], -2, left_id.to(torch.int64).view(-1))
        left = torch.index_select(action_td["nodes"], -2, left_id)
        # left1 = action_td["nodes"][left_id, :]
        # print(left.shape, left1.shape)
        # assert torch.equal(left, left1.unsqueeze(0))
        # left = action_td["nodes"][left_id, :]
        # right = self.state["nodes"][right_id, :].view(-1)
        # right = torch.index_select(self.state["nodes"], -2, torch.zeros(1).to(torch.int64)).view(-1)
        # print("Right ID", right_id, right_id.shape)
        # print("nodes shape", action_td["nodes"], action_td["nodes"].shape)

        # right = vmap_index(self.state["nodes"], -2, right_id.to(torch.int64).view(-1))
        right = torch.index_select(action_td["nodes"], -2, right_id)
        # right = action_td["nodes"][right_id, :]
        new_node = self.get_new_node(edge_type, left, right)
        action_td["nodes"] = torch.cat((action_td["nodes"], new_node), dim=-2)
        action_td["edge_type"] = torch.cat(
            (action_td["edge_type"], edge_type.view(-1)), dim=-1
        )
        action_td["left"] = torch.cat((action_td["left"], left_id.view(-1)), dim=-1)
        action_td["right"] = torch.cat((action_td["right"], right_id.view(-1)), dim=-1)
        action_td["action_mask"] = self.action_mask(action_td)

        # print("New Node", new_node, new_node.shape)
        # if torch.equal(new_node.unsqueeze(0), action_td["target"]) or torch.equal(
        #     new_node.unsqueeze(0), ~action_td["target"]
        # ):
        # print("New Node", new_node.shape)
        # print("Target", action_td["target"].shape)
        # print(torch.equal(new_node, action_td["target"]))

        action_td["terminated"] = torch.all(
            new_node == action_td["target"]
        ) | torch.all(new_node == ~action_td["target"])

        action_td["done"] = action_td["terminated"] | action_td["truncated"]
        action_td["reward"] = self._reward_function(
            action_td, self.const_node, self.reward_type, self.device
        )
        # if :
        #     action_td["done"] = torch.tensor(True, device=self.device)  # type: ignore
        #     action_td["terminated"] = torch.tensor(True, device=self.device)
        #     action_td["reward"] = self._reward_function()

        # elif action_td["truncated"]:
        #     action_td["done"] = torch.tensor(True, device=self.device)
        #     action_td["truncated"] = torch.tensor(True, device=self.device)
        #     action_td["reward"] = self._reward_function()
        self.state = action_td

        return self.state

    def batch_step(self, action_td: TensorDict) -> TensorDict:
        action_td = action_td.clone(False)

        def _batch_step(action_td: TensorDict) -> TensorDict:
            edge_type, left_id, right_id = self.unravel_index(
                action_td["action"], action_td["nodes"].shape[-2]
            )
            left = torch.index_select(action_td["nodes"], -2, left_id)
            right = torch.index_select(action_td["nodes"], -2, right_id)
            left_stack = torch.cat([left, left, ~left, ~left], dim=0)
            right_stack = torch.cat([right, ~right, right, ~right], dim=0)
            new_node = torch.index_select(left_stack & right_stack, 0, edge_type)
            action_td["nodes"] = torch.cat((action_td["nodes"], new_node), dim=-2)
            action_td["edge_type"] = torch.cat(
                (action_td["edge_type"], edge_type.view(-1)), dim=-1
            )
            action_td["left"] = torch.cat((action_td["left"], left_id), dim=-1)
            action_td["right"] = torch.cat((action_td["right"], right_id), dim=-1)
            action_td["action_mask"] = self.action_mask(action_td)
            action_td["terminated"] = torch.all(
                new_node == action_td["target"], dim=-1
            ) | torch.all(new_node == ~action_td["target"], dim=-1)
            action_td["done"] = action_td["terminated"] | action_td["truncated"]
            action_td["reward"] = action_td["terminated"].to(torch.float32)
            return action_td

        return torch.vmap(_batch_step, in_dims=0)(action_td)

    @staticmethod
    def _reward_function(
        state: TensorDict,
        const_node: bool = True,
        reward_type: str = "shaped",
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        if state["terminated"]:
            if reward_type == "shaped":
                if const_node:
                    return AIGEnv._shaped_reward_function_const(
                        state["nodes"], state["num_inputs"]
                    )
                return AIGEnv._shaped_reward_function_noconst(
                    state["nodes"], state["num_inputs"]
                )
            else:
                return AIGEnv._simple_reward_function(
                    state["nodes"], state["num_inputs"]
                )
        else:
            return torch.zeros((1), device=device)
        if state["truncated"]:
            return torch.tensor(self.negative_reward, device=self.device)

    def _construct_inputs(self) -> torch.BoolTensor:
        tts = []

        # add constant node
        if self.const_node:
            tts.append(
                torch.zeros(
                    self.embedding_size,
                    device=self.device,
                    dtype=torch.bool,
                    requires_grad=False,
                )
            )

        # number of times to repeat the truth table to match the embedding size
        repeats = self.embedding_size // 2 ** torch.sym_int(self.state["num_inputs"])  # type: ignore

        # create truth tables for each input
        for i in range(torch.sym_int(self.state["num_inputs"])):
            bits = 1 << i
            res = ~(~0 << bits)
            mask_bits = bits << 1
            for _ in range(self.state["num_inputs"] - (i + 1)):

                res |= res << mask_bits
                mask_bits <<= 1

            tts.append(
                torch.tensor(
                    [bit == "1" for bit in list("{:03b}".format(res << bits))],
                    device=self.device,
                    dtype=torch.bool,
                    requires_grad=False,
                ).repeat(repeats)
            )
        return torch.stack(tts)

    @staticmethod
    def _shaped_reward_function_const(
        nodes: torch.Tensor, num_inputs: torch.Tensor
    ) -> torch.Tensor:
        # least possible nodes for a graph n_ands = n_inputs - 1
        # a graph contains n_ands + n_inputs + 1 (constant node) nodes
        # Thus the least possible nodes for a graph is 2 *
        reward = torch.exp(num_inputs * 2 - nodes.shape[-2]).detach().to(nodes.device)
        return reward

    @staticmethod
    def _shaped_reward_function_noconst(
        nodes: torch.Tensor, num_inputs: torch.Tensor
    ) -> torch.Tensor:
        reward = (
            torch.exp(num_inputs * 2 - nodes.shape[-2] - 1).detach().to(nodes.device)
        )
        return reward

    @staticmethod
    def _simple_reward_function(
        nodes: torch.Tensor, num_inputs: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(1.0, device=nodes.device)

    def _set_seed(self, seed: int | None):
        return super()._set_seed(seed)  # type: ignore

    def num_nodes(self) -> int:
        return self.state.get("nodes").shape[-2]

    def n_ands(self) -> int:
        if self.const_node:
            return self.num_nodes() - self.state["num_inputs"].item() - 1
        return self.num_nodes() - self.state["num_inputs"].item()

    def set_state(self, state: TensorDict) -> None:
        self.state = state

    def copy_state(self) -> TensorDict:
        return self.state.clone(True)

    @staticmethod
    # @torch.jit.script
    def jitted_action_mask_const(
        nodes: torch.Tensor,
        edge_type: torch.Tensor,
        left_id: torch.Tensor,
        right_id: torch.Tensor,
    ) -> torch.Tensor:

        # mask = (
        #     torch.triu(
        #         torch.full((num_nodes, num_nodes),
        #                    -float('inf'),
        #                    dtype=torch.float32,
        #                    device=edge_type.device
        #         ), diagonal=0
        #     ).T).repeat(4, 1, 1)
        mask = torch.triu(
            torch.ones(
                (nodes.shape[-2], nodes.shape[-2]),
                dtype=torch.bool,
                device=edge_type.device,
            ),
            diagonal=1,
        ).repeat(4, 1, 1)
        # mask[edge_type, left_id, right_id] = -float('inf')
        # mask[:, 0, :] = False
        # mask[edge_type, left_id, right_id] = False
        mask = mask.index_put(
            (edge_type, left_id, right_id),
            torch.zeros(1, dtype=torch.bool, device=edge_type.device),
        )
        return mask.view(-1)

    @staticmethod
    # @torch.jit.script
    def jitted_action_mask_no_const(
        nodes: torch.Tensor,
        edge_type: torch.Tensor,
        left_id: torch.Tensor,
        right_id: torch.Tensor,
    ) -> torch.Tensor:

        # mask = (
        #     torch.triu(
        #         torch.full((num_nodes, num_nodes),
        #                    -float('inf'),
        #                    dtype=torch.float32,
        #                    device=edge_type.device
        #         ), diagonal=0
        #     ).T).repeat(4, 1, 1)
        mask = torch.triu(
            torch.ones(
                (nodes.shape[-2], nodes.shape[-2]),
                dtype=torch.bool,
                device=edge_type.device,
            ),
            diagonal=1,
        ).repeat(4, 1, 1)
        # mask[edge_type, left_id, right_id] = -float('inf')
        # print("edge_type", edge_type.shape)
        # print("left_id", left_id.shape)
        # print("right_id", right_id.shape)
        # mask[edge_type, left_id, right_id] = False
        mask = mask.index_put(
            (edge_type, left_id, right_id),
            torch.zeros(1, dtype=torch.bool, device=edge_type.device),
        )

        return mask.view(-1)

    def action_mask(self, state: TensorDict) -> torch.Tensor:
        # return torch.cond(
        #     self.const_node,
        #     self.jitted_action_mask_const,
        #     self.jitted_action_mask_no_const,
        #     (
        #         state.get("nodes"),
        #         state.get("edge_type"),
        #         state.get("left"),
        #         state.get("right")
        #     ),
        # )
        if self.const_node:
            return self.jitted_action_mask_const(
                state["nodes"],
                state["edge_type"],
                state["left"],
                state["right"],
            )
        return self.jitted_action_mask_no_const(
            state.get("nodes"),
            state.get("edge_type"),
            state.get("left"),
            state.get("right"),
        )

    def undo_action(self) -> None:
        self.state.set("nodes", self.state.get("nodes")[:-1])
        self.state.set("edge_type", self.state.get("edge_type")[:-1])
        self.state.set("left", self.state.get("left")[:-1])
        self.state.set("right", self.state.get("right")[:-1])
        self.state.set("done", torch.tensor(False))
        self.state.set("truncated", torch.tensor(False))
        self.state.set("reward", torch.tensor([0.0]))

    @staticmethod
    # @torch.jit.script
    def unravel_index(
        action: torch.LongTensor, num_nodes: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        # print("Action", action.dtype, action.shape, type(action))
        edge_type = torch.floor_divide(action, num_nodes**2)
        left_id = torch.remainder(action, num_nodes**2).div(
            num_nodes, rounding_mode="floor"
        )
        right_id = action - (edge_type * num_nodes**2 + left_id * num_nodes)
        return edge_type, left_id, right_id

    @staticmethod
    # @torch.jit.script
    def get_new_node(
        edge_type: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        # TODO: potentially stack all 4 versions and select index the one we want?

        # left = torch.cond(torch.eq(edge_type, 0) | torch.eq(edge_type, 1), ~left, left)
        # right = torch.cond(
        #     torch.eq(edge_type, 1) | torch.eq(edge_type, 3), ~right, right
        # )

        if edge_type == 0:
            new_node = left & right
        elif edge_type == 1:
            new_node = left & ~right
        elif edge_type == 2:
            new_node = ~left & right
        else:
            new_node = ~left & ~right

        return new_node