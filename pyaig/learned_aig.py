from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Iterator
from collections import deque
import networkx as nx
import torch
import random
import matplotlib.pyplot as plt

from . aig import AIG, _Node
from . aig_env import AIGEnv

class LearnedNode(_Node):
    """A class to represent a node in a learned AIG. It extends the `_Node` class from the `aig` module.
    It simplifies the representation of the original implementation and adds some useful methods and native truth table simulation with PyTorch.

    Returns:
        _type_: LearnedNode
    """    
    CONST0 = 0
    PI = 1
    # LATCH = 2
    AND = 3
    # BUFFER = 4
    PO = 5

    def __init__(
        self,
        node_type: int,
        node_id: int,
        left: LearnedNode | None = None,
        right: LearnedNode | None = None,
        left_edge_type: int | None = None,
        right_edge_type: int | None = None,
        truth_table: torch.Tensor | None = None,
    ):
        self._type = node_type
        self._left = left
        self._right = right
        self._node_id = node_id
        self._truth_table = truth_table
        self._left_edge_type = left_edge_type  # 1 or -1
        self._negated_left_edge = 0
        self._right_edge_type = right_edge_type  # 1 or -1
        self._negated_right_edge = 0
        self._fanout_type = {}
        self._fanout_id_to_object = {}
        self._level = 0

    @property
    def node_type(self) -> int:
        """Get the node type

        Returns:
            int: 0 for CONST0, 1 for PI, 3 for AND, 5 for PO
        """        
        return self._type

    @property
    def truth_table(self) -> torch.BoolTensor:
        """Returns the truth table associated with the node

        Returns:
            torch.BoolTensor: the truth table
        """        
        return self._truth_table

    @property
    def node_id(self) -> int:
        """Get the node id associated with the node in the AIG

        Returns:
            int: the node id
        """        
        return self._node_id

    @property
    def left(self) -> LearnedNode | None:
        """The left parent of the node. If the node is a PI, then the left parent is `None`

        Returns:
            LearnedNode | None: The left parent of the node
        """        
        return self._left

    @property
    def right(self) -> LearnedNode | None:
        """The right parent of the node. If the node is a PI, then the right parent is `None`

        Returns:
            LearnedNode | None: The right parent of the node
        """   
        return self._right

    @property
    def left_edge_type(self) -> int | None:
        """The type of edge connecting the left parent to the node. 1 for normal edge, -1 for negated edge.
        If the node is a PI, then the left edge type is `None`

        Returns:
            int | None: The type of edge connecting the left parent to the node
        """        
        return self._left_edge_type

    @property
    def right_edge_type(self) -> int | None:
        """The type of edge connecting the right parent to the node. 1 for normal edge, -1 for negated edge.
        If the node is a PI, then the right edge type is `None`

        Returns:
            int | None: The type of edge connecting the right parent to the node
        """  
        return self._right_edge_type

    @property
    def fanout_type(self) -> Dict[LearnedNode, int]:
        """It's a `dict` that associated  the edge type connecting the node to its fanouts (children).
        1 for normal edge, -1 for negated edge.

        Returns:
            Dict[LearnedNode, int]: The dictionary that contains the children and their edge types
        """
        return self._fanout_type

    @property
    def fanout_id_to_object(self) -> Dict[int, LearnedNode]:
        """It's a `dict` that associates the node id of the fanouts (children) to the respective node object without needing to use the AIG object.

        Returns:
            Dict[int, LearnedNode]: The dictionary that contains the node id of the children and their respective node
        """        
        return self._fanout_id_to_object

    @property
    def level(self) -> int:
        """This is the level of the node in the AIG. The level of a node is the maximum level of its parents plus 1.

        Returns:
            int: The level of the node
        """        
        return self._level

    @truth_table.setter
    def truth_table(self, truth_table: torch.BoolTensor) -> None:
        """The setter method for assigning a truth table to the node.

        Args:
            truth_table (torch.BoolTensor): The truth table to be assigned to the node
        """        
        self._truth_table = truth_table

    @node_id.setter
    def node_id(self, node_id: int) -> None:
        """The setter method for assigning a node id to the node.

        Args:
            node_id (int): The node id to be assigned to the node
        """        
        self._node_id = node_id

    @left.setter
    def left(self, left: LearnedNode) -> None:
        """The setter method for assigning a left parent to the node. 

        Args:
            left (LearnedNode): The node to be assigned as the left parent
        """        
        self._left = left
        if self._left.has_truth_table():
            self.calculate_truth_table()

    @right.setter
    def right(self, right: LearnedNode) -> None:
        """The setter method for assigning a right parent to the node

        Args:
            right (LearnedNode): The node to be assigned as the right parent
        """    
        self._right = right
        if self._right.has_truth_table():
            self.calculate_truth_table()

    @left_edge_type.setter
    def left_edge_type(self, left_edge_type: int) -> None:
        """The setter method for assigning the edge type connecting the left parent and the node.

        Args:
            left_edge_type (int): The edge type to be assigned. 1 for normal edge, -1 for negated edge.
        """   
        self._left_edge_type = left_edge_type
        if self._left.has_truth_table():
            self.calculate_truth_table()

    @right_edge_type.setter
    def right_edge_type(self, right_edge_type: int) -> None:
        """The setter method for assigning the edge type connecting the right parent and the node.

        Args:
            right_edge_type (int): The edge type to be assigned. 1 for normal edge, -1 for negated edge
        """        
        self._right_edge_type = right_edge_type
        if self._right.has_truth_table():
            self.calculate_truth_table()

    @level.setter
    def level(self, level: int) -> None:
        """The setter method for assigning the level of the node in the AIG. 

        Args:
            level (int): The level of the node
        """        
        self._level = level

    @staticmethod
    def make_po(
        node_id: int,
        input: LearnedNode | None = None,
        edge_type: int | None = None,
        truth_table: torch.BoolTensor | None = None,
    ) -> LearnedNode:
        """A static method to create a PO node

        Args:
            node_id (int): The node id of the PO node (usually negative)
            input (LearnedNode | None, optional): The node where the PO originates from. Defaults to None and can be set later.
            edge_type (int | None, optional): The type of edge to connect the input node and PO with. Defaults to None and can be set later.
            truth_table (torch.BoolTensor | None, optional): The truth table associated with the PO. Defaults to None and can be set later.

        Returns:
            LearnedNode: The newly created PO node
        """        
        return LearnedNode(
            LearnedNode.PO, node_id, input, input, edge_type, edge_type, truth_table
        )

    @staticmethod
    def make_pi(
        node_id: int, truth_table: torch.BoolTensor | None = None
    ) -> LearnedNode:
        """A static method to create a PI node

        Args:
            node_id (int): The node id of the PI node
            truth_table (torch.BoolTensor | None, optional): The truth table associated with the PI. Defaults to None.

        Returns:
            LearnedNode: The newly created PI node
        """        
        return LearnedNode(
            LearnedNode.PI, node_id, None, None, None, None, truth_table
        )

    @staticmethod
    def make_and(
        node_id: int,
        left: LearnedNode,
        right: LearnedNode,
        left_edge_type: int,
        right_edge_type: int,
    ) -> LearnedNode:
        """A static method to create an AND node

        Args:
            node_id (int): The node id of the AND node
            left (LearnedNode): The left parent of the AND node
            right (LearnedNode): The right parent of the AND node
            left_edge_type (int): The type of edge connecting the left parent to the AND node. 1 for normal edge, -1 for negated edge
            right_edge_type (int): The type of edge connecting the right parent to the AND node. 1 for normal edge, -1 for negated edge

        Returns:
            LearnedNode: The newly created AND node
        """        
        node = LearnedNode(
            LearnedNode.AND,
            node_id,
            left,
            right,
            left_edge_type,
            right_edge_type,
            None,
        )
        if left.has_truth_table() and right.has_truth_table():
            node.calculate_truth_table()
        node.update_level()
        return node

    @staticmethod
    def make_const0(truth_table_size: int | None = None) -> LearnedNode:
        """A static method to create a CONST0 node

        Args:
            truth_table_size (int | None, optional): The number of entries for the truth table. Defaults to None.

        Returns:
            LearnedNode: The newly created CONST0 node
        """        
        truth_table = None
        if truth_table_size != None:
            truth_table = torch.zeros(truth_table_size, dtype=bool)
        return LearnedNode(
            LearnedNode.CONST0, 0, None, None, None, None, truth_table
        )

    def set_left_edge(self, left: LearnedNode, left_edge_type: int) -> None:
        """A safe method for setting the left parent of the node and the edge type connecting them.
        It also updates the level of the node and ensures that the node id of the left parent is less than the node id of the right parent.

        Args:
            left (LearnedNode): The parent node to be set as the left parent
            left_edge_type (int): The type of edge connecting the left parent to the node. 1 for normal edge, -1 for negated edge
        """        
        self.left = left
        self.left_edge_type = left_edge_type
        self.swap_edges()
        if self._type == self.PO:
            self.right = left
            self.right_edge_type = left_edge_type
        self.update_level()

    def set_right_edge(self, right: LearnedNode, right_edge_type: int) -> None:
        """A safe method for setting the right parent of the node and the edge type connecting them.
        It also updates the level of the node and ensures that the node id of the right parent is greater than the node id of the left parent.

        Args:
            right (LearnedNode): The parent node to be set as the right parent
            right_edge_type (int): The type of edge connecting the right parent to the node. 1 for normal edge, -1 for negated edge
        """        
        self._right = right
        self._right_edge_type = right_edge_type
        self.swap_edges()
        if self._type == self.PO:
            self._left = right
            self._left_edge_type = right_edge_type
        if right.has_truth_table():
            self.calculate_truth_table()
        self.update_level()

    def update_edge_type(self, node: LearnedNode, edge_type: int) -> None:
        """Given the parent node and the edge type, this method updates the edge type connecting the parent node to the current node.

        Args:
            node (LearnedNode): The parent node either left or right parent
            edge_type (int): The new edge type to be assigned
        """        
        if node == self.left:
            self.left_edge_type = edge_type
        elif node == self.right:
            self.right_edge_type = edge_type
        elif node in self._fanout_type:
            self._fanout_type[node] = edge_type

    def swap_edges(self) -> None:
        """This method ensures that the node id of the left parent is less than the node id of the right parent.
        If the node id of the left parent is greater than the node id of the right parent, then the left and right parents are swapped.
        """        
        if (
            self.right != None
            and self.left != None
            and self.left.node_id
            and self.left.node_id > self.right.node_id
        ):
            self.left, self.right = self.right, self.left
            self.left_edge_type, self.right_edge_type = (
                self.right_edge_type,
                self.left_edge_type,
            )

    def get_left(self) -> int | None:
        """A method to get the node id of the left parent of the node.
        If the node is a PI, then the left parent is `None`

        Returns:
            int | None: The node id of the left parent
        """        
        if self.left != None:
            return self.left.node_id

    def get_right(self) -> int | None:
        """A method to get the node id of the right parent of the node.
        If the node is a PI, then the right parent is `None`

        Returns:
            int | None: The node id of the right parent
        """        
        if self.right != None:
            return self.right.node_id

    def add_fanout(self, target: LearnedNode, edge_type: int) -> None:
        """A method to add a fanout to the node. It associates the target node with the edge type connecting the target node to the current node.

        Args:
            target (LearnedNode): The node to be added as a fanout
            edge_type (int): The type of edge connecting the target node to the current node. 1 for normal edge, -1 for negated edge
        """        
        self._fanout_type[target] = edge_type
        self._fanout_id_to_object[target.node_id] = target

    def fanout_size(self) -> int:
        """A method to get the number of fanouts of the node

        Returns:
            int: The number of fanouts of the node
        """        
        return len(self._fanout_id_to_object)

    def delete_fanout(self, node: LearnedNode | int) -> None:
        """A method to delete a fanout from the node. It removes the target node from the fanouts of the current node.

        Args:
            node (LearnedNode | int): The node to be removed from the fanouts of the current node
        """        
        node_id = 0
        if isinstance(node, int):
            node_id = node
            node = self._fanout_id_to_object[node_id]
        else:
            node_id = node.node_id
        del self._fanout_id_to_object[node_id]
        del self._fanout_type[node]

    def has_truth_table(self) -> bool:
        """Checks if the node has a truth table. 

        Returns:
            bool: True if the node has a truth table assigned, otherwise False
        """        
        return self._truth_table != None

    def calculate_truth_table(self, force: bool = False) -> None:
        """A method to calculate the truth table of the node.
        If the node is a PI, then the truth table has to be assigned to the node.
        If the node is a CONST0, then the truth table is all zeros.
        If the node is an AND node, then the truth table is calculated based on the left and right parents.
        If the node is a PO node, then the truth table is calculated based on the input node.

        Args:
            force (bool, optional): It forces the recomputation of the truth table. Defaults to False.
        """        
        if not self.is_pi():
            if not self.left.has_truth_table():
                self.left.calculate_truth_table(force=force)
            if not self.right.has_truth_table():
                self.right.calculate_truth_table(force=force)
            if (
                self.left == None
                or self.right == None
                or self.right_edge_type == None
                or self.left_edge_type == None
            ):
                self._truth_table = None
            elif self._type == LearnedNode.PO and force:
                if self.left_edge_type == -1:
                    self._truth_table = ~self.left.truth_table
                else:
                    self._truth_table = self.left.truth_table
            else:
                if self.left_edge_type == -1 and self.right_edge_type == -1:
                    self._truth_table = ~self.left.truth_table & ~self.right.truth_table
                elif self.left_edge_type == -1 and self.right_edge_type == 1:
                    self._truth_table = ~self.left.truth_table & self.right.truth_table
                elif self.left_edge_type == 1 and self.right_edge_type == -1:
                    self._truth_table = self.left.truth_table & ~self.right.truth_table
                else:
                    self._truth_table = self.left.truth_table & self.right.truth_table

    def update_level(self):
        """A method to update the level of the node in the AIG.
        The level of a node is the maximum level of its parents plus 1.
        """        
        if self.left != None:
            self._level = self.left.level + 1
        if self.right != None and self.right.level + 1 > self._level:
            self._level = self.right.level + 1

    def __getitem__(self, node: LearnedNode | int) -> int | LearnedNode:
        """The method returns either the LearnedNode object associated with the node id or the edge type connecting the node to the fanout when given a LearnedNode.

        Args:
            node (LearnedNode | int): The node id or the LearnedNode object

        Returns:
            _type_: int | LearnedNode: The edge type connecting the node to the fanout or the LearnedNode object associated with the node id
        """        
        if isinstance(node, int):
            return self._fanout_id_to_object[node]
        else:
            return self._fanout_type[node]

    def __setitem__(self, node: LearnedNode, edge_type: int) -> None:
        """The method sets the edge type connecting the node to the fanout (child)

        Args:
            node (LearnedNode): The LearnedNode representing the fanout (child)
            edge_type (int): The edge type connecting the node to the fanout
        """        
        self._fanout_type[node] = edge_type

    def __repr__(self) -> str:
        """The method returns a string representation of the node

        Returns:
            str: The string representation of the node
        """        
        if self._type == LearnedNode.AND:
            type = "AND"
        # elif self._type==_Node.BUFFER:
        #     type = "BUFFER"
        elif self._type == _Node.CONST0:
            type = "CONST0"
        # elif self._type==_Node.LATCH:
        #     type = "LATCH"
        elif self._type == LearnedNode.PI:
            type = "PI"
            return "<pyaig.aig.LearnedNode _type=%s, _node_id=%s>" % (
                type,
                str(self.node_id),
            )
        elif self._type == LearnedNode.PO:
            type = "PO"
        else:
            type = "UNKNOWN"
        return (
            "<pyaig.aig.LearnedNode _type=%s, _node_id=%s, _left=%s, _right=%s>"
            % (
                type,
                str(self.node_id),
                str(self.left_edge_type * self.left.node_id),
                str(self.right_edge_type * self.right.node_id),
            )
        )

    def __iter__(self) -> Iterator[LearnedNode]:
        """The method returns an iterator for the node

        Returns:
            _type_: _description_

        Yields:
            Iterator[LearnedNode]: _description_
        """        
        self._keys = list(self._fanout_id_to_object.keys())
        self._idx = 0
        return self

    def __next__(self) -> LearnedNode:
        """The method returns the next node in the iterator

        Raises:
            StopIteration: _description_

        Returns:
            LearnedNode: _description_
        """        
        if self._idx == len(self._keys):
            raise StopIteration
        self._idx += 1
        return self._keys[self._idx - 1]


class LearnedAIG(AIG):
    def __init__(
        self,
        n_pis: int,
        n_pos: int,
        truth_tables: Optional[List[torch.Tensor] | torch.Tensor],
        truth_table_size: Optional[int] = None,
        name: Optional[str] = None,
        pi_names: Optional[List[str]] = None,
        po_names: Optional[List[str]] = None,
        skip_truth_tables: bool = False,
    ) -> None:

        super().__init__(name)
        self._nodes: List[LearnedNode] = []
        self._pis: List[LearnedNode] = []
        self._pos: List[LearnedNode] = []
        self._id_to_object: Dict[int, LearnedNode] = {}
        self._node_truth_tables: List[torch.Tensor] = []
        self._po_truth_tables: List[torch.Tensor] = []
        self._next_available_node_id: int = 0
        self._instantiated_truth_tables: bool

        if pi_names != None:
            assert n_pis == len(pi_names)
        if po_names != None:
            assert n_pos == len(pi_names)

        self._instantiated_truth_tables = not skip_truth_tables

        self._truth_table_size = 2 ** (n_pis)

        # Create the const node
        self.__create_const()

        # Create the PIs
        for i in range(n_pis):
            if pi_names == None:
                self.__create_pi(name=(i + 1))
            else:
                self.__create_pi(name=pi_names[i])

        # Assigned truth tables to PIs if necessary
        if self._instantiated_truth_tables:
            self.assign_pi_tts()

        if isinstance(truth_tables, torch.Tensor):
            truth_tables = [truth_tables]
        elif not self._instantiated_truth_tables or truth_tables == None:
            truth_tables = [None] * n_pos

        # Create the POs
        for i in range(n_pos):
            if po_names == None:
                self.__create_po(name=-(i + 1), truth_table=truth_tables[i])
            else:
                self.__create_po(name=po_names[i], truth_table=truth_tables[i])

    def __create_pi(
        self, name: int | str, truth_table: torch.Tensor | int | None = None
    ) -> LearnedNode:
        pi_id = self._next_available_node_id
        self._next_available_node_id += 1
        if truth_table != None and self._truth_table_size != None:
            truth_table = self.create_truth_table(
                bin=truth_table, bits=self._truth_table_size
            )
        node = LearnedNode.make_pi(node_id=pi_id, truth_table=truth_table)
        self._id_to_object[pi_id] = node
        self._nodes.append(node)
        self._pis.append(node)
        self.set_name(pi_id, name)

        return node

    def __create_po(
        self, name: int | str, truth_table: torch.Tensor | None
    ) -> LearnedNode:
        po_id = -(len(self._pos) + 1)
        node = LearnedNode.make_po(
            node_id=po_id, input=None, edge_type=None, truth_table=truth_table
        )
        self._id_to_object[po_id] = node
        self._pos.append(node)
        self.set_po_name(po_id, name)

        return node

    def __create_const(self) -> LearnedNode:
        pi_id = self._next_available_node_id
        self._next_available_node_id += 1
        size = None
        if self._instantiated_truth_tables:
            size = self._truth_table_size
        node = LearnedNode.make_const0(size)
        self._id_to_object[pi_id] = node
        self._nodes.append(node)
        return node

    def create_and(
        self,
        left: LearnedNode | int,
        right: LearnedNode | int,
        left_edge_type: int,
        right_edge_type: int,
    ) -> LearnedNode:
        if isinstance(left, int):
            left = self._id_to_object[left]
        if isinstance(right, int):
            right = self._id_to_object[right]

        if left.node_id > right.node_id:
            left, right = right, left
            left_edge_type, right_edge_type = right_edge_type, left_edge_type

        key = (LearnedNode.AND, id(left), id(right), left_edge_type, right_edge_type)

        if key in self._strash:
            return self._strash[key]
        node_id = self._next_available_node_id
        self._next_available_node_id += 1
        node = LearnedNode.make_and(
            node_id, left, right, left_edge_type, right_edge_type
        )
        self._nodes.append(node)
        self._strash[key] = node
        self._id_to_object[node_id] = node
        left.add_fanout(node, left_edge_type)
        right.add_fanout(node, right_edge_type)

        return node

    def set_left_edge(
        self, source: LearnedNode | int, target: LearnedNode | int, edge_type: int
    ) -> None:
        if isinstance(source, int):
            source = self._id_to_object[source]
        if isinstance(target, int):
            target = self._id_to_object[target]
        target.set_left_edge(source, edge_type)
        source.add_fanout(target, edge_type)

    def set_right_edge(
        self, source: LearnedNode | int, target: LearnedNode | int, edge_type: int
    ) -> None:
        if isinstance(source, int):
            source = self._id_to_object[source]
        if isinstance(target, int):
            target = self._id_to_object[target]
        target.set_right_edge(source, edge_type)
        source.add_fanout(target, edge_type)

    def set_po_edge(
        self, source: LearnedNode | int, po: LearnedNode | int, edge_type: int
    ) -> None:
        if isinstance(source, int):
            source = self._id_to_object[source]
        if isinstance(po, int):
            po = self._id_to_object[po]
        po.set_right_edge(source, edge_type)
        source.add_fanout(po, edge_type)

    def is_negated(self, idx) -> bool:
        return True

    def instantiate_truth_tables(self) -> None:
        self.assign_const_tts()
        self.assign_pi_tts()

        for node in self._nodes:
            if node.is_and():
                node.calculate_truth_table(force=True)
        for po in self._pos:
            po.calculate_truth_table(force=True)
        self._instantiated_truth_tables = True

    def assign_pi_tts(self) -> None:
        for i in range(len(self._pis)):
            bits = 1 << i
            res = ~(~0 << bits)
            mask_bits = bits << 1
            for _ in range(len(self._pis) - (i + 1)):

                res |= res << mask_bits
                mask_bits <<= 1
            # self.cofactor_masks[0].append( res )
            # self.cofactor_masks[1].append( res << bits )
            self._pis[i].truth_table = self.create_truth_table(bin=res << bits)
            # self._pis[i].truth_table = self.create_truth_table(bin=res << bits, bits=self._truth_table_size)
        # self.all_consts = [ _truth_table(self, self.mask*c) for c in (0, 1) ]
        # self.all_vars = [ [_truth_table(self, self.cofactor_masks[c][i]) for i in range(N)] for c in (0, 1) ]

    def assign_const_tts(self) -> None:
        self._nodes[0].truth_table = torch.zeros(
            self._truth_table_size, dtype=torch.bool
        )

    def create_and_nodes_from_actions(
        self,
        actions: torch.Tensor | list[list[int]],
        const_node: bool = False,
    ) -> None:
        if isinstance(actions, torch.Tensor):
            # print(actions.shape)
            # actions = actions.reshape(actions.shape[1:])
            actions = actions.T.tolist()
            # print(actions)
        assert isinstance(actions, list)
        edges = deque(actions)

        while len(edges) > 0:
            edge_type, left, right = edges.popleft()
            if not const_node:
                left += 1
                right += 1
            if left in self._id_to_object and right in self._id_to_object:
                left_edge, right_edge = self.edge_type_decoder(edge_type)
                self.create_and(left, right, left_edge, right_edge)
            else:
                edges.append([edge_type, left, right])

            self.create_and(left, right, left_edge, right_edge)

    @staticmethod
    def from_adj_matrix(
        n_pis: int,
        adj_matrix: torch.Tensor,
        n_pos: int = 1,
        truth_tables: list[torch.Tensor] | torch.Tensor | None = None,
        pi_names: list[str] | None = None,
        po_names: list[str] | None = None,
        name: str | None = None,
    ) -> LearnedAIG:

        aig = LearnedAIG(
            n_pis=n_pis,
            n_pos=n_pos,
            truth_tables=truth_tables,
            name=name,
            pi_names=pi_names,
            po_names=po_names,
            skip_truth_tables=True,
        )
        aig.assign_const_tts()
        aig.assign_pi_tts()
        aig.create_and_nodes_from_actions(adj_matrix.nonzero().tolist())
        potential_pos = []
        for node in aig._nodes[1:]:
            node.calculate_truth_table(force=True)
            if node.fanout_size() == 0:
                potential_pos.append(node.node_id)
        assert len(potential_pos) == n_pos

        if truth_tables != None:
            if isinstance(truth_tables, torch.Tensor):
                truth_tables = [truth_tables]

            for node in potential_pos:
                for po in aig._pos:
                    if po.truth_table == node.truth_table:
                        aig.set_po_edge(node, po, 1)
                    elif po.truth_table == ~node.truth_table:
                        aig.set_po_edge(node, po, -1)
        return aig

    @staticmethod
    def from_aig_env(aig_env: AIGEnv) -> LearnedAIG:
        aig = LearnedAIG(
            n_pis=int(aig_env.state["num_inputs"].item()),
            n_pos=int(aig_env.n_pos.item()),
            truth_tables=[aig_env.state["target"]],
            skip_truth_tables=True,
        )
        aig.assign_const_tts()
        aig.assign_pi_tts()
        aig[-1].truth_table = aig_env.state["target"]
        actions = torch.stack(
            [aig_env.state["edge_type"], aig_env.state["left"], aig_env.state["right"]],
            dim=0,
        )
        aig.create_and_nodes_from_actions(actions.int(), aig_env.const_node)

        potential_pos = []
        for node in aig._nodes[1:]:
            node.calculate_truth_table(force=True)
            if node.fanout_size() == 0:
                potential_pos.append(node.node_id)
        # assert len(potential_pos) == aig.n_pos

        for n in potential_pos:
            for po in aig._pos:
                node = aig[n]
                if torch.equal(po.truth_table.view(-1), node.truth_table.view(-1)):
                    aig.set_po_edge(node, po, 1)
                elif torch.equal(po.truth_table.view(-1), ~node.truth_table.view(-1)):  # type: ignore
                    aig.set_po_edge(node, po, -1)

        return aig

    @staticmethod
    def edge_type_decoder(edge_type: int) -> Tuple[int, int]:
        match edge_type:
            case 0:
                return (1, 1)
            case 1:
                return (1, -1)
            case 2:
                return (-1, 1)
            case _:
                return (-1, -1)

    @classmethod
    def create_truth_table(cls, bin: int | torch.Tensor | None) -> torch.Tensor:
        if type(bin) == torch.Tensor:
            return bin
        elif bin == None:
            return None
        # np_array = (((bin & (1 << np.arange(bits, dtype=np.uint64))[::-1])) > 0).astype(bool) # Creates a np.array of bit array representing an int
        return torch.tensor([bit == "1" for bit in list("{:03b}".format(bin))])

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()

        # G.add_node(0, node_type="CONST", node_id=0)
        for node in self._nodes:
            if node.node_type == LearnedNode.AND:
                G.add_node(node.node_id, node_type="AND", node_id=node.node_id)
            elif node.node_type == LearnedNode.PI:
                G.add_node(node.node_id, node_type="PI", node_id=node.node_id)
            else:
                G.add_node(node.node_id, node_type="CONST", node_id=node.node_id)
        for po in self._pos:
            G.add_node(po.node_id, node_type="PO", node_id=po.node_id)

        for node in self._nodes:
            if (
                node.node_type != LearnedNode.PI
                and node.node_type != LearnedNode.CONST0
            ):
                G.add_edge(
                    node.left.node_id, node.node_id, edge_type=node.left_edge_type
                )
                G.add_edge(
                    node.right.node_id, node.node_id, edge_type=node.right_edge_type
                )
        # for node in list(G.nodes):
        #     if "node_type" not in G.nodes[node]:
        #         G.nodes[node]["node_type"] = "AND"
        #         # G.nodes[node]["node_id"] = node.node_id

        for po in self._pos:
            if po.left != None:
                G.add_edge(po.left.node_id, po.node_id, edge_type=po.left_edge_type)
        return G

    def draw(self) -> None:
        G = self.to_networkx()

        pis = [
            k for k, v in nx.get_node_attributes(G, "node_type").items() if v == "PI"
        ]
        pos = [
            k for k, v in nx.get_node_attributes(G, "node_type").items() if v == "PO"
        ]
        const = [
            k for k, v in nx.get_node_attributes(G, "node_type").items() if v == "CONST"
        ]
        ands = list(set(G.nodes()) - set(pis) - set(pos) - set(const))

        normal = [
            k for k, v in nx.get_edge_attributes(G, "edge_type").items() if v == 1
        ]
        negated = [
            k for k, v in nx.get_edge_attributes(G, "edge_type").items() if v == -1
        ]

        position = nx.nx_agraph.pygraphviz_layout(G, prog="dot")

        # Flip the y-coordinates to reverse the graph vertically
        for node in position:
            x, y = position[node]
            position[node] = (x, -y)  # Negate the y-coordinate to flip vertically # type: ignore

        labels = {}
        for n, data in G.nodes(data=True):
            labels[n] = data.get("node_id")
        
        # Modern color palette
        pi_color = "#4CAF50"  # Green
        po_color = "#FF5252"  # Red
        and_color = "#2196F3"   # Blue
        const_color = "#FFC107" # Amber

        plt.figure(figsize=(12, 8))

        # Draw nodes with different shapes, colors, and sizes
        nx.draw_networkx_nodes(
            G,
            position,
            nodelist=const,
            node_color=const_color,
            node_shape="s",
            label="CONST",
            node_size=700,
            alpha=0.8
        )
        nx.draw_networkx_nodes(
            G, 
            position, 
            nodelist=pis, 
            node_color=pi_color, 
            node_shape="o", 
            label="LEAF",
            node_size=700,
            alpha=0.8
        )
        nx.draw_networkx_nodes(
            G, 
            position, 
            nodelist=pos, 
            node_color=po_color, 
            node_shape="^", 
            label="ROOT",
            node_size=900,
            alpha=0.8
        )
        nx.draw_networkx_nodes(
            G, 
            position, 
            nodelist=ands, 
            node_color=and_color, 
            node_shape="d", 
            label="CUT",
            node_size=700,
            alpha=0.8
        )

        # Draw labels with contrasting colors for better visibility
        nx.draw_networkx_labels(G, position, labels=labels, font_weight='bold', font_size=10)

        # Draw edges with different styles
        nx.draw_networkx_edges(
            G, 
            position, 
            edgelist=normal, 
            width=1.5, 
            edge_color='#555555',
            arrows=True, 
            arrowsize=15, 
            arrowstyle='->'
        )
        nx.draw_networkx_edges(
            G, 
            position, 
            edgelist=negated, 
            style="--", 
            width=1.5, 
            edge_color='#555555',
            arrows=True, 
            arrowsize=15, 
            arrowstyle='->'
        )

        # Fix the legend with smaller markers and better spacing
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=const_color, # type: ignore
                      markersize=10, label='WINDOW', alpha=0.8),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pi_color, # type: ignore
                      markersize=10, label='LEAF', alpha=0.8),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=po_color, # type: ignore
                      markersize=10, label='ROOT', alpha=0.8),
            plt.Line2D([0], [0], marker='d', color='w', markerfacecolor=and_color, # type: ignore
                      markersize=10, label='CUT', alpha=0.8)
        ]
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, 
                  framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
        
        # Remove axis
        plt.axis('off')
        
        # Add a title
        plt.title('Circuit Graph Visualization', fontsize=16)
        
        plt.tight_layout()
        plt.show()

    def set_name(self, node_id: int, name: int | str) -> None:
        # assert name not in self._name_to_id
        assert node_id not in self._id_to_name

        if not isinstance(name, str):
            name = str(name)

        self._name_to_id[name] = node_id
        self._id_to_name[node_id] = name

    def get_name(self, node: int | LearnedNode) -> str:
        if isinstance(node, LearnedNode):
            return self._id_to_name[node.node_id]
        return self._id_to_name[node]

    def set_po_name(self, po_id: int, name: int | str) -> None:
        assert name not in self._name_to_po
        assert po_id not in self._po_to_name

        self._name_to_po[name] = po_id
        self._po_to_name[po_id] = name

    def n_ands(self):
        """Returns the number of AND gates excluding the PIs and CONST node

        Returns:
            int: number of AND nodes
        """
        return len(self._nodes) - 1 - self.n_pis()

    def get_pos(self) -> tuple[int, int, int]:
        return ((po.node_id, po.left.node_id, po.node_type) for po in self._pos)

    def __getitem__(self, node_id: int) -> LearnedNode:
        return self._id_to_object[node_id]

    def __iter__(self) -> Iterator[LearnedNode]:
        self._idx = 0
        return self

    def __next__(self) -> LearnedNode:
        if self._idx == len(self._nodes):
            raise StopIteration
        self._idx += 1
        return self._nodes[self._idx - 1]

    @staticmethod
    def read_aig(path: str, skip_truth_tables: bool = True):
        old_aig = AIG()
        fin = open(path, "rb")

        header = fin.readline().split()
        assert header[0] == b"aig"

        args = [int(t) for t in header[1:]]
        (M, I, L, O, A) = args[:5]

        B = args[5] if len(args) > 5 else 0
        C = args[6] if len(args) > 6 else 0
        J = args[7] if len(args) > 7 else 0
        F = args[8] if len(args) > 8 else 0

        if I > 16:
            skip_truth_tables = True

        # print("Num PIs:", I, "-- Num POS:", O, "Total:", I+O)
        new_aig = LearnedAIG(
            n_pis=I,
            n_pos=O,
            truth_tables=None,
            truth_table_size=None,
            pi_names=None,
            po_names=None,
            skip_truth_tables=skip_truth_tables,
        )
        # print(len(new_aig._id_to_object))

        vars = []
        nexts = []

        pos_output = []
        pos_bad_states = []
        pos_constraint = []
        pos_justice = []
        pos_fairness = []

        old_to_new_id = {}
        old_to_new_id[0] = 0

        vars.append(old_aig.get_const0())
        # print("Only const in vars", vars)

        for i in range(I):
            vars.append(old_aig.create_pi())
            old_to_new_id[vars[-1]] = i + 1

        def parse_latch(line):
            tokens = line.strip().split(b" ")
            next = int(tokens[0])
            init = 0
            if len(tokens) == 2:
                if tokens[1] == "0":
                    init = LearnedAIG.INIT_ZERO
                if tokens[1] == "1":
                    init = LearnedAIG.INIT_ONE
                else:
                    init = LearnedAIG.INIT_NONDET
            return (next, init)

        for i in range(L):  # Obsolete
            vars.append(old_aig.create_latch())
            nexts.append(parse_latch(fin.readline()))

        for i in range(O):  # Fix
            pos_output.append(int(fin.readline()))

        for i in range(B):  # Obsolete
            pos_bad_states.append(int(fin.readline()))

        for i in range(C):  # Obsolete
            pos_constraint.append(int(fin.readline()))

        n_j_pos = []

        for i in range(J):  # Obsolete
            n_j_pos.append(int(fin.readline()))

        for n in n_j_pos:  # Obsolete
            pos = []
            for i in range(n):
                pos.append(int(fin.readline()))
            pos_justice.append(pos)

        for i in range(F):  # Obsolete
            pos_fairness.append(int(fin.readline()))

        def decode():
            i = 0
            res = 0
            while True:
                c = ord(fin.read(1))
                res |= (c & 0x7F) << (7 * i)
                if (c & 0x80) == 0:
                    break
                i += 1
            return res

        def lit(x):
            return old_aig.negate_if(vars[x >> 1], x & 0x1)

        edge_type_map = {1: -1, 0: 1}

        for i in range(I + L + 1, I + L + A + 1):
            d1 = decode()
            d2 = decode()
            g = i << 1
            # actual id of parent 1 is g-d1
            # actual id of parent 2 is g-d1-2
            # position of parent in the array is vars is p_id>>1 (divide by 2)
            # If the number is odd then the number is negated x&0x1, to undo and find the true parent p_id^1
            p_id1 = g - d1
            p_id2 = g - d1 - d2
            vars.append(old_aig.create_and(lit(p_id1), lit(p_id2)))
            new_node = new_aig.create_and(
                old_to_new_id[old_aig.get_positive(lit(p_id1))],
                old_to_new_id[old_aig.get_positive(lit(p_id2))],
                edge_type_map[lit(p_id1) & 1],
                edge_type_map[lit(p_id2) & 1],
            )
            old_to_new_id[vars[-1]] = new_node.node_id

        for l, v in enumerate(range(I + 1, I + L + 1)):  # Obsolete
            old_aig.set_init(vars[v], nexts[l][1])
            old_aig.set_next(vars[v], lit(nexts[l][0]))

        output_pos = []

        for i in range(len(pos_output)):
            po = pos_output[i]
            output_pos.append(old_aig.create_po(lit(po), po_type=AIG.OUTPUT))
            new_po_id = old_to_new_id[old_aig.get_positive(lit(po))]
            edge_type = edge_type_map[lit(po) & 1]
            new_aig.set_po_edge(new_po_id, -(i + 1), edge_type)
            if not skip_truth_tables:
                new_aig[-(i + 1)].calculate_truth_table(force=True)

        bad_states_pos = []

        for i in range(len(pos_bad_states)):
            bad_states_pos.append(old_aig.create_po(lit(po), po_type=AIG.BAD_STATES))

        constraint_pos = []

        for po in pos_constraint:
            constraint_pos.append(old_aig.create_po(lit(po), po_type=AIG.CONSTRAINT))

        for pos in pos_justice:
            po_ids = [old_aig.create_po(lit(po), po_type=AIG.JUSTICE) for po in pos]
            old_aig.create_justice(po_ids)

        fairness_pos = []

        for po in pos_fairness:
            fairness_pos.append(old_aig.create_po(lit(po), po_type=AIG.FAIRNESS))

        # names = set()
        # po_names = set()

        # for line in fin:
        #     m = re.match( b'i(\\d+) (.*)', line )
        #     if m:
        #         if m.group(2) not in names:
        #             aig.set_name( vars[int(m.group(1))+1], m.group(2))
        #             names.add(m.group(2))
        #         continue

        #     m = re.match( b'l(\\d+) (.*)', line )
        #     if m:
        #         if m.group(2) not in names:
        #             aig.set_name( vars[I+int(m.group(1))+1], m.group(2))
        #             names.add(m.group(2))
        #         continue

        #     m = re.match( b'o(\\d+) (.*)', line )
        #     if m:
        #         if m.group(2) not in po_names:
        #             aig.set_po_name( output_pos[int(m.group(1))], m.group(2))
        #             po_names.add(m.group(2))
        #         continue

        #     m = re.match( b'b(\\d+) (.*)', line )
        #     if m:
        #         if m.group(2) not in po_names:
        #             aig.set_po_name( bad_states_pos[int(m.group(1))], m.group(2))
        #             po_names.add(m.group(2))
        #         continue

        #     m = re.match( b'c(\\d+) (.*)', line )
        #     if m:
        #         if m.group(2) not in po_names:
        #             aig.set_po_name( constraint_pos[int(m.group(1))], m.group(2))
        #             po_names.add(m.group(2))
        #         continue

        #     m = re.match( b'f(\\d+) (.*)', line )
        #     if m:
        #         if m.group(2) not in po_names:
        #             aig.set_po_name( fairness_pos[int(m.group(1))], m.group(2))
        #             po_names.add(m.group(2))
        #         continue

        return new_aig

    def write_aig(self, path) -> None:

        fout = open(path, "wb")

        map_aiger = {}

        aiger_i = 0

        map_aiger[0] = aiger_i
        aiger_i += 1
        _bytes = bytearray()

        for pi in self._pis:
            map_aiger[pi.node_id] = aiger_i << 1
            aiger_i += 1

        for l in self.get_latches():  # Obsolete
            map_aiger[l] = aiger_i << 1
            aiger_i += 1

        # for g in self.get_nonterminals(): #and gates and buffers
        #     map_aiger[ g ] = (aiger_i<<1)
        #     aiger_i += 1

        for n in self._nodes:  # and gates and buffers
            if n.node_type == LearnedNode.AND:
                map_aiger[n.node_id] = aiger_i << 1
                aiger_i += 1

        def aiger_lit(aig_lit):

            lit_pos = self.get_positive(aig_lit)
            lit = map_aiger[lit_pos]

            if self.is_negated(aig_lit):
                return lit + 1
            else:
                return lit

        def _encode(x):
            while (x & ~0x7F) > 0:
                _bytes.append((x & 0x7F) | 0x80)
                x >>= 7
            _bytes.append(x)

        I = self.n_pis()
        L = self.n_latches()
        # O = self.n_pos_by_type(LearnedAIG.OUTPUT)
        O = len(self._pos)
        A = self.n_nonterminals()
        B = self.n_pos_by_type(LearnedAIG.BAD_STATES)
        C = self.n_pos_by_type(LearnedAIG.CONSTRAINT)
        J = self.n_justice()
        F = self.n_pos_by_type(LearnedAIG.FAIRNESS)

        M = I + L + A
        _bytes.extend(b"aig %d %d %d %d %d" % (M, I, L, O, A))

        if B + C + J + F > 0:
            _bytes.extend(b" %d" % B)

        if C + J + F > 0:
            _bytes.extend(b" %d" % C)

        if J + F > 0:
            _bytes.extend(b" %d" % J)

        if F > 0:
            _bytes.extend(b" %d" % F)

        _bytes.extend(b"\n")

        _next = (I + 1) << 1
        # writer = _aiger_writer(
        #     self.n_pis(),
        #     self.n_latches(),
        #     self.n_pos_by_type(LearnedAIG.OUTPUT),
        #     self.n_nonterminals(),
        #     self.n_pos_by_type(LearnedAIG.BAD_STATES),
        #     self.n_pos_by_type(LearnedAIG.CONSTRAINT),
        #     self.n_justice(),
        #     self.n_pos_by_type(LearnedAIG.FAIRNESS),
        #     )

        # writer.write_inputs()

        # for l in self.get_latches(): #Obsolete
        #     writer.write_latch(aiger_lit(self.get_next(l)), self.get_init(l))

        for po in self._pos:
            po_id_source = po.left.node_id
            new_po_id_source = map_aiger[po_id_source]
            if po.left_edge_type == -1:
                new_po_id_source += 1
            _bytes.extend(b"%d\n" % new_po_id_source)

        # for po in self.get_po_fanins_by_type(LearnedAIG.OUTPUT):
        #     writer.write_po(aiger_lit(po))

        # for po in self.get_po_fanins_by_type(LearnedAIG.BAD_STATES): #Obsolete
        #     writer.write_po(aiger_lit(po))

        # for po in self.get_po_fanins_by_type(LearnedAIG.CONSTRAINT): #Obsolete
        #     writer.write_po(aiger_lit(po))

        # for _, j_pos in self.get_justice_properties(): #Obsolete
        #     writer.write_justice_header(j_pos)

        # for _, j_pos in self.get_justice_properties(): #Obsolete
        #     for po_id in j_pos:
        #         writer.write_po( aiger_lit( self.get_po_fanin(po_id) ) )

        # for po in self.get_po_fanins_by_type(LearnedAIG.FAIRNESS): #Obsolete
        #     writer.write_po(aiger_lit(po))

        # for g in self.get_nonterminals(): #These are the ids of the nodes
        # for n in self_nodes
        #     n = self.deref(g) #This gets the position of the node in the array
        #     if n.is_buffer(): #Obsolete
        #         al = ar = aiger_lit( n.get_buf_in() )
        #     else:
        #         al = map_aiger[n.left.node_id]
        #         ar = map_aiger[n.right.node_id]

        #         if n.left_edge_type == -1:
        #             al += 1
        #         if n.right_edge_type == -1:
        #             ar += 1
        #         # al = aiger_lit(n.get_left())
        #         # ar = aiger_lit(n.get_right())
        #     writer.write_and(al, ar)

        for n in self._nodes:
            if n.node_type == LearnedNode.AND:
                al = map_aiger[n.left.node_id]
                ar = map_aiger[n.right.node_id]

                if n.left_edge_type == -1:
                    al += 1
                if n.right_edge_type == -1:
                    ar += 1
                if al < ar:
                    al, ar = ar, al
                _encode(_next - al)
                _encode(al - ar)
                _next += 2

            # writer.write_and(al, ar)

        # Write symbol table

        # for i, pi in enumerate(self.get_pis()): # Can be skipped
        #     if self.has_name(pi):
        #         writer.write_input_name(i, self.get_name_by_id(pi) )

        # for i, l in enumerate(self.get_latches()): # Can be skipped
        #     if self.has_name(l):
        #         writer.write_latch_name(i, self.get_name_by_id(l) )

        # for i, (po_id, _, _) in enumerate(self.get_pos_by_type(AIG.OUTPUT)): # Can be skipped
        #     if self.po_has_name(po_id):
        #         writer.write_po_name(b'o', i, self.get_name_by_po(po_id) )

        # for i, (po_id, _, _) in enumerate(self.get_pos_by_type(AIG.BAD_STATES)): # Can be skipped
        #     if self.po_has_name(po_id):
        #         writer.write_po_name(b'b', i, self.get_name_by_po(po_id) )

        # for i, (po_id, _, _) in enumerate(self.get_pos_by_type(AIG.CONSTRAINT)): # Can be skipped
        #     if self.po_has_name(po_id):
        #         writer.write_po_name(b'c', i, self.get_name_by_po(po_id) )

        # for i, po_ids in self.get_justice_properties(): # Obsolete

        #     if not po_ids:
        #         continue

        #     po_id = po_ids[0]

        #     if self.po_has_name(po_id):
        #         writer.write_po_name(b'j', i, self.get_name_by_po(po_id) )

        # for i, (po_id, _, _) in enumerate(self.get_pos_by_type(AIG.FAIRNESS)): #Obsolete
        #     if self.po_has_name(po_id):
        #         writer.write_po_name(b'f',i, self.get_name_by_po(po_id) )

        fout.write(_bytes)
        fout.close()
        # fout.write( writer.get_bytes() )

        return map_aiger

    def prepare_data(
        self, embedding_size: int | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not self._instantiated_truth_tables:
            self.instantiate_truth_tables()

        self.collect_truth_tables(embedding_size)

        (
            actions,
            edge_type_idx,
            left_parent_idx,
            right_parent_idx,
        ) = self.collect_actions()
        # a = torch.stack(self._node_truth_tables).to(torch.float32)
        # b = torch.stack(self._po_truth_tables).to(torch.float32)
        return (
            torch.stack(self._node_truth_tables),
            torch.stack(self._po_truth_tables),
            actions,
            edge_type_idx,
            left_parent_idx,
            right_parent_idx,
        )

    def collect_truth_tables(self, embedding_size: int | None = None) -> None:
        self._node_truth_tables.clear()
        self._po_truth_tables.clear()

        if not self._instantiated_truth_tables:
            self.instantiate_truth_tables()

        repeat_factor = 1
        if embedding_size != None:
            repeat_factor = embedding_size // self._truth_table_size

        for node in self._nodes:
            if repeat_factor > 1:
                self._node_truth_tables.append(node.truth_table.repeat(repeat_factor))
            else:
                self._node_truth_tables.append(node.truth_table)

        for po in self._pos:
            if repeat_factor > 1:
                self._po_truth_tables.append(po.truth_table.repeat(repeat_factor))
            else:
                self._po_truth_tables.append(po.truth_table)

    def update_node_truth_tables(self) -> None:
        repeat_factor = (
            torch.numel(self._node_truth_tables[-1]) // self._truth_table_size
        )
        for node in self._nodes[len(self._node_truth_tables) :]:
            if repeat_factor > 1:
                self._node_truth_tables.append(node.truth_table.repeat(repeat_factor))
            else:
                self._node_truth_tables.append(node.truth_table)

    def get_truth_tables(self) -> torch.Tensor:
        return (
            torch.stack(self._node_truth_tables + self._po_truth_tables)
            .to(torch.float32)
            .unsqueeze(0)
        )

    def get_action_mask(self) -> torch.Tensor:
        (
            actions,
            edge_type_idx,
            left_parent_idx,
            right_parent_idx,
        ) = self.collect_actions()
        src_mask = torch.full(
            (len(self._nodes), len(self._nodes)),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        src_mask = torch.triu(src_mask, diagonal=0).T
        src_mask = src_mask.repeat(4, 1, 1)
        src_mask[edge_type_idx, left_parent_idx, right_parent_idx] = float("-inf")
        return src_mask

    def collect_actions(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = torch.zeros(4, len(self._nodes), len(self._nodes), dtype=torch.bool)
        edge_type_idx = []
        left_parent_idx = []
        right_parent_idx = []

        for node in self._nodes:
            if node.is_and():
                if node.left_edge_type == 1 and node.right_edge_type == 1:
                    edge_type_idx.append(0)
                elif node.left_edge_type == 1 and node.right_edge_type == -1:
                    edge_type_idx.append(1)
                elif node.left_edge_type == -1 and node.right_edge_type == 1:
                    edge_type_idx.append(2)
                elif node.left_edge_type == -1 and node.right_edge_type == -1:
                    edge_type_idx.append(3)

                left_parent_idx.append(node.left.node_id)
                right_parent_idx.append(node.right.node_id)

        edge_type_idx = torch.tensor(edge_type_idx, dtype=torch.int32)
        left_parent_idx = torch.tensor(left_parent_idx, dtype=torch.int32)
        right_parent_idx = torch.tensor(right_parent_idx, dtype=torch.int32)
        actions[edge_type_idx, left_parent_idx, right_parent_idx] = True
        return actions, edge_type_idx, left_parent_idx, right_parent_idx

    def create_and_from_tensor(
        self, action: torch.Tensor, temperature: float = 0.000001
    ) -> LearnedNode:
        edge_type = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        probabilities = torch.nn.functional.softmax(
            action.squeeze().view(-1) / temperature, dim=-1
        )
        new_idx = torch.multinomial(probabilities, 1)
        # print(new_idx, torch.argmax(probabilities))
        x = new_idx // len(self._nodes) ** 2
        left = new_idx % len(self._nodes) ** 2 // len(self._nodes)
        right = new_idx - (x * len(self._nodes) ** 2 + left * len(self._nodes))

        # print(x, y, z)
        left_edge_type, right_edge_type = edge_type[x.item()]
        return self.create_and(
            left.item(), right.item(), left_edge_type, right_edge_type
        )

    def create_and_from_tensor_max(self, action: torch.Tensor) -> LearnedNode:
        edge_type = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        idx = (action.squeeze() == torch.max(action)).nonzero().squeeze()
        if len(idx.shape) == 2:
            i = random.randint(0, idx.shape[0] - 1)
            idx = idx[i]
        left_edge_type, right_edge_type = edge_type[idx[0]]
        left = idx[1].item()
        right = idx[2].item()
        return self.create_and(left, right, left_edge_type, right_edge_type)

    def clean_up(self) -> None:
        deleted_nodes = {}
        for node in self._nodes:
            if node.is_and() and node.fanout_size() == 0:
                self.delete_node(node, deleted_nodes)
        # print("Num deleted nodes:", len(deleted_nodes.keys()))
        self.remap_nodes(deleted_nodes)

    def remap_nodes(self, deleted_nodes: dict[int, int]) -> None:
        new_nodes = []
        new_id = 0
        for node in self._nodes:
            if not node.is_and():
                new_nodes.append(node)
                new_id += 1
            elif node.node_id not in deleted_nodes:
                if node.node_id != new_id:
                    # update where ids are used in AIG
                    self._id_to_object[new_id] = node
                    if node.node_id in self._id_to_name:
                        self._id_to_name[new_id] = self._id_to_name[node.node_id]
                    # update the id fanout from the parents
                    if node.left is not None:
                        node.left.delete_fanout(node)
                        node.left.add_fanout(node, node.left_edge_type)

                    if node.right is not None:
                        node.right.delete_fanout(node)
                        node.right.add_fanout(node, node.right_edge_type)
                    # change node id
                    node.node_id = new_id

                new_id += 1
                new_nodes.append(node)
        self._nodes = new_nodes

    def delete_node(self, node: LearnedNode, deleted_nodes: dict[int, int]) -> None:
        deleted_nodes[node.node_id] = 1
        left = node.left
        right = node.right
        if left is not None:
            left.delete_fanout(node)
            if left.fanout_size() == 0:
                self.delete_node(left, deleted_nodes)
        if right is not None:
            right.delete_fanout(node)
            if right.fanout_size() == 0:
                self.delete_node(right, deleted_nodes)

