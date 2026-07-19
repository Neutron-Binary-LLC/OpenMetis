import torch
import torch.nn as nn
from typing import List, Optional
from .expression import ExpressionTree, ExpressionNode

class SymbolicExpert(nn.Module):
    """
    Expert that proposes symbolic rewrites based on algebraic rules.
    Acts as a 'Symbolic Expert' in the neuro-symbolic reasoning cycle.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Project hidden state to rule selection logits
        self.rule_selector = nn.Linear(d_model, 8) # 8 example rules
        
    def forward(self, h: torch.Tensor, trees: List[Optional[ExpressionTree]]) -> List[Optional[ExpressionTree]]:
        """
        h: (batch, seq_len, d_model) or (batch, d_model)
        trees: List of current expression trees in the workspace
        """
        if h.dim() == 3:
            h_pool = h.mean(dim=1)
        else:
            h_pool = h
            
        rule_logits = self.rule_selector(h_pool)
        selected_rules = torch.argmax(rule_logits, dim=-1)
        
        new_trees = []
        for i, tree in enumerate(trees):
            if tree is None:
                new_trees.append(None)
                continue
            
            # Apply rules based on neural guidance or just exhaustive simplification
            new_root = self.simplify(tree.root)
            new_trees.append(ExpressionTree(new_root))
            
        return new_trees

    def simplify(self, node: ExpressionNode) -> ExpressionNode:
        """Recursive algebraic simplification."""
        new_children = [self.simplify(c) for c in node.children]
        
        # Algebraic Rewrite Rules
        
        # 1. Addition rules
        if node.value == "+":
            if len(new_children) == 2:
                # x + 0 -> x
                if new_children[1].value == "0": return new_children[0]
                if new_children[0].value == "0": return new_children[1]
                # constant folding (if both are numeric)
                try:
                    v1 = float(new_children[0].value)
                    v2 = float(new_children[1].value)
                    return ExpressionNode(str(v1 + v2))
                except ValueError:
                    pass
        
        # 2. Multiplication rules
        if node.value == "*":
            if len(new_children) == 2:
                # x * 0 -> 0
                if new_children[1].value == "0" or new_children[0].value == "0":
                    return ExpressionNode("0")
                # x * 1 -> x
                if new_children[1].value == "1": return new_children[0]
                if new_children[0].value == "1": return new_children[1]
                # constant folding
                try:
                    v1 = float(new_children[0].value)
                    v2 = float(new_children[1].value)
                    return ExpressionNode(str(v1 * v2))
                except ValueError:
                    pass

        # 3. Subtraction rules
        if node.value == "-":
            if len(new_children) == 2:
                # x - 0 -> x
                if new_children[1].value == "0": return new_children[0]
                # x - x -> 0
                if str(new_children[0]) == str(new_children[1]):
                    return ExpressionNode("0")
        
        return ExpressionNode(node.value, new_children)
