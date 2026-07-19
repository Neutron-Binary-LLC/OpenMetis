import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.expression import ExpressionNode, ExpressionTree
from nn.workspace import MathWorkspace
from nn.symbolic_expert import SymbolicExpert
from nn.block import NeuroSymbolicReasoningCell, MathConfig

def test_expression_tree():
    print("Testing ExpressionTree...")
    # (x + 0) * 1
    x = ExpressionNode("x")
    zero = ExpressionNode("0")
    one = ExpressionNode("1")
    plus = ExpressionNode("+", [x, zero])
    mult = ExpressionNode("*", [plus, one])
    tree = ExpressionTree(mult)
    
    print(f"Original: {tree.to_infix()}")
    assert tree.to_infix() == "((x + 0) * 1)"
    return tree

def test_symbolic_expert(tree):
    print("Testing SymbolicExpert...")
    expert = SymbolicExpert(d_model=128)
    
    # Manually call simplify
    simplified_root = expert.simplify(tree.root)
    simplified_tree = ExpressionTree(simplified_root)
    
    print(f"Simplified: {simplified_tree.to_infix()}")
    assert simplified_tree.to_infix() == "x"

def test_workspace_with_trees():
    print("Testing MathWorkspace with trees...")
    batch_size = 2
    ws = MathWorkspace.new(batch_size, 128, torch.device('cpu'))
    
    tree1 = ExpressionTree(ExpressionNode("+", [ExpressionNode("a"), ExpressionNode("b")]))
    tree2 = ExpressionTree(ExpressionNode("*", [ExpressionNode("c"), ExpressionNode("d")]))
    
    delta_latent = torch.randn(batch_size, 128)
    ws.update(delta_latent, delta_expr=["a+b", "c*d"], delta_trees=[tree1, tree2])
    
    assert ws.expression_trees[0].to_infix() == "(a + b)"
    assert ws.expression_trees[1].to_infix() == "(c * d)"
    print("Workspace tree update successful.")

def test_cell_integration():
    print("Testing NeuroSymbolicReasoningCell integration...")
    config = MathConfig(dim=128, workspace_dim=128, max_loop_iters=2)
    cell = NeuroSymbolicReasoningCell(config)
    
    x = torch.randn(1, 10, 128)
    h, ws, _ = cell(x)
    
    print(f"Cell output shape: {h.shape}")
    print(f"Workspace iteration count: {ws.iteration_count}")
    print(f"Final expressions: {ws.symbolic_expr}")
    
    assert h.shape == (1, 10, 128)
    assert ws.iteration_count == 2
    print("Cell integration successful.")

if __name__ == "__main__":
    try:
        tree = test_expression_tree()
        test_symbolic_expert(tree)
        test_workspace_with_trees()
        test_cell_integration()
        print("\nAll Phase 2 tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
