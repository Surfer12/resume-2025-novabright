import ast
import operator
from typing import Any, Dict

__all__ = [
    "parse_constraint",
]

# ---------------------------------------------------------------------------
# SAFE EXPRESSION EVALUATOR
# ---------------------------------------------------------------------------
# Only the following operators are permitted when parsing constraint strings.
_ALLOWED_BIN_OPS: Dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    # Comparisons
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    # Boolean
    ast.And: operator.and_,
    ast.Or: operator.or_,
}
_ALLOWED_UNARY_OPS: Dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}


class _SafeEvaluator(ast.NodeVisitor):
    """Recursively evaluates a *restricted* Python expression AST.

    Only the operations defined in _ALLOWED_BIN_OPS / _ALLOWED_UNARY_OPS are
    permitted. Variable names are looked-up from the provided *symbols*
    mapping. Any attempt to use a disallowed construct raises *ValueError*.
    """

    def __init__(self, symbols: Dict[str, Any]):
        self._symbols = symbols or {}

    # ---------------------------------------------------------------------
    # Literals & names
    # ---------------------------------------------------------------------
    def visit_Constant(self, node: ast.Constant):  # type: ignore[override]
        return node.value

    # Support for legacy < Py3.8 "Num", "Str", etc.
    visit_Num = visit_Str = visit_Bytes = visit_NameConstant = visit_Constant

    def visit_Name(self, node: ast.Name):  # type: ignore[override]
        if node.id not in self._symbols:
            raise ValueError(f"Unknown symbol {node.id!r} in constraint")
        return self._symbols[node.id]

    # ---------------------------------------------------------------------
    # Unary operators (e.g. -x, not x)
    # ---------------------------------------------------------------------
    def visit_UnaryOp(self, node: ast.UnaryOp):  # type: ignore[override]
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        operand = self.visit(node.operand)
        return _ALLOWED_UNARY_OPS[op_type](operand)

    # ---------------------------------------------------------------------
    # Binary operators (e.g. a + b)
    # ---------------------------------------------------------------------
    def visit_BinOp(self, node: ast.BinOp):  # type: ignore[override]
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _ALLOWED_BIN_OPS[op_type](left, right)

    # ---------------------------------------------------------------------
    # Boolean operators (e.g. a and b)
    # ---------------------------------------------------------------------
    def visit_BoolOp(self, node: ast.BoolOp):  # type: ignore[override]
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        if op_type is ast.And:
            result = True
            for value in node.values:
                result = result and bool(self.visit(value))
                if not result:
                    break
            return result
        elif op_type is ast.Or:
            result = False
            for value in node.values:
                result = result or bool(self.visit(value))
                if result:
                    break
            return result
        else:
            # Should never reach due to _ALLOWED_BIN_OPS filter above
            raise ValueError(f"Boolean operator {op_type.__name__} not supported")

    # ---------------------------------------------------------------------
    # Comparisons (e.g. 1 < x <= y)
    # ---------------------------------------------------------------------
    def visit_Compare(self, node: ast.Compare):  # type: ignore[override]
        left_val = self.visit(node.left)
        comparisons = zip(node.ops, node.comparators)
        result = True
        for op, comparator in comparisons:
            op_type = type(op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError(f"Comparison {op_type.__name__} not allowed")
            right_val = self.visit(comparator)
            result = result and _ALLOWED_BIN_OPS[op_type](left_val, right_val)
            left_val = right_val  # chained comparisons
            if not result:
                break
        return result

    # ---------------------------------------------------------------------
    # Any other AST node is rejected
    # ---------------------------------------------------------------------
    def generic_visit(self, node):  # type: ignore[override]
        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}")


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def parse_constraint(expr: str, symbols: Dict[str, Any] | None = None) -> Any:
    """Safely evaluate a logical/arithmetical *expr* using *symbols*.

    Example
    -------
    >>> parse_constraint("x > y and y < 10", {"x": 3, "y": 1})
    True
    """
    tree = ast.parse(expr, mode="eval")
    evaluator = _SafeEvaluator(symbols or {})
    return evaluator.visit(tree.body)