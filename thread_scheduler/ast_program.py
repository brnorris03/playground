#!/usr/bin/env python3
"""
AST-based program generation for natural Python syntax.

Allows writing programs with completely natural Python syntax:
    def my_program():
        x = 10
        y = 20
        sum_val = x + y
        diff = x - y
        result = sum_val * diff
        return result
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any, Callable, List, Optional
from .simulator import Operation
from .utils import SourceLocation


class ASTProgram:
    """
    Converts Python functions to device operations using AST traversal.

    Usage:
        def my_computation():
            x = 10
            y = 20
            result = x + y
            return result

        ops = ASTProgram(dev).compile(my_computation)
    """

    def __init__(
        self,
        device,
        scope_prefix: Optional[str] = None,
        closure_vars: Optional[dict] = None,
    ):
        self.device = device
        self.operations: List[Operation] = []
        self.variables: dict[str, Any] = {}
        self.scope_prefix = scope_prefix  # Prefix for variable scoping
        self.closure_vars = closure_vars or {}  # Closure variables from outer scope
        self.source_location: Optional[SourceLocation] = SourceLocation(None, None)

    def compile(self, func: Callable) -> List[Operation]:
        """
        Compile a Python function to device operations.

        Args:
            func: Python function to compile

        Returns:
            List of device operations
        """
        # Get the source file and line number of the function
        self.source_file = inspect.getsourcefile(func)
        source_lines, start_line = inspect.getsourcelines(func)

        # Remove decorators from source lines - they start with @
        clean_lines = []
        for line in source_lines:
            stripped = line.lstrip()
            if not stripped.startswith("@"):
                clean_lines.append(line)

        # Adjust start_line to account for removed decorators
        num_decorators = len(source_lines) - len(clean_lines)
        self.source_line_offset = start_line + num_decorators - 1  # Convert to 0-based

        # Dedent and parse the cleaned source
        source = "".join(clean_lines)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if func_def is None:
            raise ValueError("Could not find function definition")

        # Process the function body
        self.operations = []
        self.variables = {}

        for stmt in func_def.body:
            self._process_statement(stmt)

        return self.operations

    def _get_source_location(self, node: ast.AST) -> Optional[SourceLocation]:
        """
        Get the source file and line number for an AST node.

        Args:
            node: AST node

        Returns:
            SourceLocation
        """
        if hasattr(node, "lineno") and node.lineno is not None:
            # AST line numbers are relative to the parsed source
            # Add the offset to get the actual line number in the file
            actual_line = node.lineno + self.source_line_offset
            return SourceLocation(self.source_file, actual_line)
        return SourceLocation(None, None)

    def _scoped_name(self, var_name: str) -> str:
        """
        Convert a variable name to a scoped name.

        Args:
            var_name: Original variable name

        Returns:
            Scoped variable name (e.g., "func_name.var_name")
        """
        if self.scope_prefix:
            return f"{self.scope_prefix}.{var_name}"
        return var_name

    def _process_statement(self, stmt: ast.stmt):
        """Process a single statement."""
        if isinstance(stmt, ast.Assign):
            self._process_assignment(stmt)
        elif isinstance(stmt, ast.Return):
            # Return statements are ignored - we just generate all operations
            pass
        elif isinstance(stmt, ast.Expr):
            # Expression statements (e.g., function calls like store())
            if isinstance(stmt.value, ast.Call):
                self._process_expression_call(stmt.value, stmt)
        else:
            raise NotImplementedError(
                f"Statement type not supported: {type(stmt).__name__}"
            )

    def _process_assignment(self, stmt: ast.Assign):
        """Process an assignment statement: x = value or x = expr."""
        if len(stmt.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")

        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Only simple variable assignments supported")

        var_name = target.id
        scoped_var_name = self._scoped_name(var_name)
        value_node = stmt.value

        # Get source location for this statement
        location = self._get_source_location(stmt)

        # Check if it's a literal value (initialization)
        if isinstance(value_node, ast.Constant):
            # x = 10
            value = value_node.value
            self.operations.append(
                self.device.write(scoped_var_name, value, location=location)
            )
            self.operations.append(self.device.push(scoped_var_name, location=location))
            self.variables[var_name] = value

        elif isinstance(value_node, ast.BinOp):
            # x = a + b (or other binary operation)
            self._process_binary_op(scoped_var_name, value_node, stmt)

        elif isinstance(value_node, ast.Name):
            # x = y (variable reference)
            source_name = value_node.id
            scoped_source_name = self._scoped_name(source_name)
            self.operations.append(
                self.device.wait(scoped_source_name, location=location)
            )
            # Note: We don't create a new variable, just wait for the source
            self.variables[var_name] = source_name

        elif isinstance(value_node, ast.Call):
            # x = read(value) or x = func()
            self._process_call(scoped_var_name, value_node, stmt)

        else:
            raise NotImplementedError(
                f"Value type not supported: {type(value_node).__name__}"
            )

    def _process_expression_call(self, node: ast.Call, stmt: ast.AST):
        """Process a function call as a statement (e.g., store())."""
        # Get source location for this statement
        location = self._get_source_location(stmt)

        # Check if it's the store() function
        if isinstance(node.func, ast.Name) and node.func.id == "store":
            # store(source, destination) -> wait(source) + write(destination, source) + push(destination)
            if len(node.args) != 2:
                raise ValueError(
                    "store() expects exactly two arguments: store(source, destination)"
                )

            source_arg = node.args[0]
            dest_arg = node.args[1]

            # Get source variable name (should be a Name node)
            if isinstance(source_arg, ast.Name):
                source_name = self._scoped_name(source_arg.id)
            else:
                raise NotImplementedError(
                    f"store() source must be a variable name, got: {type(source_arg).__name__}"
                )

            # Get destination name (should be a string constant or f-string)
            if isinstance(dest_arg, ast.Constant) and isinstance(dest_arg.value, str):
                dest_name = dest_arg.value
            elif isinstance(dest_arg, ast.Name):
                # If it's a variable name, evaluate it from closure
                var_name = dest_arg.id
                if var_name in self.closure_vars:
                    dest_name = self.closure_vars[var_name]
                else:
                    dest_name = var_name
            elif isinstance(dest_arg, ast.JoinedStr):
                # F-string - need to evaluate it using closure variables
                # For now, use ast.unparse and eval with closure_vars
                try:
                    dest_name = eval(ast.unparse(dest_arg), {}, self.closure_vars)
                except Exception as e:
                    raise ValueError(
                        f"Could not evaluate f-string destination: {ast.unparse(dest_arg)}"
                    ) from e
            else:
                raise NotImplementedError(
                    f"store() destination must be a string constant, variable, or f-string, got: {type(dest_arg).__name__}"
                )

            # Generate wait + write + push operations
            self.operations.append(self.device.wait(source_name, location=location))
            self.operations.append(
                self.device.write(dest_name, source_name, location=location)
            )
            self.operations.append(self.device.push(dest_name, location=location))
        else:
            # Other function calls are not supported
            raise NotImplementedError(
                f"Function call not supported: {ast.unparse(node)}"
            )

    def _process_call(self, result_name: str, node: ast.Call, stmt: ast.AST):
        """Process a function call: x = read(value) or x = func()."""
        # Get source location for this statement
        location = self._get_source_location(stmt)

        # Check if it's the read() function
        if isinstance(node.func, ast.Name) and node.func.id == "read":
            # x = read(value) -> write(x, value) + push(x)
            if len(node.args) != 1:
                raise ValueError("read() expects exactly one argument")

            # Get the value to read (could be a constant, variable, or subscript)
            arg = node.args[0]

            # Evaluate the argument to get the actual value
            # For now, we support: read(constant), read(variable), read(var[slice])
            if isinstance(arg, ast.Constant):
                # read(10) -> write(x, 10) + push(x)
                value = arg.value
            elif isinstance(arg, ast.Name):
                # read(some_var) -> write(x, some_var) + push(x)
                # The value is a reference to a Python variable (closure)
                var_name = arg.id
                if var_name in self.closure_vars:
                    # Use the actual value from closure
                    value = self.closure_vars[var_name]
                else:
                    # Fall back to the variable name (might be a scoped variable)
                    value = var_name
            elif isinstance(arg, ast.Subscript):
                # read(data[0:1024]) -> write(x, "data[0:1024]") + push(x)
                # For now, just use a string representation
                value = ast.unparse(arg)
            else:
                raise NotImplementedError(
                    f"read() argument type not supported: {type(arg).__name__}"
                )

            # Generate write + push operations
            self.operations.append(
                self.device.write(result_name, value, location=location)
            )
            self.operations.append(self.device.push(result_name, location=location))
            self.variables[result_name] = value
        else:
            raise NotImplementedError(
                f"Function call not supported: {ast.unparse(node)}"
            )

    def _process_binary_op(self, result_name: str, node: ast.BinOp, stmt: ast.AST):
        """Process a binary operation: result = left op right."""
        # Get source location for this statement
        location = self._get_source_location(stmt)

        # Get operand names
        left_name = self._get_operand_name(node.left, stmt)
        right_name = self._get_operand_name(node.right, stmt)

        # Generate wait operations
        self.operations.append(self.device.wait(left_name, location=location))
        self.operations.append(self.device.wait(right_name, location=location))

        # Generate the operation based on operator type
        if isinstance(node.op, ast.Add):
            self.operations.append(
                self.device.add(left_name, right_name, result_name, location=location)
            )
        elif isinstance(node.op, ast.Sub):
            self.operations.append(
                self.device.subtract(
                    left_name, right_name, result_name, location=location
                )
            )
        elif isinstance(node.op, ast.Mult):
            self.operations.append(
                self.device.multiply(
                    left_name, right_name, result_name, location=location
                )
            )
        else:
            raise NotImplementedError(
                f"Operator not supported: {type(node.op).__name__}"
            )

        # Push the result
        self.operations.append(self.device.push(result_name, location=location))
        self.variables[result_name] = None  # Mark as computed

    def _get_operand_name(self, node: ast.expr, parent_stmt: ast.AST = None) -> str:
        """Get the name of an operand (variable or nested expression)."""
        if isinstance(node, ast.Name):
            return self._scoped_name(node.id)
        elif isinstance(node, ast.Constant):
            # Handle inline constants by creating a temporary variable
            temp_name = f"_const_{id(node)}"
            scoped_temp_name = self._scoped_name(temp_name)
            location = self._get_source_location(
                node if parent_stmt is None else parent_stmt
            )
            self.operations.append(
                self.device.write(scoped_temp_name, node.value, location=location)
            )
            self.operations.append(
                self.device.push(scoped_temp_name, location=location)
            )
            return scoped_temp_name
        elif isinstance(node, ast.BinOp):
            # Nested expression - need to compute it first
            temp_name = f"_temp_{len(self.variables)}"
            scoped_temp_name = self._scoped_name(temp_name)
            self._process_binary_op(
                scoped_temp_name, node, node if parent_stmt is None else parent_stmt
            )
            return scoped_temp_name
        else:
            raise NotImplementedError(
                f"Operand type not supported: {type(node).__name__}"
            )


def read(value):
    """
    Placeholder function for AST compilation.

    In AST-compiled code, `x = read(value)` generates:
        dev.write("x", value)
        dev.push("x")

    This function should never be called at runtime - it's only
    recognized by the AST compiler.

    Args:
        value: The value to read (can be a constant, variable, or slice)

    Returns:
        The value (but this is never actually executed)

    Example:
        @sim.thread(name="reader", iteration=0)
        def reader():
            a = read(82)  # Generates: write(iter_0.a, 82) + push(iter_0.a)
            b = read(15)
            return a, b
    """
    raise RuntimeError(
        "read() should only be used in AST-compiled code, not called directly"
    )


def store(source, destination):
    """
    Placeholder function for AST compilation.

    In AST-compiled code, `store(source, destination)` generates:
        dev.wait("source")
        dev.write("destination", "source")
        dev.push("destination")

    This function should never be called at runtime - it's only
    recognized by the AST compiler.

    Args:
        source: The variable to read from (scoped variable name)
        destination: The destination name (typically a string for external storage)

    Returns:
        None (but this is never actually executed)

    Example:
        @sim.thread(name="writer", iteration=0)
        def writer():
            store(result, "output_0")  # Generates: wait(iter_0.result) + write(output_0, iter_0.result) + push(output_0)
            return None
    """
    raise RuntimeError(
        "store() should only be used in AST-compiled code, not called directly"
    )


def program(device):
    """
    Decorator to convert a Python function to device operations using AST.

    Usage:
        @program(dev)
        def my_computation():
            x = 10
            y = 20
            result = x + y
            return result

    The decorated function returns a list of operations instead of executing.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper() -> List[Operation]:
            compiler = ASTProgram(device)
            return compiler.compile(func)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
