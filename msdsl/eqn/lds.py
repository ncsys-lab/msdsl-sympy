from numbers import Number

import numpy as np
import scipy.linalg

import sympy as sp
from msdsl.assignment import Assignment
from msdsl.expr.expr import Array, LessThan, GreaterThan, Product, Sum, EqualTo, UIntConstant, RealConstant, Constant
from msdsl.expr.signals import AnalogSignal, DigitalSignal

class LDS:
    def __init__(self, A=None, B=None, C=None, D=None):
        # save settings
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def discretize(self, dt: Number):
        # discretize A
        if self.A is not None:
            A_tilde = scipy.linalg.expm(dt * self.A)
        else:
            A_tilde = None

        # discretize B
        if self.A is not None and self.B is not None:
            I = np.eye(*self.A.shape) # identity matrix with shape of A
            B_tilde = np.linalg.solve(self.A, (A_tilde - I).dot(self.B))
        else:
            B_tilde = None

        # discretize C
        if self.C is not None:
            C_tilde = self.C.copy()
        else:
            C_tilde = None

        # discretize D
        if self.D is not None:
            D_tilde = self.D.copy()
        else:
            D_tilde = None

        # return result
        return LDS(A=A_tilde, B=B_tilde, C=C_tilde, D=D_tilde)

    # overloaded methods

    def __str__(self):
        # build up list of lines
        retval = ['*** Linear Dynamical System ***']
        for k, (name, mat) in enumerate([('A', self.A), ('B', self.B), ('C', self.C), ('D', self.D)]):
            retval.append('')
            retval.append(f'{name} matrix')
            retval.append(str(mat))

        # add newlines
        retval = '\n'.join(retval)

        # return result
        return retval
    
    def convert_to_sympy(self, states, inputs, outputs):
        state_strings = list(map(lambda x: str(x), states))
        inputs_strings = list(map(lambda x: str(x), inputs))
        outputs_strings = list(map(lambda x: str(x), outputs))
        # Convert state, input, and output strings to sympy symbols
        states = sp.symbols(state_strings)
        inputs = sp.symbols(inputs_strings)
        outputs = sp.symbols(outputs_strings)

        # Convert numpy arrays to sympy matrices
        A_sym = sp.Matrix(self.A)
        B_sym = sp.Matrix(self.B)
        C_sym = sp.Matrix(self.C)
        D_sym = sp.Matrix(self.D)
        
        # Define state-space equations
        state_eq = A_sym * sp.Matrix(states) + B_sym * sp.Matrix(inputs)
        output_eq = C_sym * sp.Matrix(states) + D_sym * sp.Matrix(inputs)

        # Explicitly compute the derivatives (dot{x})
        state_ode = sp.Matrix([sp.diff(state, 't') for state in states]) - state_eq

        return state_ode, output_eq


class LdsCollection:
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.D = None

    def append(self, lds: LDS):
        # Add extra dimension to each array
        A = lds.A[:, :, np.newaxis] if lds.A is not None else None
        B = lds.B[:, :, np.newaxis] if lds.B is not None else None
        C = lds.C[:, :, np.newaxis] if lds.C is not None else None
        D = lds.D[:, :, np.newaxis] if lds.D is not None else None

        # Update each array
        self.A = np.concatenate((self.A, A), axis=2) if self.A is not None else A
        self.B = np.concatenate((self.B, B), axis=2) if self.B is not None else B
        self.C = np.concatenate((self.C, C), axis=2) if self.C is not None else C
        self.D = np.concatenate((self.D, D), axis=2) if self.D is not None else D


    def convert_to_sympy_piecewise(self, states, inputs, outputs, sel_bits, sel_eqns):
        # Convert states, inputs, and outputs to sympy symbols
        state_strings = list(map(str, states))
        inputs_strings = list(map(str, inputs))
        outputs_strings = list(map(str, outputs))

        states = sp.symbols(state_strings)
        inputs = sp.symbols(inputs_strings)
        outputs = sp.symbols(outputs_strings)

        # Convert sel_bits to SymPy symbols if they aren't already
        sel_bits_sympy = [sp.Symbol(str(sel_bit)) if not isinstance(sel_bit, sp.Basic) else sel_bit for sel_bit in sel_bits]

        # Initialize lists for state and output equations with default expressions
        state_eq_piecewise = [sp.Piecewise((0, True)) for _ in range(len(states))]
        output_eq_piecewise = [sp.Piecewise((0, True)) for _ in range(len(outputs))]

        # Iterate over all possible configurations of sel_bits
        
        for k in range(self.A.shape[2]):  # Number of scenarios
            A_sym = sp.Matrix(self.A[:, :, k])
            B_sym = sp.Matrix(self.B[:, :, k])
            C_sym = sp.Matrix(self.C[:, :, k])
            D_sym = sp.Matrix(self.D[:, :, k])

            # Define state-space equations
            state_eq = A_sym * sp.Matrix(states) + B_sym * sp.Matrix(inputs)
            output_eq = C_sym * sp.Matrix(states) + D_sym * sp.Matrix(inputs)

            # Compute derivatives (dot{x}) for the state equations
            state_ode = sp.Matrix([sp.diff(state, 't') for state in states]) - state_eq

            # Create the condition for this scenario
            condition = True
            for i, sel_bit_sym in enumerate(sel_bits_sympy):
                bit_value = (k >> i) & 1
                # Use logical AND to build up the condition
                condition = sp.And(condition, sp.Eq(sel_bit_sym, bit_value))
                

            # Process sel_eqns to adjust the condition if needed
            for sel_eqn in sel_eqns:
                
            
                if isinstance(sel_eqn, Assignment):
                    signal = sp.Symbol(sel_eqn.signal.name)
                    expr = msdsl_ast_to_sympy(sel_eqn.expr)

                    condition = condition.subs(signal, expr)
                    
            # Assign the corresponding equations to the Piecewise objects
            for i in range(len(states)):

                state_eq_piecewise[i] = sp.Piecewise((state_ode[i], condition), (state_eq_piecewise[i], True))

            for i in range(len(outputs)):
                output_eq_piecewise[i] = sp.Piecewise((output_eq[i], condition), (output_eq_piecewise[i], True))

        return state_eq_piecewise + output_eq_piecewise




def msdsl_ast_to_sympy(ast):
    """
    Convert an AST from msdsl to a sympy expression.
    """
    if isinstance(ast, LessThan):
        return sp.Lt(msdsl_ast_to_sympy(ast.lhs), msdsl_ast_to_sympy(ast.rhs))
    elif isinstance(ast, GreaterThan):
        return sp.Gt(msdsl_ast_to_sympy(ast.lhs), msdsl_ast_to_sympy(ast.rhs))
    elif isinstance(ast, Product):
        accum = 1  # Corrected from 0 to 1 to properly accumulate products
        for operand in ast.operands:
            accum *= msdsl_ast_to_sympy(operand)
        return accum
    elif isinstance(ast, Sum):
        accum = 0
        for operand in ast.operands:
            accum += msdsl_ast_to_sympy(operand)
        return accum
    elif isinstance(ast, EqualTo):
        return sp.Eq(msdsl_ast_to_sympy(ast.lhs), msdsl_ast_to_sympy(ast.rhs))
    elif isinstance(ast, Array):
        elements = ast.operands[:-1]
        address = msdsl_ast_to_sympy(ast.operands[-1])
        return sp.Piecewise(*[(msdsl_ast_to_sympy(elem), address if i == 1 else sp.Not(address)) for i, elem in enumerate(elements)])
    elif isinstance(ast, Constant):
        return ast.value
    elif isinstance(ast, AnalogSignal) or isinstance(ast, DigitalSignal):
        return sp.Symbol(str(ast))
    else:
        raise Exception(f"Unsupported AST node: {type(ast)}")
