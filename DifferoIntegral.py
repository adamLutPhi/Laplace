#sympy
#from sympy.core.function import ( Lambda, diff) #Function
import sympy as sp
from sympy import Symbol

from sympy import exp, I, pi #exp1?
from sympy.functions.elementary.exponential import exp, log #exp2?
# sympy function wrappers
from sympy.functions.elementary.trigonometric import (acos, cos, cot, sin,
                                                      sinc, tan)
from sympy import exp, I, pi
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp, log
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.function import (Function, Lambda, diff)

from sympy.integrals.integrals import Integral

from sympy.core.numbers import (E, Float, I, Rational, oo, pi)

# impotant functions

#from ABC import abc
class DiffeoIntegral(object): #(abc): # static 
    """classically differentiate and Integration (lambda) functions
    class has to be `static` (in a java sense)
    pythonic approach using a `classmethod` as a class decorator`
    not forgetting the class inheriting from `object`
    instance (no `__new__` is reqired
    """

    @classmethod
    def defineVar(cls,xStr : str ='x'):#un-useful
        """ Define a (1) single variable"""
        x =  sp.Symbol(xStr)
        return x
    @classmethod
    def definef(cls,f= lambda x: exp(x**2) ,x='x'):#useful

        x = sp.Symbol(x)
    
        #f = sp.implemented_function('f', lambda x: x+1)
        #f= sp.lambdify(x, f(x)) #<
        f = Lambda(x, exp(-x**2))
        return x, f

    @classmethod
    # differentiate
    def derive(cls, f, x):
        diffLambda = diff(f(x), x)
        return diffLambda
    @classmethod
    # Integrate
    def integrate(cls,f): #takes no x (?!)
        intLambda =Integral(f)
        return intLambda
    
#x = DiffeoIntegral.defineVar('x')
#DEMO
#Pass a lambda
x,f = DiffeoIntegral.definef() # use default #<

print(f"Original Lmabda f= {f}")
print(type(f)) # <class 'function'>

# Does the differentiation (Newton method?)
f1 = DiffeoIntegral.derive( f , x)
print(f"Derivative f'= {f1}")

# a form (struct/class) of an integral
F = DiffeoIntegral.integrate(f1)

print(f"Integral of last function F(f')= {F}")

print( f"Is original lambda(f) == Integral(derivative(f))?: { f == F }" )

print("Note: sympy Integral did not Integrate (as expected)...")

