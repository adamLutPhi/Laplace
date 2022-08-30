"""
Created on Thu Aug 25 19:45:07 2022

@author: Ahmad Lutfi

# the Best 
# Software Design Implementation: 
Ast parsing Requires 2 cycles: 

1. parse Phase:
   Transforms a string to an ast Object

2. deParse Phase:
intent: What: a Program that understands input
How: Transforms an ast Object to a custom-built model (class, function,variable,...)

"""

import ast

from ast import fix_missing_locations, Subscript, Constant, Name, Load, NodeTransformer
import math

LambdaEqn = 'lambda x: x**2 '

astOperators = [ast.Pow, ast.Add, ast.Sub, ast.Mult, ast.Div ]

#--------------------------
# checkIfAstOperator

def checkIfAstOperator(operator, astOperators):
    """ checks if an operator uses an astOperators """
    
    operation = None
    IsOperator = False
    #opTuple = IsOperator
    #opTuple[0] = operation #[0] = operation
    for op in astOperators:
        # if p
        # pass
        if operator.__eq__(op):
            IsOperator = True
            operation = op
            #opTuple[0] = operation
            #(operation, IsOperator)
            print("#Notice: Pow is parsed Correctly")
            break
    return IsOperator, operation  # tuple(opTuple) # IsOperator , operation


# ----------------------------
# parseBodyOp Handling
#IsOperator,operation =  checkIfAstOperator(parsedBodyOp,astOperators)
# print("IsOperator ",IsOperator)
# print("operation ",operation)

# Implements a  NodeTrabsfirner
class RewriteName(NodeTransformer):
    """ a clase for a Node Transformer object """ 

    def visit_Name(self, node):
        """ a function that visits a particular node """
        
        return Subscript(
            value=Name(id='data', ctx=Load()), # add a Name 
            slice=Constant(value=node.id), # attach a Constant on a value 
            ctx=node.ctx # assing a ctx 
        )

if __name__ == 'main':

    _string = input() # let's user input  a string
    parsedBody = parseLambda(_string)
    
    leftbody, op, rightBody =  deParseLambda(parsedBody) # parsedBodyLeftValueSliceValue, operation, parsedBodyRightValue
    switchOp(op)


# start: 
# takeaway: example: paring a lambda: x: x**2, mode='eval'

def parse(_string = 'lambda x: x**2 ', mode = 'eval'):
    """returns a treeExpression """ 
    tree = ast.parse('lambda x: x**2 ', mode='eval')
    # Recalculate the `location information`
    #Viable line  Fix missing locations, rewrite & visit Tree
    treeExpression = fix_missing_locations(RewriteName().visit(tree))  # new_tree

    return treeExpression

def parseLambda(_string = 'lambda x: x**2 '):
    """parsed a lambda string , returns the tree of a parsed object
    fixes missing locations
    returns a treeExpression"""

    tree = ast.parse(_string, mode='eval')
    # Recalculate the `location information`
    #Viable line  Fix missing locations, rewrite & visit Tree
    treeExpression = parse(_string) 
    fix_missing_locations(RewriteName().visit(tree))  # new_tree
    return treeExpression

treeExpression = parse(LambdaEqn)

print("treeExpression", ast.parse(treeExpression))
print("treeExpression", parseLambda(treeExpression))


# ----------------------
# switchOp
def switchOp(astOp):
    """ switching a parsing ast Operator, into a mathematical Operator
    
    Example:

    switch(**)
     """ 
    #pass
    if astOp == ast.Pow: # **
        return math.pow
    elif astOp == ast.Add: # + 
        return math.Add
    elif astOp == ast.Sub: # - 
        return math.Sub
    elif astOp == ast.Mul: # *
        return math.mul
    elif astOp == ast.Div: # / 
        return math.Div 
    
# ----------------------

# LambdaClass 

class LambdaClass():
    """
    def __init__(self, treeExpression):
        self._lambda = treeExpression.body 
        return self._lambda 
    """
# store lambda 
    #def deparseLambda()
    def storeLambda(self, _string='lambda x: x**2 '):
        """ gets a string, returns a parsedtree, treeExpression, _lambda""" 
        self.parsedtree = ast.parse(_string, mode='eval')
        # recalculate the `location information`
        # fix missing locations, rewrite & visit Tree
        self.treeExpression = fix_missing_locations(
            RewriteName().visit(self.parsedtree))
        self._lambda = treeExpression.body
        # return self._lambda
        Lambda = self._lambda
        return Lambda

#-----------------------------------------------------------       
class LambdaClass():
    
    def __init__(self, parsedBody):
        # parsedBody.left.value.slice.value # x : str
        self.x = parseLeftBody(parsedBody)
        self.op = parsedBody.op  # Pow()
        self.n = parseRightBody(parsedBody)  # parsedBody.right  # 2
    
    
    def __init__(self, x, astOp, scalar):
        # parsedBody.left.value.slice.value # x : str
        self.x = x #parseLeftBody(parsedBody)
        self.op = astOp # parsedBody.op  # Pow()
        self.n = scalar #parseRightBody(parsedBody)  # parsedBody.right  # 2

# -----------
# lambdaParse 
# parsing Phase 1: from string to ast treeExpression object, to a Lambda

#--------------------------
# lambdaParse
def lambdaParse(_string='lambda x: x**2 '):
    """parses string that has a lambda into an ast Lambda Object """

    parsedtree = ast.parse(_string, mode='eval')
    # recalculate the `location information`
    # fix missing locations, rewrite & visit Tree
    treeExpression = fix_missing_locations(
        RewriteName().visit(parsedtree))  # new_tree
    _lambda = treeExpression.body
    return _lambda
    

#--------------------------
# checkifLambda

def checkifLambda(LambdaEqn): # LambdaEqn
    """checks if given lambda is an ast Lambda object """
    
    if type(LambdaEqn) == ast.Lambda:
        return True

#-------------------
# Check `Lambda` content 
print("lambda ", ast.dump(Lambda))

# -------
# Lambda ---> parsedBody
def lambda2ParsedBody(LambdaEqn):
    """returns a parsedBody """
    parsedLambda = (ast.parse(LambdaEqn))

    #print("parsed = ",parsed)
    parsedBody = parsedLambda.body  # BinOp
    parsedBody = ast.dump(parsedBody)
    
    print("parsedBody = ", parsedBody)
    return parsedBody
    

print("lambda tree = ", LambdaEqn)
parsedLambda = (ast.parse(LambdaEqn))

#print("parsed = ",parsed)
parsedBody = parsedLambda.body  # BinOp
print("parsedBody = ", ast.dump(parsedBody))


#--------------------------

# Helper Function 
# lambda2ParsedBody
def lambda2ParsedBody(LambdaEqn):
    """converts a lambda object into an ast parsed Lambda parsedBody """
    
    # Lambda ---> parsedBody
    print("lambda tree = ", LambdaEqn)
    parsedLambda = (ast.parse(LambdaEqn))

    #print("parsed = ",parsed)
    parsedBody = parsedLambda.body  # BinOp
    print("parsedBody = ", ast.dump(parsedBody))
    return parsedBody

lambda2ParsedBody(Lambda)

#--------------------------
# parseRightBody

def parseRightBody(parsedBody):  # Constant Object (of _ast module)
    """ parsed the Right-hand side of a lambda, to get a scalar value """
    #_attributes : tuple
    #_fields : tuple
    # kind NoneType
    # n , s, value : int
    # useful (int)  2 # Constant object [2 tuples ]
    parsedBodyRight = parsedBody.right
    print("parsedBodyRight", (parsedBodyRight))

    # useful (int)  2  [Terminus]
    parsedBodyRightValue = parsedBody.right.value
    print("parsedBodyRightValue", (parsedBodyRightValue))

    return parsedBodyRightValue

#--------------------------
# parseLeftBody

def parseLeftBody(parsedBody):  # Constant Object (of _ast module)
    """ parses the left part of lambda, which is a Script object, the if sliced correctly,
    retrives back the variable name  """
    # LEFT
    parsedBodyLeft = parsedBody.left  # not much useful
    print("parsedBodyLeft", ast.dump(parsedBodyLeft))

    parsedBodyLeftValue = parsedBody.left  # complicated object: #Subscript
    print("parsedBodyLeftValue", ast.dump(parsedBodyLeft.value))
    # -----------
    # Constant   #ast.name (object
    parsedBodyLeftValueSlice = parsedBodyLeftValue.slice
    print("parsedBodyLeftValueSlice", ast.dump(
        parsedBodyLeftValueSlice))  # object
    # -------------------------

    parsedBodyLeftValueSliceValue = parsedBodyLeftValueSlice.value  #
    print("parsedBodyLeftValueSlice", parsedBodyLeftValueSliceValue)  # x

    print("type ", type(parsedBodyLeftValueSliceValue))  # <class 'str'>

    return parsedBodyLeftValueSliceValue

#-------------------------------------------
# parseLambda

def parseLambda(_String: str):
    """ parses lambda from string to Lambda Object (of ast )"""
    parsedBody = lambdaParse(_String)
    parsedBody = parsedBody.body
    return parsedBody
# ---------------------------------------------------
# Parsing a Lambda Body (of type BinOp: Left Op Right )

#--------------------------
# deParseLambda

def deParseLambda(parsedBody, astOperators):
    """ deparses the lambda an ast object of type BinOp, into its preliminary elements  """

    # Op
    # parsedBodyOp : checks if operation is valid (belongs to one of the ast operators )
    parsedBodyOp = parsedBody.op  # i.e. Pow()

    #isOperator , operation = None, None
    # op : Parsed operator (of lambda body) is being checked , against a  given astOperator

    isOperator, operation = checkIfAstOperator(
        parsedBodyOp, astOperators)  # checks if operator of choice exists

    # then, if it's a valid operator (valid means it's one of the accepted ast Operations)
    if not isOperator:
        print("check the input operation, then try again")

    elif isOperator:

        # Left
        parsedBodyLeftValueSliceValue = parseLeftBody(parsedBody)

        # Right
        parsedBodyRightValue = parseRightBody(parsedBody)

        print("parsing BinOp result:")
        print("parsedBodyRightValue = ", parsedBodyRightValue)  # Parse Right
        print("parsedBodyLeftValueSliceValue",
              parsedBodyLeftValueSliceValue)  # Parse Left
  
        return parsedBodyLeftValueSliceValue, operation, parsedBodyRightValue

#--------------------------

