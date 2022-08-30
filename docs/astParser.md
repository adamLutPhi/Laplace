
# Laplace Documentation contents 

# Methods

## def parseLambda(_String: str)
    **input:**
    _String : str

    **output:**
    paresdBody

    """ parses lambda from string to Lambda Object (of ast )"""
    parsedBody = lambdaParse(_String)
    parsedBody = parsedBody.body
    return parsedBody


parses a `String` object
Parsing a Lambda Body ( `BinOp` : `Left`, `Op` `Right` )

Calls `lambdaParse(_String)`, returning  a parsed lambda body, `parsedBody`  (of type `BinOp` : which has a further: `Left`, `Op`, `Right` parts )

#------------
## Example

# def lambdaParse(_string='lambda x: x**2 '):
    """parses string that has a lambda into an ast Lambda Object """

    parsedtree = ast.parse(_string, mode='eval')
    # recalculate the `location information`
    # fix missing locations, rewrites the expression Tree
    treeExpression = fix_missing_locations(
        RewriteName().visit(parsedtree))  # new_tree
    _lambda = treeExpression.body
    return _lambda
    

# Helper Function 

def lambda2ParsedBody(Lambda):
    # Lambda ---> parsedBody
    print("lambda tree = ", Lambda)
    parsedLambda = (ast.parse(Lambda))

    #print("parsed = ",parsed)
    parsedBody = parsedLambda.body  # BinOp
    print("parsedBody = ", ast.dump(parsedBody))
    return parsedBody


lambda2ParsedBody(Lambda)




def deParseLambda(parsedBody, astOperators):
    """ deparses the lambda object of type BinOp, into its preliminary elements: Left, op, Right  """

    # Op
    # parsedBodyOp : checks if operation is valid (belongs to one of the ast operators )

    #
    parsedBodyOp = parsedBody.op  # Pow() #

    #    print("IsOperator ",IsOperator)
    #    print("operation ",operation)
    #isOperator , operation = None, None

    # op : Parsed operator (of lambda body) is being checked , against a  given astOperator
    isOperator, operation = checkIfAstOperator(
        parsedBodyOp, astOperators)  # checks if operator of choice exists


    # if it's a valid operator (valid means it's one of the accepted ast Operations)
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
        # , [isOperator , operation ]
        return parsedBodyLeftValueSliceValue, operation,parsedBodyRightValue

# `deParseLambda(parsedBody, astOperators)`

deparses a lambda using its `parsedBody`, & `astOperators`

def switchOp(astOp):
    """ switching a parsing operator, into a mathematical Operator """ 
    #pass
    if astOp == ast.Pow:
        return math.pow
    elif astOp == ast.Add:
        return math.Add
    elif astOp == ast.Sub:
        return math.Sub
    elif astOp == ast.Mul:
        return math.mul
    elif astOp == ast.Div:
        return math.Div 
    
    `switchOp(astOp)`

    def checkifLambda(Lambda):
    if type(Lambda) == ast.Lambda:
        return True
    
    checks input lambda is of type ast `Lambda`
# ------------------
# Right

def parseRightBody(parsedBody):  # Constant Object (of _ast module)
    """ parsed the Right-hand side of a 
    return parsedBodyRightValue

Returns a `parsedBodyRightValue` which is a ast `Constant` , leading to a terminal value, i.e. `int`


# ---------------------
# Left

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
    return parsedBodyLeftValueSliceValue

    print("type ", type(parsedBodyLeftValueSliceValue))  # <class 'str'>

    return parsedBodyLeftValueSliceValue


    Extracts the value of the left part of a lambda `parsedBody`
    then takes its ast `slice` object, in order  to retrieve the variable symbol name; in this example lambda , it is `x`



# Refactored code
# ----------
## Tutorial 

    _string =  "lambda x: x ** 5"


    parsedBody = parseLambda(_string)

    = deparseLambda( parsedBody , astOperators)
    