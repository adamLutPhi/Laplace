from mpmath import *
from math import exp

"""
#Q1. how to display the error in mpMath
#Q2. how to calculate the error, in Trapizodal Integration method ?

#Q3. how can we verify which function is closer to the real (trap or Quad), true `value` of approximation?
#a. Use verification method, perhaps?

"""
###########################
# MISC. ROUTINES #
###########################

def isIndexValid(x, arr, margin=0):
    """ checks if index is valid """ 
    condition = x <= len(arr) - margin
    return condition
    raise ValueError # value of the given index is invalid
def isScalar(x):
    """ checks if object `x` is scalar , i.e. has a cardinality of exactly 1 item """
    #1. check type
    _type = str(type(x))
    #2. check condition
    condition = _type == "int" or _type == "float" #unary condition
    ##DEBUG:
    if condition :
        print("variable " + str(x) +" is a scalar (int or float)")
        
    elif not condition:
        print("variable " + str(x) +" is not a scalar (int or float) ")
    # condition = len(x) == 1
    return condition
    raise TypeError # the type of object `x` is invalid



def getTuple(x, start=0):
    """gets a tuple, if index is valid  """
    # check index is valid
    index_valid = isIndexValid(start, x)
    if index_valid:
        return x[start], x[start+1]
    elif not index_valid: # fallback condition
        # worst case: return an invalid cartesian  point index(-1 , -1)
        return (-1, -1)
    
    raise TypeError # the type of object `x` is invalid

def checkTuple(x):

    if x ==(-1, -1):
        return -1;
    else:
        return x
    raise ValueError


def handleTuple(x):
    condition = False
    if x == -1 :
        print("Fall-back condition activated ")
        return True
    return condition
    raise ValueError("ValueError: please recheck input, then retry")

        
def diff(xVal,yVal, kernel = lambda x,y: 3*x*y + 2*y - x):
    """ returns the differentiation of a `kernel` """
    res = 0.0
    is_x_scalar = isScalar(xVal) #<-
    is_y_scalar = isScalar(yVal)

    if is_x_scalar:
        
        #res = diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (0,1))

        res = diff( kernel , (0.25, 0.5), (0,1))
        res =  diff( kernel , xVal , yVal ) 
    return res

#Differentiation

def differentiate(kernel =lambda x: x**2 + x ,x=1.0):
    """ prints lambda , returns lambda """
    print(diff(lambda x: x**2 + x, x) )
    res = f(1.0)
    dres = diff(res)
    
    f = lambda x :  kernel  #(lambda x: x**2 + x)
    print("f", f)
    print("dres = ",dres)
    return dres

def f(x):
    """ lambda function of single variable, x """
    #return x * 2
    return x**2 + x

# Demo:
print("Debug: checkTuple(-1, -1)")
x = checkTuple((-1, -1)) # -1
x = handleTuple(x)



print("f'() = ", diff((0.25, 0.5) ,(0,1)) )

def kernel(x):x: x**2 + x
# differentiate
f = lambda x :  kernel(x)



print("f1 = ",f(1) )
# res = f(x) # wount be displayed # erroneous
res = lambda x: x**2 + x #, 1.0   #erroneous
res2 = diff(lambda x: x**2 + x, 1.0, 2) #<------

#res2 = iterator(lambda x: x / 4 + 12, 100 )
print("res1 = ",res(1) ) #works
#print("res2 = ",res2(1) ) #erroneous 

""" f'(x) ; where x = 1.0 , 3*x*y + 2*y - x, x-Axis = (0.25, 0.5), y-Axis = (0,1) """
res = diff(lambda x,y: 3*x*y + 2*y - x, (0.25, 0.5), (0,1))
print("res = ", res )

# print("diff = " ,diff(res(1.0)) )

#print( , 5 ) )

# print("res() = ", res(x))  #err
# print("res() = ", res(1.0)) #err

print("f(1.) = ", res)

""" f'(x) ; where x = 1.0 , f(x) = x**2 + x """
dres = diff ( lambda x: x**2 + x, 1.0) #ok 
print("lambda x: x**2 + x, 1.0 = df(1.) = ", dres) #3.0

#print("res = ",res ) # ok #2.75

#print("f = ", f(x)) 
#f = mpmath.diff(lambda x: x**2 + x, 1.0)

# print("f = ", f(x)) #error 
#f(x) ; where x = 1.0
print("lambda x: x**2 + x, 1.0 = f = ", f(1.0))
#diff(lambda x: x**2 + x, 1.0)


#Integration


#1. set config(uration)s
mp.dps = 15  ; mp.pretty = True; error=True;

config = [mp, error ]

# Integration

## set interval:
interval1 = [0, pi]
print("Integration, using the Quadrature method (quad):")


#standard quadrature 
q = quad(sin, interval1 )
print(f"quad Integral: q1: Int[0, pi]( sin(x)) = {q}")  

# Example: 2D Integral:

f = lambda x, y : cos(x+y /2)
interval2 =[-pi/2, pi/2]
q2 = quad(f, interval2 , interval1  )
print(f"quad Integral: q2: Int[0,pi](cos(x+y /2))= {q2}")
print(f"type = {type(q2)}" )
print(f"ae = {q2.ae}")
print("with error: +/- {q2.ae}")


#trapeziodal rule

def trapizoidal(interval):
    """ source: @numericalmethodsguy
        https://www.youtube.com/watch?v=Xo9RzN5OFBo
    Uses a point-wise integration on the interval= [a,b], to get an approximate Integration value  where:
    -a : min interval
    -b : max interval
    formula :
    [width of Interval] * [the Average Value ]

    
    """
    a = interval[0]
    b = interval[1]
    Integration = (b - a ) * ( f(a) + f(b) )/2
    return Integration
    raise
    BaseException("ERROR: Trapezoidal Integration : Unexpected error occured, please recheck input, then try again ")

###########################
# INTEGRATION    ROUTINES #
###########################

#general formula
#Int[a , b]  Lambda 
#Int[0.1 , 1.3] 5 * exp(-2* x) dx
#Interval
a = 0.1; b = 1.3;

#Q. How to use `Trapezoidal Rule` to find out the approximate value of this integral ?
#A. `Trapezoidal`: any function, Integratable, from a to b, can be approximated by the formula:
#     - need for fundamental measure of the integration error

intervalAb = [a,b]
def integrateQuadTanhSinh(f, intervalAb , method = 'tanh-sinh', verbose=True):
    
    return quad(f, intervalAb , method = 'tanh-sinh', verbose = verbose)
    #return quadts(f, intervalAb )

def integrateQuadGaussLegendre(f, intervalAb , verbose=True):

    return quad(f, intervalAb , method = 'gauss-legendre', verbose=verbose)
    #return quadgl(f, intervalAb )

intervalAb = [0,1]
#quad(f, [0,1], method='tanh-sinh')  # Good
#quad(f, intervalAb , method = 'gauss-legendre', verbose=True)
#res1 = integrateQuadGaussLegendre(f, intervalAb)
#print(res1.epsilon)


#an important lambnda function [Integrand]
#Integrand: Lambda Function, f 
f = lambda x: 5 * exp(-2* x) # dx

# This function has lots of Applications, in differeny `Dynamical Systems`
#(circuits, spring mass systems, ... )


Integration = trapizoidal(intervalAb) #TODO: Iterative (if applicable)
print(f"Integration(trapizoidal) = {Integration}" )

integrationQuad = quad(f, intervalAb, verbose=True )

                    
print("IntegrateQuadGaussLegendre = ",integrateQuadGaussLegendre(f, intervalAb ))
print("integrateQuadTanhSinh = ",integrateQuadTanhSinh(f, intervalAb ))
print("quadts = ",quadts(f, intervalAb ))
print("quadgl = ",quadgl(f, intervalAb ))


print(f"Integration(Quad) = {integrationQuad}" )
