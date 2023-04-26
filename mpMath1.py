from mpmath import *


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

# Demo:
print("Debug: checkTuple(-1, -1)")
x = checkTuple((-1, -1)) # -1
x = handleTuple(x)

        
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

print("f'() = ", diff((0.25, 0.5) ,(0,1)) )
#Differentiation

def differentiate(kernel =lambda x: x**2 + x ,x=1.0):
    print(diff(lambda x: x**2 + x, x) )
    res = f(1.0)
    dres = diff(res)
    
    f = lambda x :  kernel  #(lambda x: x**2 + x)

    return f

differentiate
f = lambda x :  kernel(x)
def f(x):
    """ lambda function of single variable, x """
    #return x * 2
    return x**2 + x
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


#set config(uration)s
mp.dps = 15  ; mp.pretty = True; error=True;

config = [mp, error ]

#Integration
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

""" credit: @cyrusgeyer """

def unnamedFunction(lambdas,parameters, max_lmbd = 32):
    #max_lmbd = 32
    lmbds = range(1, max_lmbd + 1)      
    log_moments = []
    
    for _lambda in lambdas:
        log_moment = 0
        for q, sigma, T in parameters:
            log_moment += compute_log_moment(q, sigma, T, lmbd)
            log_moments.append((lmbd, log_moment))
    eps, delta = get_privacy_spent(log_moments, target_delta=delta)
    return eps, delta


#trapeziodal rule
from math import exp

#an important lambnda function [Integrand]
#Integrand: Lambda Function, f 
f = lambda x: 5 * exp(-2* x) # dx

# This function has lots of Applications, in differeny `Dynamical Systems`
#(circuits, spring mass systems, ... )

#general formula
#Int[a , b]  Lambda 
#Int[0.1 , 1.3] 5 * exp(-2* x) dx
#Interval
a = 0.1; b = 1.3; 
#Q. How to use `Trapezoidal Rule` to find out the approximate value of this integral ?
#A. `Trapezoidal`: any function, Integratable, from a to b, can be approximated by the formula:
#     - need for fundamental measure of the integration error
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

intervalAb = [a,b]
Integration = trapizoidal(intervalAb) #TODO: Iterative (if applicable)
print(f"Integration(trapizoidal) = {Integration}" )

integrationQuad = quad(f, intervalAb, verbose=True )

def integrateQuadTanhSinh(f, intervalAb , method = 'tanh-sinh', verbose=True):
    
    return quad(f, intervalAb , method = 'tanh-sinh', verbose = verbose)
    #return quadts(f, intervalAb )

def integrateQuadGaussLegendre(f, intervalAb , verbose=True):

    return quad(f, intervalAb , method = 'gauss-legendre', verbose=verbose)
    #return quadgl(f, intervalAb )

res1 = integrateQuadGaussLegendre(f, intervalAb)
#print(res1.epsilon)


                    
print("IntegrateQuadGaussLegendre = ",integrateQuadGaussLegendre(f, intervalAb ))
print("integrateQuadTanhSinh = ",integrateQuadTanhSinh(f, intervalAb ))
print("quadts = ",quadts(f, intervalAb ))
print("quadgl = ",quadgl(f, intervalAb ))


print(f"Integration(Quad) = {integrationQuad}" )
#1. how to display the error in mpMath
#2. hwo to calculate the error, in Trapizodal Integration method ?

#q3. how can we verify which function is closer to the real (trap or Quad), true `value` of approximation?
#a. Use verification method, perhaps?


###########################
# MULTIPRECISION ROUTINES #
###########################


def pdf_gauss_mp(x, sigma, mean):
  return mp.mpf(1.) / mp.sqrt(mp.mpf("2.") * sigma ** 2 * mp.pi) * mp.exp(
      - (x - mean) ** 2 / (mp.mpf("2.") * sigma ** 2))


def integral_inf_mp(fn):
  integral, _ = mp.quad(fn, [-mp.inf, mp.inf], error=True)
  return integral


def integral_bounded_mp(fn, lb, ub):
  integral, _ = mp.quad(fn, [lb, ub], error=True)
  return integral


def distributions_mp(sigma, q):
  mu0 = lambda y: pdf_gauss_mp(y, sigma=sigma, mean=mp.mpf(0))
  mu1 = lambda y: pdf_gauss_mp(y, sigma=sigma, mean=mp.mpf(1))
  mu = lambda y: (1 - q) * mu0(y) + q * mu1(y)
  return mu0, mu1, mu

def compute_a_mp(sigma, q, lmbd, verbose=False):

  lmbd_int = int(math.ceil(lmbd))
  condition = lmbd_int == 0
  if condition:
    return 1.0
  elif not condition:
        return  lmbd_int
  else: raise ValueError


def compute_b_mp(sigma, q, lmbd, verbose=False):
  lmbd_int = int(math.ceil(lmbd))
  if lmbd_int == 0:
    return 1.0

# __repr__

#DEMO 1
def demo():

    a_lambda = integral_inf_mp(a_lambda_fn)
    a_lambda_first_term = integral_inf_mp(a_lambda_first_term_fn)
    a_lambda_second_term = integral_inf_mp(a_lambda_second_term_fn)

    if verbose:
        print( "A: by numerical integration {} = {} + {}".format(
        a_lambda,
        (1 - q) * a_lambda_first_term,
        q * a_lambda_second_term) ) 

    return _to_np_float64(a_lambda)

# DEMO2:
def demo2():
    mu0, _, mu = distributions_mp(sigma, q)

    b_lambda_fn = lambda z: mu0(z) * (mu0(z) / mu(z)) ** lmbd_int
    b_lambda = integral_inf_mp(b_lambda_fn)

    m = sigma ** 2 * (mp.log((2 - q) / (1 - q)) + 1 / (2 * (sigma ** 2)))
    b_fn = lambda z: ((mu0(z) / mu(z)) ** lmbd_int -
                    (mu(-z) / mu0(z)) ** lmbd_int)
    if verbose:
        print ("M =", m)
        print ("f(-M) = {} f(M) = {}".format(b_fn(-m), b_fn(m)) ) 
    assert b_fn(-m) < 0 and b_fn(m) < 0

# DEMO3:
def demo3():
    b_lambda_int1_fn = lambda z: mu0(z) * (mu0(z) / mu(z)) ** lmbd_int
    b_lambda_int2_fn = lambda z: mu0(z) * (mu(z) / mu0(z)) ** lmbd_int
    b_int1 = integral_bounded_mp(b_lambda_int1_fn, -m, m)
    b_int2 = integral_bounded_mp(b_lambda_int2_fn, -m, m)
    a_lambda_m1 = compute_a_mp(sigma, q, lmbd - 1)
    b_bound = a_lambda_m1 + b_int1 - b_int2

    if verbose:
        print("B by numerical integration", b_lambda)
        print("B must be no more than    ", b_bound)
        assert b_lambda < b_bound + 1e-5
    return _to_np_float64(b_lambda)

# _compute_delta
def _compute_delta(log_moments, eps):
  """Compute delta for given log_moments and eps.
  Args:
    log_moments: the log moments of privacy loss, in the form of pairs
      of (moment_order, log_moment)
    eps: the target epsilon.
  Returns:
    delta
  """
  min_delta = 1.0
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    if log_moment < moment_order * eps:
      min_delta = min(min_delta,
                      math.exp(log_moment - moment_order * eps))
  return min_delta

# _compute_eps
def _compute_eps(log_moments, delta):
  """Compute epsilon for given log_moments and delta.
  Args:
    log_moments: the log moments of privacy loss, in the form of pairs
      of (moment_order, log_moment)
    delta: the target delta.
  Returns:
    epsilon
  """
  min_eps = float("inf")
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
  return min_eps


def compute_log_moment(q, sigma, steps, lmbd, verify=False, verbose=False):
  """Compute the log moment of Gaussian mechanism for given parameters.
  Args:
    q: the sampling ratio.
    sigma: the noise sigma.
    steps: the number of steps.
    lmbd: the moment order.
    verify: if False, only compute the symbolic version. If True, computes
      both symbolic and numerical solutions and verifies the results match.
    verbose: if True, print out debug information.
  Returns:
    the log moment with type np.float64, could be np.inf.
  """
  moment = compute_a(sigma, q, lmbd, verbose=verbose)
  momentnp = np.inf
  if verify:
    mp.dps = 50
    moment_a_mp = compute_a_mp(sigma, q, lmbd, verbose=verbose)
    moment_b_mp = compute_b_mp(sigma, q, lmbd, verbose=verbose)
    np.testing.assert_allclose(moment, moment_a_mp, rtol=1e-10)
    if not np.isinf(moment_a_mp):
      # The following test fails for (1, np.inf)!
      np.testing.assert_array_less(moment_b_mp, moment_a_mp)
  if np.isinf(moment):
    return momentnp #np.inf
  else:
    momentnp = np.log(moment) * steps
    return momentnp #np.log(moment) * steps

def get_privacy_spent(log_moments, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from log moments.
  Args:
    log_moments: array of (moment_order, log_moment) pairs.
    target_eps: if not None, the epsilon for which we would like to compute
      corresponding delta value.
    target_delta: if not None, the delta for which we would like to compute
      corresponding epsilon value. Exactly one of target_eps and target_delta
      is None.
  Returns:
    eps, delta pair
  """
# Test
##assert

## assert (target_eps is None) ^ (target_delta is None)
## assert not ((target_eps is None) and (target_delta is None))

def testTargetEps(target_eps,log_moments,target_delta):
    if target_eps is not None:
        return (target_eps, _compute_delta(log_moments, target_eps))
    else:
        return (_compute_eps(log_moments, target_delta), target_delta)


