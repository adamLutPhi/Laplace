# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print('Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# this won't work properly
def degree(poly):
    while poly and poly[-1] == 0:
        poly.pop()  # normalize
        return len(poly) - 1


# big O = # not optimized - not at all!
def poly_div(N, D):
    dD = degree(D)
    dN = degree(N)
    if dD < 0: raise ZeroDivisionError
    if dN >= dD:
        q = [0] * dN
        while dN >= dD:
            d = [0] * (dN - dD) + D
            mult = q[dN - dD] = N[-1] / float(d[-1])
            d = [coeff * mult for coeff in d]
            N = [coeff * mult for coeff in d]
            dN = 0  # de: to be continued


import numpy as np

# lists in numpy faast: uses fixed types
# i.e. 3 x 4 matrix integer is converted & stored as bytes binary cached into Int32 (can specify all 4 bytes , Int16,
# object value: has own bits, reference cound ( many counts ) , size (of that integer value)

# numpy use less bytes Iin memory) faster in that regard
# when iterating item in numpy no need for `type checking`
# contiguous Memory L array-structure is a pc memory - imagine  & list is scattered aroundd
# blocks aren't necessary close, to each ither
# list (or arr) contain pointerrs to actual locations of memory
## (bounces around) - not super fast
# Numpy uses contiguous memory: also distort start, total size, type (memory block) much easier
# 1. cpu sIMD vector (single instruction, multiple data) Processing
# perform computations
# 2.utilize more cache: effective utilization (go back reload that in cache)
# how lists different from numpy : insert, ,delete, append, concatenation # numpy even much more

a = [1, 3, 5]
b = [1, 2, 3]
# a*b

a = np.array([1, 3, 5])
b = np.array([1, 2, 3])

print(a * b)  # [ 1  6 15] # numpy allows item-wise computation
# matlab replacement[scipy has even more math]
# backend: pandas, connect 4, Digital photography)
# plots: matplotlib
# ML : 1 key libraries (concepts) are tensors (physicss ) in tf tensor

a = np.array([1, 2, 3])
print(a)
# list, within a list

b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])  # array of an array
print(b)
# size (Dimension
b.ndim
# b.Shape
b.shape

# get (d)type

a.dtype

a = np.array([1, 2, 3], dtype='int16')

# Get size
a.itemsize  # 4

# Get total size
# a.size #3
a.itemsize
b.itemsize
a.size * a.itemsize  # 3 *(4) = 12

# Get total size
a.nbytes  # 12

# Acessing / changin  specifict elements

a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])

a.shape
# R,C

print(a[1, 5])

# get a specific row
print(a[0, :])

# a[:, column]

print(a[:, 2])

# [1 2 3 4 5 6 7] # FROM INDEX 1: 2 till -1 (B4 last i.e. 6)
# [2 4 6] # move 2 steps i.e. from 2 +2 =t= 4 from it +2 =to= 6
print(a[0, 1:6:2])  # start:end:step_size # same
print(a[0, 1:-2:2])
print(a[0, 1:-1:2])  # 2  , 2+2=4, 4+2 = 6
# assert a[0,1:-1:2] == a[0,1:6:2]

# set a value
a[1, 5] = 20
print(a[1, 5])

a[:, 2] = 5
print(a[:, 2])

# a[:,2] = [1,2]
print(a[:, 2])  # 2nd column
print(a)

# 3D example:

b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(b)

# get specifivc elemenet (work outside in)

b[:, 1, :]
# Get specific element
b[:, 0, :]

print(b[:, 0, 0])
print(b[0, 1, 1])

# replace
print(b[:, 1, :])
# change content
b[:, 1, :] = [[9, 9], [8, 8]]

print(b[:, 1, :])
print(b)

# initialize arrays

# all 0s arays

print(np.zeros((2, 3)))  # 3d
print(np.zeros((2, 3, 3)))  # 3d

print(np.zeros((2, 3, 3, 4)))  # 4d - crazy

# any other number
print(np.full((2, 2), 99, dtype='float32'))  # takes 3 input parameters

# full_like method  takes an already built shape, reuse that

np.full_like(a.shape, 4)  # inherit a's shape,

np.full_like(a, 4)  # inherit a: all a are 4 '

# Random integer values

print(np.random.randint(7, size=(3, 3)))  # low, high

print(np.random.randint(4, 7, size=(3, 3)))

print(np.random.randint(-4, 8, size=(3, 3)))  # allow negative numbers

np.identity(5)  # identity is a square matrix

# Repeat an array (say, n times)

# arr = np.array([1,2,3])
# r1 = np.repeat(arr,3) #repeat each numer in arr, 3 times
# print(r1)

# r1 = np.repeat(arr,3,axis=1)
# print(r1)

output = np.ones((5, 5), dtype='int32')
print(output)

z = np.zeros((3, 3))
z[1, 1] = 9
print(z)
output[1:4, 1:4] = z

# beware array copy

a = np.array([1, 2, 3])
b = a
b[0] = 100
print(a)

# Mathematics

a = np.array([1, 2, 3, 4])
print(a)

print(a + 2)

a += 2

print(a)

a - 2

a * 2

a / 2

b = np.array([1, 0, 1, 0])
a + b

a ** 2

# take the sin
np.cos(a)

# for a lot more (scipy  # routines math

# linear algebra

a = np.ones((2, 3))
print(a)

a = np.full((2, 3), 2)
print(b)
# a*b
# a.*b
# a*.b
# np.dot(a,b) # error shapes (2,3) and (4,) not aligned:

##print(np.matmul(a,b))

# find the determinant
c = np.identity(3)
print(c)
print(np.linalg.det(c))

a = np.ones((2, 3))
print(a)

b = np.full((3, 2), 2)
print(b)

# linear algebra cols first matri  == rows of the 2nd

# end up by 2 x 2 matrix at the end

print(np.matmul(a, b))  # dtype='Int16'))

# find the determinant
c = np.identity(3)
np.linalg.det(c)

# numpy
# 1 Determinant det
# 2 Trace
# 3 Singular Vector Decomposition
# 4 Eighenvalues
# 5 Matrix Norm (metric

# 6 Inve

print(type(10))


# print(type(10) == int)

# print(type(10.2) ==float)

def isNumeric(x):
    """
        Checks if a value has an actual value
    :param x:
    :return:
    """
    # res = None
    if type(x) == int or type(x) == float:
        return True

    else:
        return False


if type(10) == int or type(10.2) == float:
    res = True

else:
    res = False

# print(res)

res1 = isNumeric(10)
res2 = isNumeric("10")

print("res1 = ", res1)
print("res2 = ", res2)
# numeroic means
# <class 'int'>

# <class 'd'>
x = 1000
print()


# def sumDigits(x)

def _div(a, b):
    """
        returns the subtraction of 2 numbers

    :param a: first large number
    :param b: second small number
    :return: their subtraction (is positive)
    """
    if a > b:
        return a - b


print(_div(10, 2))

n = None
if type(n) == 'NoneType':
    print(True)
else:
    print(False)


def _div(wholeNumber, div, n=None):  # b, n=None):
    subtraction = wholeNumber
    # @    if isNumeric(n)==True :  # is not Nonw '#n != None:
    i = 0

    for i in range(n):
        if wholeNumber > div:
            subtraction -= div
    return subtraction
    # else:
    # return _div(wholeNumber, div)


# TODO: Uncomment me
# print(_div(10, 2,None)) #print(_div(10,2))


def divby(x, div=3):
    # return
    rem = 0
    rem = math.fabs(x) - math.fabs(div)
    if x % div == 0 and rem % div == 0:
        return True
    else:  # if (x % div) != 0 and  rem % div !=0:
        return False


print(divby(21, 3))
val = 20 / 3

requiredAddition = round(val) - val
# print( requiredAddition) #0.33333333333333304

if type(requiredAddition) == float:
    requiredAddition > 0 and requiredAddition < 1
    requiredAddition = 1


def _try(x):
    return


print(divby(21, 3))
rem = 0
# output, progrsm:
if divby(10, 2):
    print("Not Implemented yet")  # rem = _div(10, 2,None) #todo: Uncomment Me


# print(rem) #Uncomment me

def get10Factors2(num):
    _listSub = []
    left = 0
    # if powerBase >1 :
    powerBase = math.log10(num)
    print("init powerBase = ", powerBase)
    intBase = 1
    # while int(powerBase)>1: #intBase > 0: #powerBase > 1 and num >= 0:  # powerBase

    if powerBase > 1:

        powerBase = math.log10(num);
        print("powerBase", powerBase)  # init powerBase =  3.278753600952829
        # Todo: how many zeroes we subtracted
        intBase = int(powerBase) - 1;
        print("initBase = ", intBase)
        # if intBase == 0: intBase += 1 #worst case int is 0: augment by 1 # el
        powerBase -= intBase  # subtract
        print("powerBase = ", powerBase)
        if intBase < 0: print(-1)  # erroneous input
        factor1 = 10 ** intBase
        _listSub.append(factor1)
        print("10^initBase = ", 10 ** intBase)  # 100
        factor2 = int(calcOtherFactor(num, factor1))
        _listSub.append(factor2)
        print("leaving loop, powerBase = ", int(powerBase))
    if powerBase > 1:
        get10Factors2(max(factor1, factor2))
    else:
        return factor1, factor2, _listSub

    # ---
    #    if intNumber == 1: return _listSub
    # addition:
    #  intNumber -= 1
    #  if intNumber ==0: intNumber = 1
    #   print("intNum = ", intNumber)
    #  sub = 10 ** intNumber  # 10**3 = 1000 not 9!
    # TODO: register the sub (AS factor1)
    #  _listSub.append(sub)

    #  print("left = ", left)


# if powerBase <= 1 or powerBase >= 0:
# check division
#    return powerBase  # OR _div(res) #
#    print("powerBase = ", powerBase)
# return _listSub


def get10Factors(num):
    # intNumber = 0
    # res = math.log10(num)
    _listSub = []
    # left = 0
    # if powerBase >1 :
    powerBase = math.log10(num);
    print("init powerBase = ", powerBase)
    while powerBase > 1 and num >= 0:  # powerBase
        powerBase = math.log10(num)
        intNumber: int = int(powerBase)  # 3
        # addition:
        intNumber -= 1
        if intNumber == 0: intNumber = 1
        print("intNum = ", intNumber)
        sub = 10 ** intNumber  # 10**3 = 1000 not 9!
        # TODO: register the sub (AS factor1)
        _listSub.append(sub)
        num = num / sub  # Accepted
        # print("sub = ", sub)
        # num -= sub  #  correct: 900 (1900 - 1000)
        print("num", num)
        powerBase = math.log10(num)  # best final move  #feedback: plugin output as the next input argument
        print("powerBase = ", powerBase)
        # powerBase -= sub #1 1900 - 900
        # print("powerBase =",powerBase)
        left = num  # 1.9
        # needs this: reassign powerBase
        # num -= sub # num =  1891
        print("left = ", left)

    # print(n)

    # el
    if powerBase <= 1 or powerBase >= 0:
        # check division
        return powerBase  # OR _div(res) #
        print("powerBase = ", powerBase)
    return _listSub


def factor10Subtracting(num):
    # intNumber = 0
    # res = math.log10(num)
    _listSub = []
    # left = 0
    # if powerBase >1 :
    powerBase = math.log10(num);
    print("init powerBase = ", powerBase)

    while powerBase > 1 and num >= 0:  # powerBase
        powerBase = math.log10(num)
        intNumber: int = int(powerBase)  # 3
        # addition:
        intNumber -= 1
        if intNumber == 0: intNumber = 1  # this guarantees it runs, at least, once
        print("intNum = ", intNumber)
        sub = 10 ** intNumber  # 10**3 = 1000 not 9!
        # TODO: register the sub (AS factor1)
        _listSub.append(sub)
        num -= sub  # # tesing the utility factor of this function
        # num = num / sub  # Accepted
        # print("sub = ", sub)
        # num -= sub  #  correct: 900 (1900 - 1000)
        print("num", num)
        powerBase = math.log10(num)  # best final move  #feedback: plugin output as the next input argument
        print("powerBase = ", powerBase)
        # powerBase -= sub #1 1900 - 900
        # print("powerBase =",powerBase)
        left = num  # 1.9
        # needs this: reassign powerBase
        # num -= sub # num =  1891
        print("left = ", left)


n = None
if n == None:
    print(True)
elif n != None:
    print(False)

print(get10Factors(1900))  # returns correct number : 9 #TODO: return  also the created power


# 1 removes 1000 from 1900 , done , returns 900
# 2 in: 900 log10 = 2 , 10^2==


def isCommonFactor(amount, factor):
    if amount > factor and amount % factor == 0:  # amount - factor > 0
        return True
    else:
        return False


# 1 isCommonFactor
print(isCommonFactor(1900, 100))  # true


# 1900%100
# calcOtherFactor
def calcOtherFactor(amount, factor):  # propose: ValcOtherFactorDiv
    """divides amount by factor  """
    if amount > factor and factor != 0 and amount % factor == 0:  # 3 factor is a denomintor. Impossible to be in denominator
        return amount / factor
    else:
        raise Exception("Error calculating ", amount, " / ", factor, "\nReason: factor does not Divde the amount ",
                        amount)


# def calcOtherBySub(amount, factor):


def isInteger(amount):
    """checks if an amount is an integer"""
    if float.is_integer(amount):
        return True
    else:
        return False


factor1 = 100  # calculated
factor2 = calcOtherFactor(1900, factor1)
print(float.is_integer(factor2))

# value to be checked:
print(factor1 * factor2 == 1900)  #

get10Factors2(1900)
# ------
powerBase = math.log10(1900)
print(math.log10(1900))  # init powerBase =  3.278753600952829
intBase = int(powerBase) - 1
print(intBase)
# if intBase == 0: intBase += 1 #worst case int is 0: augment by 1
# el

if intBase < 0: raise Exception("intBase =", intBase, " cannot be Negative")  # print(-1)  # erroneous input
factor1 = 10 ** intBase
print(10 ** intBase)  # 100
factor2 = int(calcOtherFactor(1900, factor1))

if isCommonFactor(1900, factor1):  # purose: rename to isCommonDivFactor
    print(int(calcOtherFactor(1900, factor1)))  # : print(-1 )

assert factor1 * factor2 == 1900


# ---------
# amount / factor

def isFloatAnInt(x):
    """checks if a float exactly equals the integer value of a number """
    return float.__int__(x) == int(x)


def isCommonFactorSub(amount, factor):
    # pass
    if amount > factor and factor != 0:  # 3 factor is a denomintor. Impossible to be in denominator
        return amount - factor


def calcOtherBySub(amount, factor):
    """calculates the amount left from
     subtracting factor from the total amount
    """
    if amount > factor and factor != 0:  # 3 factor is a denomintor. Impossible to be in denominator
        return amount - factor


factor = 100
if isCommonFactorSub(1900, factor):
    print(int(calcOtherBySub(1900, factor)))


# print(math.log10(1900 ))

# ------

def isEven(x):
    """ checks if a number is Even (divides by 2).

    This assumes (any assumptions )
     some time has elapsed in the connection timeout and
    computes the read timeout appropriately.

    If self.total is set, the read timeout is dependent on the amount of
    time taken by the connect timeout. If the connection time has not been
    established, a :exc:`~urllib3.exceptions.TimeoutStateError` will be
    raised.

    :return: Value to use for the read timeout.
    :rtype: int, float, :attr:`Timeout.DEFAULT_TIMEOUT` or None
    :raises urllib3.exceptions.TimeoutStateError: If :meth:`start_connect`
        has not yet been called on this object.
    """
    if x % 2 == 0:
        return True

    elif x % 2 != 0:
        return False


print("Even = ", isEven(10))

# ------

multiple2 = [4, 6, 8]  # 2*2 , 2*3, 2*4
multiple3 = [6, 9]
odds = [5, 7]

# ------
# def _loop(x):
#    i=0
#    for i in arange(x): # is last inclusive?
#        print("x = ",x)
#        i +=1
# _loop(3)


fac1, fac2, _list = get10Factors2(10 ** 6)
print(_list)


# --------
class laplaceTransform:

    # pass
    def __init__(self, fx, x):
        self.fx = fx
        self.x = x


def f(self, x):
    pass


def diffuse(self):
    pass


# l1 =laplaceTransform() # TODO: complete the most basic laplace Transform

def X(x):  # integral?
    return x  # todo: do sth more meaningful


def a(a):
    return a  # todo: do sth more meaningful


def f(x):
    res = 0
    if x >= 0:
        res = x
    elif x < 0:
        res = a(x) * X(x)
    return res


# else pass ;#raise("Unexpected Error - Something went wrong")


# def _calc(value):

#    math.log1p(value,)

# print()

math.log10(10 ** 3) == math.log(10, 10 ** 3)

base = math.log(3, 10 ** 3)
base = math.log(10, 10 ** 3)
print(base)

# examples of using the log:
print(math.log(3, 10 ** 3))
print(math.log(3, 10 ** 3))

_digits = int(math.log10(1000))  # 3 from log:learned how homay digits = 3+1 = 4

if float.__int__(math.log10(1000)) == int(math.log10(1000)):
    print(True)
else:
    print(False)

print(float.__int__(math.log10(1000)))  # ==  3

print((int)(math.log10(1000)))  # 3.0

int(math.log10(1000))  # orphaned

# checks if both amounts are equal
if ((int)(math.log10(1000)) == float.__int__(math.log10(1000))) == True:
    print(True)


# CHAIN Rule

def expandTerm(a, b):  # expands a,b st a/b
    """
    here we need to define 2 terms:
     definces a c TODO: how to define extension expansion (is there an automatable way to implement it?)


    :param a:
    :param b:
    :return:  the 2 terms a/c c/b
    """
    # a/b # TODO: calculate c, then call it here  #TODO: define term
    # calls c()
    c = 1
    return a / c, c / b


# ---

class Eqn:

    def __init__(self, Rhs, Lhs):
        self.Rhs = Rhs
        self.Lhs = Lhs
        self.rhsTerms = None
        self.lhsTerms = None

    class term:
        def __init(self):
            """
            :return: the variable x  , by default
            """
            self.x = x  # assigned value x , by default

            return self.x

        def __init__(self, variable=x):
            #  """or if you got another idea , maybe another sumbol variable representation """
            self.x = variable
            return self.x;

        def __init__(self, coefficient):
            self.coefficient = coefficient

            return self.coefficient * self.x

        def __init__(self, coefficient, power):
            self.coefficient = coefficient
            self.power = power
            return self.coefficient * self.x ** self.power

        def __init__(self, coefficient, variable=x):
            self.coefficient = coefficient
            self.variable = variable

            return self.coefficient * self.variable

        def __init__(self, coefficient, variable=x, power=1):
            self.coefficient = coefficient
            self.x = variable
            self.power = power

            return self.coefficient * self.x ** self.power

        def __init__(self, coefficient, power, variable=x):
            self.x = variable
            self.coefficient = coefficient
            self.power = power
            return self.coefficient * self.x ** self.power


# --------

class division:
    def __init__(self):
        self.numerator = 1  #
        self.denominator = 1

    def __init__(self, numerator):
        self.numerator = numerator
        self.denominator = 1

        def __init__(self, numerator, denominator):
            self.numerator = numerator
            self.denominator = denominator

    class limit:  # TODO: limit T to infinity on function

        # checks if function is convergent at som value
        def classic_isCongent(self, f, x):
            """ if limit exists on the right and on the left of a number, then limit `Has Got to` exist """
            # if
            pass
            #

        def classic_iscongergentRight(self, f, x):
            pass

        def classic_isConvergentLeft(self, f, x):
            pass


# class integral
# n = input()
# x =0
# overview: lambda is a function, just without binding it to a name like f


## _rtest = lambda n, x: (-x) * x*(n-1) (1)
## print("rtest: 1,2 = ", _rtest(1,2) )


class gammaFunction:
    """the general function of the Exponential """

    def __init__(self, n):  # input: n onlt
        # forms a lam
        self.Eqn_gamma = lambda x: lambda x: np.exp(-x) * x ** (n - 1)  # (1) #np.exp(-s*t) * (1)

        self.Eqn_gamma_n_1 = lambda x: np.exp(-x) * x ** (n - 1)  # *(1)  #TODO: change immediately

    def __init(self, n, s, t):  # n = 1

        self.Eqn_gamma = lambda s, t: np.exp(-s * t) * t ** (n - 1)  # *(1) #np.exp(-s*t) * (1)

        self.Eqn_gamma_n_1 = lambda s, t: np.exp(-s * t) * t ** (n - 1)  # *(1)  #TODO: change immediately

    #  if n == 1:
    #      self.integralL = lambda s: 1/s
    def integrate(self, a=0, b=1000):
        # def gamma(self,n):
        #   math.exp(-x)* x^n
        pass


# ---
# if f provided as a string , u have to build its  corresponding parser (ok ;))
class LaplaceTransform:
    def __init__(self, f):
        self.coeffInput == f  # TODO: parse input f (from string, to a complete lambda t
        self.laplaceFactor = lambda s, t: math.exp(-s * t)
        # if self.coeff == f #1:  # TODO:
        #   return
        # return e^-exp
        self.res = self.coeffInput * self.laplaceFactor

    def integrate(self):
        pass


# stringFormula = input("prompt: please enter a formula of your choice,  \nSupported ops: + - * / ^ ") #  with respect to time

stringFormula = " 56*x^2 + 100*x + 12  "

# trim spaces: first & last
stringFormula = stringFormula.lstrip()
stringFormula = stringFormula.rstrip()
print(stringFormula)
# split : using stringSplit
res = stringFormula.split(" ")  # now got coeffs & ops # or only numbers (at  end
f = float(res[-1])  # default behavior if token isn't symbolOperation or

# in general Input tokes have to be in the following form:
# number*, symbol, var,symbol, number,

print(f)

print(res)
# Regular expression handling
import re

t1 = res[1]
tokens = re.findall(r"[\w]", res[1])
# in general, better use re.search to find things matching a pattern (of choice ) i.w. pat = r'.*?\[(.*)].*'
# tokens = t1.split(["+","-","*","/","^"])
pat = r'.*?\[(.*)].*'  # @0 # found none
ops = r'+|-|^|*|//'
var = r'x'
digits = '\d+'  # works
print(res[0])
match = re.search(pat, res[0])
print(match)
match = re.search(digits, res[0])  # 56 # span (0, 2) # this returns the first match, unfortunately
print(res[0][2:])
# ---------
match = re.findall(digits, res[0])
ops = "['+'| '*'|'-']+"
Div = "/"
numeric = "[0-9]+"
alpha = "[a-zA-Z]+"

matchOps = re.findall(ops, res[0])
print(matchOps)
matchOps = re.search(ops, res[0])  # <re.Match object; span=(2, 3), match='*'>
print(matchOps)
print(matchOps.span())

# match =re.findall(ops,res[0])
# print(match)
# match = re.search(ops,res[1])
print(match)  # ['56', '2']

match = re.search(var, res[1])


# print(match) # None
# print(tokens)

def findOccurrences(s, ch):  # if 1 character, it works flawlessless
    return [i for i, letter in enumerate(s) if letter == ch]


op = '^'
print(findOccurrences(res[0], op))  # [4]
var = 'x'
print(findOccurrences(res[0], var))  # [3]
# --------


# piping using macropy  ;
"""
it has many advantages dis advantages 

pros:

1. no need for an infix function 
2. no overhead during runtime 

con: 1. takes a while to get loaded up 
"""


# Desired  parsed string into a list  i.e. ["56", "x^2", "+", "100", "*", "x"]
def limit(f):  f  # TODO: Implement a Convergence Algorithm (i.e. Direct Convergence)


def factorial(n):
    product = 1
    for i in range(n):
        product *= n
        n -= 1
    return product


def integrateString(f, val0, valInf):  # f  # TODO: Implement an Integration (Table Integration )

    if f == "t^n":  # lambda t,n : t^n) :
        return lambda n, s: factorial(n) / s ** (n + 1)  # note: the power sounds a bit overwhelming

    elif f == "e^at" or "e^a*t" or "e**(at)" or "e**(a*t)":
        return lambda s, a: 1 / s - a


# lambda (operator) equality
import operator

## vanilla example:
coco = operator.attrgetter('co_code', 'co_consts')
testCoco = ((lambda x: x + 2).__code__) == coco((lambda x: x + 2).__code__)  # True
# print("coco( x + 2) , x =2 ", # Idea
print(testCoco)


# MIT:
# (An Exponential) a: is a more general Complex Number

def linearfun(a, b, c):  a + (b * c)


def S(sigma, j, omega):  linearfun(sigma, j, omega)  # s is not just j *omega, but sigma + j*omega


# General Exp
##given s,t as input: generate a new equation of h
def expGen(s, t): lambda tao, h: integrate(h(tao) * math.exp(s * (t - tao)))


def simplifyExp(f, s, t):
    if f.__code__ == (lambda tao, h: integrate(h(tao) * math.exp(s * (t - tao)))).__code__:  # as exp(s*t)* exp(-s*tao
        # simplify the inside exp()
        return lambda tao, h: math.exp(s * t) * integrate(h(tao) * math.exp(-s * tao))
    # conclude: a complex eponential, with any complex number, s
    # would generate, as output, a complex exponential of the same form, multiplied by
    # (whatever this integral is ) [Note: this integral depends on what value of s is ]
    # what all of the integal can be reduced to Is some function, H(s),
    # that depends on the value s


def H(s):
    """representation of a `Response`, of complex variable  in a linear time
    which is a complex constant, Depending on s , multiplying the same function
    that excited the system"""

    # as exp(s*t)* exp(-s*tao)
    # simplify the inside exp()
    return lambda h, tao: integrate(h(tao) * math.exp(-s * tao))


# Eigenfunction Property: more generally, a complex exponential
# laplace is a transformation, on a time-function 4g35
# ( laplace transform), of that time function , it is a function of s
# it is the result of this transformation (of a respon se function), on x(t)
# Denoted as x(s)
# time domain: time function x(t)
# LaplaceTransform domain: function X(s)

# Function represents the Transform(ed) Pain
# Thew process of this mapping is very simiplar to the one that's given us the fourier Transform
def fourierTransform(x, t):
    return lambda j, omega: integrate(x(t) * math.exp(-j * omega * t))

print("fourierTransform(1,2)",fourierTransform(1,2))


# Notice: note if sigma = 0 then s = j*omega , hence LaplaceTransform == Fourier Transform

def fourierTransformGen():  # TODO: WARNING: Inputs are unused

    return lambda x, t, s: integrate(x(t) * math.exp(- s * t))


def fourierTransformGen(j, omega, sigma):
    return lambda x, t: integrate(x(t) * math.exp(- (sigma + j * omega) * t))


# in fact fourier transform may or may not converge : time function is absolutely integrable (we're Transforming)
# if it's a function grows exponentially : when we multiply it by this exponential factor
# (that's embodied in the laplace transform )
# that brings function down , for positive time
# we'll impose absolute integrability
# on the product of x times  e to the minus sigma t
# the fourier transform (of product ) may converge evemthought
# the fourier  transform x(t) doesn't
# even though numerator Polymoial has a root at s p
#  Example: laplace transform of the sum equals the sum of laplace transforms

# region of convergence , the real part of s, greater than -1: Re(s) > 1
def isAddition(f):
    response = None
    if f.__code__ == (f()).__code__:
        response = True, a, b
    elif f.__code__ != (f()).__code__:
        response = False
    else:
        raise Exception("Unexpected input detected : please do check, then try again later ")
    return response


def isSubtraction(f):
    response = None
    if f.__code__ == (Sub()).__code__:
        response = True, a, b
    elif f.__code__ != (Sub()).__code__:
        response = False
    else:
        raise Exception("Unexpected input detected : please do check, then try again later ")
    return response


def isMultiplication(f):
    response = None
    if f.__code__ == (Mul()).__code__:
        response = True, a, b
    elif f.__code__ != (Mul()).__code__:
        response = False
    else:
        raise Exception("Unexpected input detected : please do check, then try again later ")
    return response


def isDivision(f):
    response = None
    if f.__code__ == (Div()).__code__:
        response = True, a, b
    elif f.__code__ != (Div()).__code__:
        response = False
    else:
        raise Exception("Unexpected input detected : please do check, then try again later ")
    return response


def isLog10(f):
    response = None
    if f.__code__ == (Log10()).__code__:
        response = True, a, b
    elif f.__code__ != (Log10()).__code__:
        response = False
    else:
        raise Exception("Unexpected input detected : please do check, then try again later ")
    return response


def isExp(f):
    response = None

    if f.__code__ == (Exp()).__code__:
        response = True, a, b
    elif f.__code__ != (Exp()).__code__:
        response = False
    else:
        raise Exception("Unexpected input detected : please do check, then try again later ")
    return response


def isPow(f):
    response = None
    if f.__code__ == (Pow()).__code__:  # (lambda a, b: a ** b).__code__:
        response = True, a, b
    elif f.__code__ != (Pow()).__code__:  # (lambda a, b: a ** b).__code__:
        response = False
    else:
        raise Exception("Unexpected input detected : please do check, then try again later ")
    return response

# --
# Operations: 

_digits = int(math.log10(1000))  # 3 from log:learned how homay digits = 3+1 = 4

# _Add:lambda a,b: a+b
# print("Add(1,2)", _Add(1,2))

Sub: lambda a, b: a - b
Mul: lambda a, b: a * b
Div: lambda a, b: a / b
Exp: lambda a: math.exp(a)
Pow: lambda a, b: a ** b
Log10: lambda a: math.log10(a)

#_x: lambda x: _x

# Log10: lambda a,_x: math.log10(a*_x)
# print("Log10", Log10(x-1),2)
# print(_x(x))

"""
def Sub(): return lambda a,b: a-b
def Mul(): return lambda a,b: a*b
def Div(): return lambda a,b: a/b
def Exp(): return lambda a: math.exp(a)
def Pow(): return lambda a,b: a **b
def Log10(): return lambda a: math.log10(a)
"""


def check(f, operation=None, _flag=None):
    # operation = None
    # _flag = None
    result = None
    while True:
        result = isDivision(f)
        # _flag == isDivision(f)[0]==True #def isDivision(f: {__code__}) -> Union[Tuple[bool,
        # Union[List[int], ndarray, (a: Any) -> Any
        # , (x: {__pow__}) -> Any],
        # Union[List[int], ndarray, (x: Any) -> Any]], bool]
        if _flag == result[0] == True:  # _flag :
            operation = Div, result[1], result[2]  # result[1], result[2]  # True, a, b
            break

        result = isAddition(f)
        if _flag == result[0] == True:
            operation = isAddition(1), result[1], result[2]  # result[1], result[2]
            break

        result = isSubtraction(f)
        if _flag == result[0] == True:
            operation = Sub(), result[1] - result[2]  # result[1] - result[2] #result[1], result[2]
            break

        result = isMultiplication(f)
        if _flag == result[0] == True:
            operation = Mul(), result[1], result[2]  # result[1] * result[2]
            break
        result = isPow(f)
        if _flag == result[0] == True:
            operation = Pow(), result[1], result[2]  # math.pow( result[1], result[2])
            break
        return operation
    #


# -----------
# Not Good Enough: as it consideres f(1) == f(2) # 2 . not good if DataUsed (in lambda) Changed

# thus eyou can Compare the bytecode and constants (of each other
# ----------
def integrate(f):  # , val0, valInf):  # f  # TODO: Implement an Integration (Table Integration )

    if f.__code__ == (lambda t, n: t ** n).__code__:
        return lambda n, s: factorial(n) / s ** (n + 1)
    elif f.__code__ == (lambda a, t: math.exp(a, t)).__code__:
        return lambda a, s: 1 / (s - a)

    elif f.__code__ == (lambda w, t: (math.cos(w * t))).__code__:
        return lambda w, s: s / s ** 2 + w ** 2

    elif f.__code__ == (lambda w, t: (math.sin(w * t))).__code__:
        return lambda w, s: w / s ** 2 + w ** 2


def differentiate(f):
    if f.__code__ == (lambda n, s: factorial(n) / s ** (n + 1)).__code__:
        return lambda t, n: t ** n
    elif f.__code__ == (lambda a, s: (1 / (s - a))).__code__:
        # (lambda a, s : 1/(s-a)).__code__:
        return lambda a, t: math.exp(a, t)

    elif f.__code__ == (lambda w, s: s / s ** 2 + w ** 2).__code__:
        return lambda w, t: (math.cos(w * t))

    elif f.__code__ == (lambda w, s: w / s ** 2 + w ** 2).__code__:
        return lambda w, t: (math.sin(w * t))


# def a(s,t) : lambda s,t : math.exp(-s*t) # note: you cannot have anonymous function

# def exp(s,t) : math.exp(-s*t) # #misleading # caller would be easilt confused with the sign of s you would like to pass in input , only via a named function

# limit is over
valInf = 100_000

#TODO: The goal: Run the following:

# Note : erroneous limit: limist must cover all sides of laplace  polynomial
#laplace: lambda f, s, t: f * math.exp(-s, t)  # limit(f) * integrate(math.exp(-s, t), 0, valInf)

#laplace(x ** 2, 1, 3)

S: lambda S: S
_f: lambda _f: _f

Laplace: lambda _f, S, t: _f * math.exp(-S, t)  # limit(f) * integrate(math.exp(-s, t), 0, valInf)

#print("Laplace(x**2, -s+1, t)", Laplace(x ** 2, -s + 1, t))
#print("laplace(x*2, 1, 2)", laplace(x * 2, 1, 2))



def derive(a, t):
    return a(t + 1)


fun1: lambda a, t: a(t + 1)  # """ note: a() takes 1 positional argument but 2 were given"""

print(a(11)) #not a(1,1)
print(a(10))


# print(a(s,0)) #ERROR: s not defined

# variable = t
# if variable == t
#    lambda variable: a

# friendly note : since a & b :
def a(x):
    return x ** 2


b = lambda x: x ** 2

# note: I cannot use `type` to differentiate them , since they're both of the same type


assert type(a) == type(b)

# also types.LambdaTypes doesn't help:

import types

isinstance(a, types.LambdaType)

isinstance(b, types.LambdaType)
# (but) one casn use __name__ (could have been modified)
# Warning: "`is_lambda_function` is not guaranteed to return the Correct result!
a.__name__ = '<lambda>'  # me: warning: this sets name to <lambda> # (i.e. meaningless )

# is_lambda_function(a) # unknown function
# python3 has one type: which is `function
# there are indeed different side effects using def &  a `lambda`:
## def: sets the name to the name ( and qualified name ) of the function
## ans can set
import inspect


def f(x):
    return x ** 2


g = lambda x: x ** 2


def is_lambda_func(f):
    if not inspect.isfunction(f):
        raise TypeError("not a function ")
    src = inspect.getsource(f)
    return not src.startswith("def") and not src.startswith("@")  # provision for decorated funcs


def is_lambda_func(f):
    return f.__code__.co_name == "<lambda>"  # this is a read-only value (contrsry to name )


g.__name__ = 'g'  # set name to 'g'
g.__qualname__ = 'g'  # set qualified name to 'g'

print(f, is_lambda_func(f))
print(g, is_lambda_func(g))

# provided you give it a Qualified name more. a function
# it is unMeaningful : what's difference between the `list` by a list comprehension vs list populated by for loop
# just like difference between string of single quotes, vs string of double quotes
