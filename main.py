from astParser import *

class main:
    
    if __name__ == 'main':
        

        #_string  = input() # prompts user for input

        print("please enter a lambda:\ne.g. lambda x : x ** 2 ")
            
        _string = input("welcome to Laplace, please Input a lambda:\ne.g. lambda x : x **2\n")

        parsedBody = parseLambda(_string)
         
        
        leftBody, op, rightBody =  deParseLambda(parsedBody) # parsedBodyLeftValueSliceValue, operation, parsedBodyRightValue
        op = switchOp(op)
        

        print("\nleft = ", leftBody , "\nop = ", op, "\nright = ",rightBody)
        
        parsedBody = astParser.parseLambda(_string)
        parsedBody = ast.dump(parsedBody)
        
        print("parsedBody = ", parsedBody)

        print("Welcome to Laplace \n")
        print("please insert a lambda\n")
        print("e.g. lambda x : x ** 2 \n")
        input()
