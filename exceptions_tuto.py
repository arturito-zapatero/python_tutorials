# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:12:55 2017

@author: aszewczyk
"""

#CatchWhatYouCanHandle
#An exception is an error that happens during execution of a program. When that
#error occurs, Python generate an exception that can be handled, which avoids your
#program to crash.

#If an exception occurs, the rest of the try block will be skipped and the
#except clause will be executed.

(x,y) = (5,0)

def div_x_by_y(x,y):
    z = x/y
    print z

#this will do the same, the difference is that if run form outside the class will
#not be seen??
div_x_by_y(4,0)

#catches only dividing by 0 error and prints your message
try:
    z = x/y
#except ZeroDivisionError:
#    print "divide by zero"

#as above but prints the system message
except ZeroDivisionError as e:
    z = e # representation: "<exceptions.ZeroDivisionError instance at 0x817426c>"
    print z # output: "integer division or modulo by zero"
    
    
# catch *all* exceptions
import sys
try:
   z = x/y
except: 
   e = sys.exc_info()[0]
   print( "<p>Error: %s</p>" % e )
   
   
#if y=0 it will set y to 1 and raise the last exception, otherwise performs try 
#operations and do else
try:
    z = x/y
except:
    #rollback()
    y = 1        #will be set to 1
    print "y is now set to: " + str(1)
    raise
else:
     #commit()
     print "Brawo kur..a!"
finally:
    print "would see this error or not"
    

try:
    z = x/y
finally:
    print "cos tam cos tam"