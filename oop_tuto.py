class Employee:
    'Common class for all the clients'
    #class variable
    empCount = 0

    def __init__(self, name, salary):
        #instance variable
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print "Total Employee %d" % Employee.empCount

    def displayEmployee(self):
        print "Name :", self.name, ", Salary: ", self.salary


#create an instence  of class Employee, insantiation
emp1 = Employee('Artur', 45000)
#access class variable
Employee.empCount

#add attrribute to object
emp1.age
getattr(emp1, 'age')
hasattr(emp1, 'age')



class Parent:
    parentAttrb = 100
    def __init__(self):
        print 'Calling parent constructor'

    def parentMethod(self):
        print 'Calling parent method'

    def setAttr(self, attr):
        Parent.parentAttrb = attr

    def getAttr(self):
        print 'Parent Attribute is ', Parent.parentAttrb

class Child(Parent): #define chile class
    def __init__(self):
        print 'Calling Child Construct'

    def childMethod(self):
        print 'Calling child method'

    def parentMethod(self):
        print 'Childs overrides parents method'


#Overloading Operators , defines adddition operator for this class
class Vector:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def __str__(self):
        return 'Vector (%d, %d)' % (self.a, self.b)

    def __add__(self, other):
        return Vector(self.a + other.a, self.b + other.b)


#hidden veriables
class JustCounter:
    __secretCount = 0

    def count(self):
        self.__secretCount += 1
        print self.__secretCount




#Object Oriented Design (OOD) is useful in many cases and is a fundamental part of Python and most medium to large projects built
# in Python. For those reasons, it must be learned and should often be used by data scientists. However, the additional abstraction
# makes it easy to get intro trouble. Our view is that a non-professional programmer should take care when working with objects.
# 
#The main purpose of OOD should be to bundle together data with methods that use, make available, and, (with caution) modify the data. 
#For example:
class MyModel(object):
    def __init__(self, X, Y,...):
        self.X, self.Y = X, Y
        self.model_has_been_fit = False
        # Some code.

    def fit(self):
        # Some code that uses self.X, self.Y.
        self.coefficients = coefficients
        self.model_has_been_fit = True

    def predict(self):
        # Some code to predict Yhat given self.X
        return Yhat

    def plot(self):
        # Some code.

    def get_data_stats(self):
        # Some code.
        return statistical_summary_of_data
    
#The fit method is an example of a method that uses the data. It should (and does) set the coefficients and an attribute to
# tell you that the model has been fit. A common mishap is to also transform the X, Y data within fit. For example, we could 
# rescale self.X and self.Y. This would be a side effect and should be avoided.
#The plot method makes the data available (as a plot). This is a convenience method since it probably just wraps some matplotlib 
#code. So long as this method has no side effect, it is fine to include. The get_data_stats method is similar.
#There is another issue with MyModel as written: Consider the fact that we have bundled together the training data with the object
# and only through this training data do we set self.coefficients. Only using these coefficients can we predict (using self.predict()).
# What if the user wanted to store the (lightweight) coefficients in a file, and then use them at some point in the future 
# to make predictions on new data? What if a user wanted to use a subset of the training data to fit (in e.g. cross validation). 
# Neither of these could be done! This sort of difficulty can be seen in the statsmodels package, which was meant to be used to
# analyze one non-changing piece of data, rather than predict on multiple new pieces of data.
#
#To avoid this problems, we suggest the following revision where we no longer bundle the main data with the model. This is
# more in line with the scikit-learn philosophy.

class MyImprovedModel(object):
    def __init__(self,...):
        # Some code.  Notice that we don't initialize the model with any data.
        self.model_has_been_fit = False

    def fit(self, X, Y):
        # Some code.
        self.coefficients = coefficients
        self.model_has_been_fit = True

    def predict(self, X, coefficients=None):
        # Some code to predict Yhat given X.
        # If coefficients is None, use self.coefficients.
        return Yhat
