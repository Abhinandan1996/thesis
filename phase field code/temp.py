import chaospy as cp                       # To create distributions
import numpy as np                         # For the time array
from scipy.integrate import odeint         # To integrate our equation
import uncertainpy as un
from matplotlib import pyplot
from math import *

# Create the coffee cup model function
def coffee_cup(a, b,c):
    # Initial temperature and time array
    time = np.linspace(0, 200, 2)            # Minutes                                   # Celsius

    # # The equation describing the model
    # def f(a, b,c):
    #     return sin(a) + sin(b)**2 + sin(a)*sin(c)
    # # Solving the equation by integration
    # temperature = []
    # f
    # Return time and model output
    return time, np.array([sin(a) + sin(b)**2 + sin(a)*sin(c), sin(a) + sin(b)**2 + sin(a)*sin(c)])


# Create a model from the coffee_cup function and add labels
model = un.Model(run=coffee_cup, labels=["Time (min)", "Temperature (C)"])

# Create the distributions
a_dist = cp.Uniform(0.025, 0.075)
b_dist = cp.Uniform(15, 25)
last = cp.Uniform(20,30)

# Define the parameter dictionary
parameters = {"a": a_dist, "b": b_dist, "c": last}

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model, parameters=parameters)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
# We set the seed to easier be able to reproduce the result
data = UQ.quantify(seed=10)
# un.plotting.PlotUncertainty(data['a'])
# #variance = data["sobol_first_average"].variance
# pyplot.ploy.show()
print(data['coffee_cup'].sobol_first)
un.plotting.PlotUncertainty()