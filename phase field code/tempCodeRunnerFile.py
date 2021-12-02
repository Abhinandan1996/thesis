    return -alpha*kappa_hat*(T - T_env)
    # Solving the equation by integration.
    temperature = odeint(f, T_0, time, args=(alpha, kappa_hat, T_env))[:, 0]
    # Return time and model results
    return time, temperature
    # Create a model from the coffee_cup_dependent function and add labels
    model = un.Model(coffee_cup_dependent, labels=["Time (s)", "Temperature (C)"])
    # Create the distributions