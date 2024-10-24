from sympy import symbols, solve, Eq



if __name__ == '__main__':

    '''# Define the variable
    p,q,x,y = symbols('p q x y')

    # Define the equation
    equation_1 = Eq((42-7/6)*x + (42-(7/6)*q)*y, 42)
    equation_2 = Eq((42-p)*x + 41*y, 42)


    # Find solutions
    solutions = solve((equation_1,equation_2), (p,q,x,y))

    # Filter integer solutions
    #integer_solutions = [s for s in solutions if s.is_Integer]

    print(solutions)'''

    # This code block assumes you are using a SageMath environment

    # Define the variables
    '''p,q,x,y = var('p q x y')

    # Define the system of equations
    equation1 = (42-7/6)*x + (42-(7/6)*q)*y == 42  # Example: 2x + 3y - 6 = 0
    equation2 = (42-p)*x + 41*y == 42  # Example: x^2 - y = 0

    # Solve the system
    solutions = solve([equation1, equation2], p,q,x,y)

    # Display the solutions
    print("Solutions:", solutions)'''

    # Define the range for x and y
    x_range = range(-10, 11)  # Example range, adjust as needed
    y_range = range(-10, 11)  # Example range, adjust as needed

    # Initialize a list to store integer solutions
    integer_solutions = []

    # Brute-force search
    for x_val in x_range:
        for y_val in y_range:
            if x_val == 0 or y_val == 0:  # Avoid division by zero
                continue
            z_val = (42.0 * x_val + 41.0 * y_val - 42.0) / x_val
            w_val = (35.0 * x_val + 36.0 * y_val - 36.0) / y_val
            if z_val.is_integer() and w_val.is_integer():
                integer_solutions.append((x_val, y_val, z_val, w_val))

    # Display the found integer solutions
    integer_solutions = []

    x_range = range(0, 1000)  # Example range, adjust as needed
    y_range = range(0, 1000)

    for P in x_range:
        for B in y_range:
            try:
                # Equation for M
                M = (42 * B - 36) / ((36 * P - 42) + (42 * B - 36) + (1 - P * B))

                # Corrected equation for N
                N = (36 * P - 42) / ((36 * P - 42) + (42 * B - 36) + (1 - P * B))

                # Equation for T
                T = 42 * (P * B - 1) / ((36 * P - 42) + (42 * B - 36) + (1 - P * B))\

                if M.is_integer() and N.is_integer() and T.is_integer() and M > 0 and N > 0 and T > 0:
                    integer_solutions.append(f"P:{P},Q:{B},M:{M},N:{N},T:{T}")
            except Exception as ex:
                pass

    # Display the found integer solutions
    print(integer_solutions)
