import scipy
import json
import numpy
import copy
import numpy as np
from sympy import symbols, Matrix
from fractions import Fraction
from scipy.special import comb
import sys
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Use the TkAgg backend, replace with an appropriate backend for your system
import matplotlib.pyplot as plt
def make_partitions(elements):
    yield from _make_partitions(sorted(elements, reverse=True), [], [])
def _make_partitions(elements, active_partitions, inactive_partitions):
    if not elements:
        yield active_partitions + inactive_partitions
        return

    elem = elements.pop()

    # Make create a new partition
    active_partitions.append([elem])
    yield from _make_partitions(elements, active_partitions, inactive_partitions)
    active_partitions.pop()

    # Add element to each existing partition in turn
    size = len(active_partitions)
    for part in active_partitions[::-1]:
        if len(part) < 2:
            part.append(elem)
            yield from _make_partitions(elements, active_partitions, inactive_partitions)
            part.pop()

        # Remove partition that would create a cross if new elements were added
        inactive_partitions.append(active_partitions.pop())

    # Add back removed partitions
    for _ in range(size):
        active_partitions.append(inactive_partitions.pop())

    elements.append(elem)
def catalan_number(n):
    return comb(2 * n, n, exact=True) // (n + 1)
def w_function(a, b, num_points_interpolated, n, i, j):
    file_path = "/Users/an/Desktop/cm/rationality_of_loop/data/noncrossing_matching.json"
    data = json.load(open(file_path,'r'))
    nc_matching = data[f'{n}']
    list_x = np.linspace(a, b, num_points_interpolated)
    list_y = []
    for x in list_x:
        matrix = [[x ** (len(join_partition(nc_matching[i], nc_matching[j]))) for j in range(len(nc_matching))] for i in
                  range(len(nc_matching))]
        matrix_np = np.array(matrix)
        matrix_inverse = np.linalg.inv(matrix_np)
        list_y.append(matrix_inverse[i, j])
    return list_y

def rational_function(a, b, num_points_interpolated, P, Q):
    list_x = np.linspace(a, b, num_points_interpolated)
    list_y = []
    for x in list_x:
        numerator = P(x)
        denominator = Q(x)
        # Handling division by zero
        if denominator == 0:
            return np.nan  # or np.inf, depending on your requirements
        list_y.append(numerator / denominator)
    return list_y

def intersect_trivially(set1, set2):
    return set1.intersection(set2) == set()



def join_partition(par1, par2):
    par1_set = []
    par2_set = []
    for block in par1:
        par1_set.append(set(block))
    for block in par2:
        par2_set.append(set(block))

    for block_par2 in par2_set:
        first_true_indicator = False

        par1_set_temp = copy.deepcopy(par1_set)
        for block_par1 in par1_set:
            if intersect_trivially(block_par1, block_par2) is False:
                if first_true_indicator is False:
                    first_true_indicator = True
                    temp_block_par1 = block_par1
                else:
                    temp_block_par1 = temp_block_par1.union(block_par1)

                par1_set_temp.remove(block_par1)

        par1_set_temp.append(temp_block_par1)

        par1_set = copy.deepcopy(par1_set_temp)

    par1_set = list(par1_set)
    return par1_set


if __name__ == '__main__':

    for partitions in make_partitions([1,2,3,4,5,6]):
        skip_outer_iteration = False
        for partition in partitions:
            if len(partition) == 1:
                skip_outer_iteration = True
                break
        if skip_outer_iteration:
            continue
        print(partitions)
    sys.exit()

    a = 10
    b = -10
    num_points_interpolated = 200
    dim_of_nc_matching = 3

    x = np.linspace(a, b, num_points_interpolated)  # 400 points between -10 and 10

    P_s12s = np.poly1d([3, 0, 3])
    Q_s12s = np.poly1d([8, 0, -2])

    P_s12ss34s = np.poly1d([97,0, 82, 0, -107, 0, -792])
    Q_s12ss34s = np.poly1d([512, 0, -1408, 0, 608, 0, -72])

    y = rational_function(a,b,num_points_interpolated,P_s12s,Q_s12s)
    list_y_w_function = w_function(a, b, num_points_interpolated, dim_of_nc_matching ,0, 0)
    list_y_rational_function = rational_function(a,b,num_points_interpolated,P_s12s,Q_s12s)
    close = np.allclose(list_y_w_function, list_y_rational_function, rtol=1e-05, atol=1)

    #print(f"if two list are close: {close}")

    counter = 0
    for dim in range(1,8):
        for i in range(0,catalan_number(dim)):
            for j in range(0,catalan_number(dim)):
                list_y_w_function = w_function(a, b, num_points_interpolated, dim, i, j)
                list_y_rational_function = rational_function(a, b, num_points_interpolated, P_s12s, Q_s12s)
                close = np.allclose(list_y_w_function, list_y_rational_function, rtol=1e-05, atol=1)
                print(f"x-axis:\n{np.linspace(a, b, num_points_interpolated).tolist()}")
                print(f"w_function generated:\n{list_y_w_function}")
                print(f"rational function generated:\n{list_y_rational_function}")

                x = np.linspace(a, b, num_points_interpolated)
                #plt.figure()
                plt.plot(x, list_y_w_function)
                plt.plot(x, list_y_rational_function)
                plt.ylim(-45, 45)
                #plt.savefig(f"./data/Graph/s12ss34s/plot_dim_{dim}_cat_dim_{catalan_number(dim)}_i_{i}_j_{j}_s12ss34s.png")
                #plt.close()
                plt.show()
                counter += 1
                print(f"for dimension {dim}, matrix index {i}, {j}:-------------")
                print(f"Catalan number is {catalan_number(dim)}")
                if close:
                    print(f"Amazing,{i},{j}, dimension is {dim_of_nc_matching}")
                else:
                    print('Not close under current error tolerance')


