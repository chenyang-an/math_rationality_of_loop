from sympy import symbols, Matrix
import numpy as np
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
import time
import random


def catalan_number(n):
    return comb(2 * n, n, exact=True) // (n + 1)
def w_function(a, b, num_points_interpolated, n, i, j):
    file_path = "/Users/an/Desktop/cm/rationality_of_loop/data/noncrossing_matching.json"
    #file_path = "/Users/anchenyang/Desktop/cm/rationality_of_loop/data/noncrossing_matching.json"

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

    file_path = "/Users/an/Desktop/cm/rationality_of_loop/data/noncrossing_matching.json"
    # file_path = "/Users/anchenyang/Desktop/cm/rationality_of_loop/data/noncrossing_matching.json"
    data = json.load(open(file_path, 'r'))
    nc_matching = data[f'{2}']
    i = 0
    j = 0
    # Define symbols
    a, b, c, d = symbols('a b c d')

    # Create a 2x2 symbolic matrix
    M = Matrix([[d**2, d], [d, d**2]])

    # Compute the inverse
    M_inv = M.inv()


    # Display the inverse
    print(M_inv)
    matrix = [[a ** (len(join_partition(nc_matching[i], nc_matching[j]))) for j in range(len(nc_matching))] for i in
                  range(len(nc_matching))]
    matrix_np = Matrix(matrix)

    matrix_inverse = matrix_np.inv()
    print(matrix_inverse)

    for dim in range(1,10):
        start_time = time.time()
        nc_matching = data[f'{dim}']
        a= symbols('a')
        matrix = [[a ** (len(join_partition(nc_matching[i], nc_matching[j]))) for j in range(len(nc_matching))]
                  for i in
                  range(len(nc_matching))]
        matrix_np = Matrix(matrix)

        matrix_inverse = matrix_np.inv()
        print(f"dimension is {dim}")
        print(matrix_inverse)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")