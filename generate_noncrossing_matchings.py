from copy import deepcopy
import numpy as np
from scipy.special import comb
import time
import json

def catalan_number(n):
    return comb(2 * n, n, exact=True) // (n + 1)

def noncrossing_matching(elements):
    temp = [deepcopy(partitions) for partitions in make_partitions(elements)]
    results = []
    for partitions in temp:
        skip_outer_iteration = False
        for partition in partitions:
            if len(partition) ==1:
                skip_outer_iteration = True
                break
        if skip_outer_iteration:
            continue
        results.append(partitions)
    return results
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

if __name__ == '__main__':
    temp_dict = {}
    counter = 0
    results = [deepcopy(partitions) for partitions in make_partitions([1,2,3,4,5,6,7,8])]


    for partitions in results:
        skip_outer_iteration = False
        for partition in partitions:
            if len(partition) ==1:
                skip_outer_iteration = True
                break
        if skip_outer_iteration:
            continue
        print(partitions)

    results = noncrossing_matching([1,2,3,4,5,6])
    for ele in results:
        print(ele)
    print('length is ', len(results))

    counter = 0
    data_list = {}
    #saved_data_path = "/home/c5an/rationality_of_loop/data/noncrossing_matching.json"
    saved_data_path = "/Users/an/Desktop/cm/rationality_of_loop/data/noncrossing_matching.json"

    for i in range(1,100):
        counter += 1
        print(f"{counter}th----------------------")
        start_time = time.time()

        numbers_list = [j for j in range(1, 2*i + 1)]

        data_list[f'{i}'] = noncrossing_matching(numbers_list)
        #json.dump(data_list, open(saved_data_path,'w'))
        #for item in data_list[f'{i}']:
        #    print(item)
        print()
        print(len(data_list[f'{i}']))
        print(catalan_number(i))
        end_time = time.time()
        duration = end_time - start_time

        print("Time taken: {:.2f} seconds".format(duration))


