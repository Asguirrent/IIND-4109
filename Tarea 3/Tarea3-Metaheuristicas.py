#%%
import numpy as np
from collections import deque

def compute_makespan(permutation, processing_times):
    # Number of jobs
    n = len(permutation)
    # Number of machines
    m = len(processing_times[0])
    # makespan matrix
    completion = np.zeros((n, m), dtype=int)
    
    # First job
    completion[0, 0] = processing_times[permutation[0]][0]
    for s in range(1, m):
        completion[0, s] = completion[0, s-1] + processing_times[permutation[0]][s]
    
    # Remaining jobs
    for k in range(1, n):
        completion[k, 0] = completion[k-1, 0] + processing_times[permutation[k]][0]
        for s in range(1, m):
            completion[k, s] = max(completion[k-1, s], completion[k, s-1]) + processing_times[permutation[k]][s]
    
    return completion[-1, -1]

def constructive(processing_times):
    # Number of jobs
    n = len(processing_times)
    # m = len(processing_times[0])
    total = [sum(job) for job in processing_times]
    # Sort jobs by total processing time (from largest to smallest)
    sorted_jobs = sorted(range(n), key=lambda x: -total[x])
    
    current_perm = [sorted_jobs[0]]
    for i in range(1, n):
        job = sorted_jobs[i]
        best_cmax = float('inf')
        best_pos = 0
        # Try inserting the job in all possible positions
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            cmax = compute_makespan(new_perm, processing_times)
            # Update best position
            if cmax < best_cmax:
                best_cmax = cmax
                best_pos = pos
        # Insert job in best position
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
    return current_perm

def generate_insertion_neighbors(current_solution):
    neighbors = []
    n = len(current_solution)
    # i is the current position of the job to move
    for i in range(n):
        # j is the new position of the job
        for j in range(n):
            if i != j:
                # Copy current solution for not afecting future iterations
                new_sol = current_solution.copy()
                job = new_sol.pop(i)
                new_sol.insert(j, job)
                neighbors.append((new_sol, ('insert', job, i)))
    return neighbors

def generate_swap11_neighbors(current_solution):
    neighbors = []
    n = len(current_solution)
    # n-1 posible swaps
    for i in range(n - 1):
        new_sol = current_solution.copy()
        new_sol[i], new_sol[i+1] = new_sol[i+1], new_sol[i]
        neighbors.append((new_sol, ('swap11', i, i+1)))
    return neighbors

def generate_swap_non_adjacent_neighbors(current_solution):
    neighbors = []
    n = len(current_solution)
    for i in range(n):
        for j in range(i + 2, n): 
            new_sol = current_solution.copy()
            new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
            neighbors.append((new_sol, ('swap_non_adj', i, j)))
    return neighbors
# print(generate_swap_non_adjacent_neighbors([2, 16, 8, 7, 14, 13, 10, 15, 12, 18, 5, 3, 4, 17, 0, 1, 9, 6, 19, 11]))
def generate_kshift_neighbors(current_solution, k):
    neighbors = []
    n = len(current_solution)
    for i in range(n):
        for shift in range(1, k+1):
            if i + shift < n:
                new_sol = current_solution.copy()
                job = new_sol.pop(i)
                new_sol.insert(i + shift, job)
                neighbors.append((new_sol, ('shift', job, i, i + shift)))
            if i - shift >= 0:
                new_sol = current_solution.copy()
                job = new_sol.pop(i)
                new_sol.insert(i - shift, job)
                neighbors.append((new_sol, ('shift', job, i, i - shift)))
    return neighbors
#%%
def tabu_search(processing_times, tenure, max_iter):
    current_solution = constructive(processing_times)
    best_solution = current_solution.copy()
    best_cmax = compute_makespan(best_solution, processing_times)
    
    tabu_list = deque()
    
    for iter_num in range(max_iter):
        best_move = None
        best_move_cmax = float('inf')
        all_neighbors = []
        
        # progress
        progress = (iter_num + 1) / max_iter
        if progress <= 0.7:
            all_neighbors += generate_insertion_neighbors(current_solution)
        else:
            # for the last 30% of iterations, replace insertion with k-shift
            all_neighbors += generate_kshift_neighbors(current_solution, 5)
        # generate all neighbors
        # all_neighbors += generate_insertion_neighbors(current_solution)
        all_neighbors += generate_swap11_neighbors(current_solution)
        all_neighbors += generate_swap_non_adjacent_neighbors(current_solution)
        all_neighbors += generate_kshift_neighbors(current_solution, 2)
        
        # notice that we are not generating the same neighbor twice
        # also we are not generating the reverse of a move (e.g. swap(1, 2) and swap(2, 1))
        # this is because we are using a set to store the tabu moves, and each neighborhood is a tuple
        # the second element of the tuple is the move type, and the rest are the parameters of the move
        # so we can compare the sets of parameters to check if a move is tabu
        for neighbor, move_info in all_neighbors:
            # check if move is tabu
            is_tabu = any(
                # same move
                (move_info == tabu_move) or 
                # simeetric move
                (move_info[0] == 'swap' and set(move_info[1:]) == set(tabu_move[1:]))
                for tabu_move in tabu_list
            )
            
            if is_tabu:
                # print(is_tabu)
                # aspiration criterion
                cmax = compute_makespan(neighbor, processing_times)
                # print(cmax)
                if cmax >= best_cmax:
                    continue 
            
            cmax = compute_makespan(neighbor, processing_times)
            # selecting the best neighbor
            if cmax < best_move_cmax:
                best_move = neighbor
                best_move_cmax = cmax
                best_move_info = move_info
                # print(best_move_cmax,best_move)
        
        if best_move is None:
            continue

        # actualize tabu list
        current_solution = best_move.copy()
        tabu_list.append(best_move_info)
        # print(current_solution)
        # actualize tabu list
        if len(tabu_list) > tenure:
            tabu_list.popleft()
        # actualize best solution
        # print(best_move_cmax,best_cmax)
        if best_move_cmax <= best_cmax:
            best_cmax = best_move_cmax
            best_solution = current_solution.copy()
            # print(1,best_solution)
    
    return best_solution, best_cmax

# %%
def lcg(seed, n):
    m = 2**31 - 1   
    a = 16807
    b = 127773
    c = 2836
    results = []
    
    for _ in range(n):
        k = seed // b
        seed = a * (seed % b) - c * k
        if seed < 0:
            seed += m
        
        results.append(1 + (seed * 99 // m)) 
    
    return results

# print(lcg(896678084        , 500))
def convert_data(results, n, m):
    matriz = [[0] * n for _ in range(m)] 

    index = 0
    for col in range(n):  
        for row in range(m):  
            matriz[row][col] = results[index]
            index += 1
    
    return matriz


# %%
import time
tenure=41
max_iter=211
#Seeds
# 20x5
seed_20x5=[873654221,379008056,1866992158,216771124,
           495070989,402959317, 1369363414,2021925980,
           573109518,88325120]

# 20x10
seed_20x10=[587595453,1401007982,873136276,268827376,
             1634173168,691823909,73807235,1273398721,
             2065119309,1672900551]

# 20x20
seed_20x20=[479340445, 268827376,1958948863,918272953,
            555010963,2010851491,1519833303,1748670931,
            1923497586,1829909967]

# 50x5
seed_50x5=[1328042058,200382020,496319842,1203030903,
            1730708564,450926852,1303135678,1273398721,
            587288402,248421594]

# 50x10
seed_50x10=[1958948863,575633267,655816003,1977864101,
            93805469,1803345551, 49612559,1899802599,
            2013025619,578962478]

# 50x20
seed_50x20=[1539989115,691823909,655816003,1315102446,
             1949668355,1923497586,1805594913,1861070898,
             715643788,464843328]
############
# 100x5
seed_100x5=[896678084,1179439976,1122278347,416756875,
            267829958,1835213917,1328833962,1418570761,
            161033112,304212574]

# 100x10
seed_100x10=[1539989115,655816003,960914243,1915696806,
            2013025619,1168140026,1923497586,167698528,
            1528387973,993794175]

# 100x20
seed_100x20=[450926852,1462772409,1021685265,83696007,
            508154254,1861070898,26482542,444956424,
            2115448041,118254244]

# 200x10
seed_200x10=[471503978,1215892992,135346136,1602504050,
            160037322,551454346,519485142,383947510,
            1968171878,540872513]

# 200x20
seed_200x20=[2013025619,475051709,914834335,810642687,
            1019331795,2056065863,1342855162,1325809384,
            1988803007,765656702]

# 500x20
seed_500x20=[1368624604,450181436,1927888393,1759567256,
            606425239,19268348,1298201670,2041736264,
            379756761,28837162]

for seed in seed_20x5:
    start_time = time.time()
    best_perm, cmax = tabu_search(convert_data(lcg(seed, 100), 5, 20), tenure, max_iter)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Cmax: {cmax}, Time: {elapsed_time}")
#%%
# Ta=0
# for seed in seed_20x5:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 100), 5, 20), tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_20x10:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 200), 10, 20),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_20x20:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 400), 20, 20),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_50x5:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 250), 5, 50),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_50x10:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 500), 10, 50),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_50x20:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 1000), 20, 50),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_100x5:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 500), 5, 100),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_100x10:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 1000), 10, 100),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_100x20:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 2000), 20, 100),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_200x10:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 2000), 10, 200),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_200x20:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 4000), 20, 200),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")
# for seed in seed_500x20:
#     start_time = time.time()
#     best_perm, cmax = tabu_search(convert_data(lcg(seed, 10000), 20, 500),tenure, max_iter)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     Ta+=1
#     print(f"Taillard {Ta}: Cmax: {cmax}, Time: {elapsed_time}")