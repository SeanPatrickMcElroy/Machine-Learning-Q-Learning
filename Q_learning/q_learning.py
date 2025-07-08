import numpy as np
import os
import random

def AgentModel_Q_Learning(environment_file, ntr, gamma, number_of_moves, Ne):
    file_data,i_locations,terminal_states = read_data(environment_file)
    zero_array = np.zeros_like(file_data,dtype=float)
    #Each array is set up as follows Qsa[y_axis][x_axis][vector]
    #Qsa is state = [y_axis][x_axis] and action = vector
    #The vector will be [up,right,down,left]
    Qsa = np.repeat(zero_array[:, :, np.newaxis], 4, axis=2)
    Nsa = np.repeat(zero_array[:, :, np.newaxis], 4, axis=2)
    moves = 0

    while moves <= number_of_moves:
        s = None
        r = None
        a = None
        aa = None
        sprime = random.choice(i_locations).copy()
        while moves <= number_of_moves:
            #Checks if curr state is a terminal and if true stops loop
            curr_terminal = False
            #Checks the state in the file data for a reward if there is none use ntr
            sprime, rprime = SenseStateAndReward(sprime,ntr,file_data,a,aa)
            Function_Q_Learning_Update(s,r,a,sprime,rprime,gamma,Qsa,Nsa)
            for state in terminal_states:
                if state == sprime:
                    curr_terminal = True
            if curr_terminal:
                break
            actions_values = f(Qsa, Nsa, sprime, Ne)
            max_value = max(actions_values)
            max_indices = [i for i, value in enumerate(actions_values) if value == max_value]
            a = random.choice(max_indices)
            a,aa = ExecuteAction(a)
            s = sprime.copy()
            r = rprime
            moves = moves + 1
    print_data(file_data,Qsa)

def SenseStateAndReward(sprime,ntr,file_data,a,aa):
    swall = sprime.copy()
    if a == None:
        return sprime,ntr
    if aa == 0:sprime[0] = sprime[0] - 1
    if aa == 1:sprime[1] = sprime[1] + 1
    if aa == 2:sprime[0] = sprime[0] + 1
    if aa == 3:sprime[1] = sprime[1] - 1
    if check_valid_move(aa,file_data,sprime) == None:
        sprime = swall
        return sprime,ntr
    if file_data[sprime[0]][sprime[1]] == 'X':
        sprime = swall
        return sprime,ntr
    if file_data[sprime[0]][sprime[1]] == '1.0':
        return sprime,1
    if file_data[sprime[0]][sprime[1]] == '-1.0':
        return sprime,-1
    return sprime,ntr

def Function_Q_Learning_Update(s,r,a,sprime,rprime,gamma,Qsa,Nsa):
    if rprime == 1.0 or rprime == -1.0:
        Qsa[sprime[0]][sprime[1]] = rprime
    if s == None:
        return
    Nsa[s[0]][s[1]][a] += 1
    c = 20/(19 + Nsa[s[0]][s[1]][a])
    Qsa[s[0]][s[1]][a] = (1 - c) * Qsa[s[0]][s[1]][a] + c * (r + gamma * max(Qsa[sprime[0]][sprime[1]]))

def f(Qsa,Nsa,sprime,Ne):
    actions = [0,0,0,0]
    i = 0
    for _ in actions:
        if Nsa[sprime[0]][sprime[1]][i] < Ne:
            actions[i] = 1
        else:
            actions[i] = Qsa[sprime[0]][sprime[1]][i]
        i = i + 1
    return actions

def ExecuteAction(a):
    l = 0
    r = 0
    if a == 0:l,r = 3,1
    if a == 1:l,r = 0,2
    if a == 2:l,r = 1,3
    if a == 3:l,r = 2,0
    prob_of_correct_action = .8
    prob_of_going_left = .1
    prob_of_going_right = .1
    action = random.uniform(0,1)
    if action <= prob_of_correct_action:
        return a,a
    if action > prob_of_correct_action and action <=prob_of_correct_action + prob_of_going_left:
        return a,l
    if action > prob_of_going_right + prob_of_correct_action:
        return a,r
    
def check_valid_move(a,file_data,sprime):
    if a == 0 and sprime[0] >= 0: return a
    if a == 1 and sprime[1] < len(file_data[0]): return a
    if a == 2 and sprime[0] < len(file_data): return a
    if a == 3 and sprime[1] >= 0: return a
    return None

def read_data(pathname):
    if not os.path.isfile(pathname):
        print(f"read_data: {pathname} not found")
        return None

    # Open and read file lines
    try:
        with open(pathname, 'r') as in_file:
            file_lines = in_file.readlines()
    except Exception as e:
        print(f"read_data: Error reading {pathname} - {e}")
        return None

    answer_array = []
    i_location = []
    terminal_states = []
    #Tracks x and y for where I's are
    y = 0
    for line in file_lines:
        line = line.strip()
        line = line.split(',')
        answer_line = []
        x = 0
        for value in line:
            value = value.strip()
            if value == 'I':
                i_location.append([y,x])
            if value == '1.0':
                terminal_states.append([y,x])
            if value == '-1.0':
                terminal_states.append([y,x])
            x = x + 1
            answer_line.append(value)
        y = y + 1
        answer_array.append(answer_line)
    return answer_array,i_location,terminal_states

def print_data(file_data,Qsa):
    row = 0
    utilitys = []
    directions = []
    for utility in Qsa:
        temp_util_arr = []
        temp_dir_arr = []
        for action in utility:
            chosen_action = max(action) 
            chosen_direction = np.argmax(action)
            temp_util_arr.append(chosen_action)
            temp_dir_arr.append(chosen_direction)
        utilitys.append(temp_util_arr)
        directions.append(temp_dir_arr)
    directions_array_final = []
    row = 0
    for arr in directions:
        col = 0
        directions_array = []
        for dir in arr:
            if file_data[row][col] == '1.0' or file_data[row][col] == '-1.0':
                directions_array.append('o')
                col = col + 1
                continue
            if file_data[row][col] == 'X':
                directions_array.append('X')
                col = col + 1
                continue
            if dir == 0:directions_array.append('^')
            if dir == 1:directions_array.append('>')
            if dir == 2:directions_array.append('v')
            if dir == 3:directions_array.append('<')
            col = col + 1
        directions_array_final.append(directions_array)
        row = row + 1
    for row in utilitys:
        print(" ".join(f"{value:6.3f}" for value in row))
    for row in directions_array_final:
        print(" ".join(f"{element:6s}" for element in row))

