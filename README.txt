# Machine-Learning-Q-Learning
The full assignment of Q learning 

To run this code: 
Download all files and rum q_learning_main.py

What it does:
Implements the Q_learning algorithm which is an algorithm that explores a graph like in the environment.txt and self learns through the graph to find the best route to the reward. In environment.txt the I is the start any X is an impassable wall 1.0 is the reward and -1.0 is the negative reward, everything else is neutral movement. Using these parameters it will loop the desired number of times noted in the variable number_of_moves until the q_learning has the best possible outcome.

Q-learning:
Here are the implemented steps for the implementation
Step 1:
Attempts move, checks for wall or terminal and returns reward (rprime) and new state (sprime)
Step 2:
Updates the q-learning algorithm Q[s,a] = (1 - c) * Q[s,a] + c * (r + γ * max_a' Q[s',a']) using the decaying learning rate c = 20 / (19 + Nsa[s][a]) and if it is terminal it will set the value
Step 3:
Chooses the next best exploration using the function f(Qsa, Nsa, s, Ne) 
If: state-action visited < Ne: optimistic value = 1 (encourages exploration)
Else: use learned Q-value
Step 4:
It will execute action by simulating moving in a direction using ExecuteAction(a) there is also randomness to improve the training 
80% move in intended direction 
10% go left 
10% go right

After the learning process it will print out the trained graph with the best possible path turning enviornment 2 form
., ., .,  1.0
., X, ., -1.0
I, ., .,    .

to

>      >      >      o     
^      X      ^      o     
^      >      ^      < 

here are the values it generated as well 
 0.511  0.650  0.796  1.000
 0.399  0.000  0.477 -1.000
 0.297  0.256  0.344  0.137
