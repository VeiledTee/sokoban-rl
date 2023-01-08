# sokoban-rl

### Files
Running the ```sokoban_box_coords.py``` file will employ a state vector
representing the
```x``` and ```y``` coordinates of the agent, and the ```x``` and ```y``` coordinates
of all boxes in the Sokoban level. For example, if the agent is at location
```[2, 4]``` and two boxes are at ```[[3, 5], [1, 2]]``` respectively, the state 
vector will be ```[1, 2, 4, 3, 5, 1, 2]```. All state vectors begin with ```1``` as
a static digit to ensure a component of the vector does not rely solely  on the state.

Running the ```sokoban_box_centroid.py``` file will employ a state vector
representing the
```x``` and ```y``` coordinates of the agent, and the ```x``` and ```y``` coordinate
of the centroid of all the boxes in each level. If there is only one box, the
centroid will be the location of the single box. For example, if the agent is at location
```[2, 4]``` and two boxes are at ```[[3, 5], [1, 2]]``` respectively, the state 
vector will be ```[1, 2, 4, 2, 3.5]```. This ensures all state vectors have a size 
of ```5``` regardless of the number of boxes in the level.

Each Sokoban environment has the option to print the level (and action
most recently taken) at each iteration of the training process, and save the figure depicting
the average reward for each episode.
The default value for both of these variables is set to ```False``` to allow faster training and prevent 
the saving of unwanted files. Changing the 
global variables at the top of the script will print the level at each training step.

Changing the ```a``` and ```e``` values updates the 
```alpha``` and ```epsilon``` values for SARSA, respectively.

### Levels
All levels are ```.txt``` files and custom ones can be created. The 
directory structure for the project is outlined in the associated report. 
