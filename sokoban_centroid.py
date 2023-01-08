import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

global SHOW_LEVEL
global SAVE_FIG

SHOW_LEVEL = False
SAVE_FIG = True


def load_level(level_file):
    level_layout = []
    locations_dict = {"Free": [], 'Wall': [], 'Box': [], 'Goal': [], "Agent": None}
    with open(level_file, 'r') as level:
        for x, row in enumerate(level):
            for y, char in enumerate(row):
                if char == "#":
                    locations_dict["Wall"].append((x, y))
                elif char == "B":
                    locations_dict["Box"].append(list((x, y)))
                elif char == "G":
                    locations_dict["Goal"].append(list((x, y)))
                elif char == "$":
                    locations_dict["Box"].append(list((x, y)))
                    locations_dict["Goal"].append(list((x, y)))
                elif char == "A":
                    locations_dict["Agent"] = list((x, y))
            level_layout.append(row.strip())
    return level_layout, locations_dict


def list_sum(a_list):
    total = 0
    for i in a_list:
        if isinstance(i, list):
            total += list_sum(i)
        else:
            total += i
    return total


def manhattan_dist(x, y):
    return abs(x[0] - x[1]) + abs(y[0] - y[1])


class SokobanEnvironment(gym.Env):
    def __init__(self, load_file: str = 'Levels/level_0.txt', max_steps: int = 100):
        self.step_count = 0
        self.obs_dict = {}
        self.max_steps = max_steps
        self.base_level, self.base_locations = load_level(load_file)
        self.level, self.locations = self.base_level.copy(), self.base_locations.copy()
        self.max_states = 0
        self._agent_loc = self.base_locations["Agent"]
        self._box_loc = self.base_locations["Box"]
        self._goal_loc = self.base_locations["Goal"]
        self._wall_loc = self.base_locations["Wall"]

        length = len(self.level)
        width = len(self.level[0])

        self.max_states = int(2 * (len(self._box_loc) + 1) * str(len(self.level)))

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=max(length, width), shape=(2,), dtype=int),
                "boxes": spaces.Box(low=0, high=max(length, width), shape=(len(self._box_loc), 2), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4, start=0)  # up, down, left, right

        # define rewards
        self.step_reward = -0.0001
        self.box_off_reward = -5
        self.box_on_reward = 10
        self.box_move_reward = 0
        self.done_reward = 0

    def _get_obs(self):
        # agent & box locations as an int
        return {'agent': self._agent_loc, 'boxes': self._box_loc}

    def _get_info(self):
        # how many boxes on targets
        return sum([True if self._box_loc[i] == self._goal_loc[i] else False for i in range(len(self._box_loc))])

    def step(self, action, show_level: bool = SHOW_LEVEL):
        """
        Performs a single timestep of the Sokoban env
        :param action: The action the agent will take
        :param show_level: Do you want the level shown? Default value of True
        :return: Five element tuple
        """
        self.level_str_to_list()
        reward = self.step_reward  # initialize reward
        self.step_count += 1
        boxes_before = self.count_boxes()  # count boxes on goals
        if self._move(action):  # if move executes successfully
            reward += self.box_move_reward
            if type(self.level[0]) == list:
                self.level_list_to_str()
            boxes_after = self.count_boxes()  # count boxes in case we moved one onto goal
            # find reward
            if boxes_after > boxes_before:
                reward += self.box_on_reward
            elif boxes_after < boxes_before:
                reward += self.box_off_reward
        did_win = self.check_done()
        if did_win and self.check_boxes():
            reward += self.done_reward
        if show_level:
            self.show(action)
        return self._get_obs(), reward, did_win, False, self._get_info()

    def _move(self, action):
        """
        Moves the agent, and if it collides with a box, moves the box too. Move must not cause agent or box to occupy same space as a wall or another box.
        Also updates visual representation of level, moving agent and box if necessary.
        :param action: The action the agent is taking -> [0, 3]
        :return: A boolean stating if the move was successful or not
        """
        # convert level to list for easy index and updating
        self.level_str_to_list()
        new_position = [sum(x) for x in zip(self._agent_loc, coord_dict[action])]  # find new position for agent
        new_position_check = [sum(x) for x in zip(new_position, coord_dict[action])]  # position 2 steps in direction of action
        box_check = self._box_loc.copy()
        if tuple(new_position) in self._wall_loc:  # make sure we don't clip into a wall
            return False  # move couldn't be made
        elif new_position in self._box_loc:  # check action won't have us clipping into a box
            move_box = self._box_loc.index(new_position)  # the box we will move
            # check that, if this box is moved, it won't clip into a wall, another box, or a box on a goal state
            if tuple(new_position_check) in self._wall_loc or new_position_check in box_check.pop(move_box) or self.level[new_position_check[0]][new_position_check[1]] == "$":
                return False  # move couldn't be made
            else:
                new_box_position = [sum(x) for x in zip(self._box_loc[move_box], coord_dict[action])]  # find new position for cur box
                # level updates for box
                if self.level[new_box_position[0]][new_box_position[1]] == "G":
                    self.level[new_box_position[0]][new_box_position[1]] = "$"
                elif self.level[new_box_position[0]][new_box_position[1]] == ".":
                    self.level[new_box_position[0]][new_box_position[1]] = "B"
                elif self.level[new_box_position[0]][new_box_position[1]] == "B":
                    return False
                closest_goal = np.argmin([manhattan_dist(self._box_loc[move_box], i) if i not in self._box_loc else np.inf for i in self._goal_loc])
                if manhattan_dist(self._box_loc[move_box], self._goal_loc[closest_goal]) > manhattan_dist(new_box_position, self._goal_loc[closest_goal]):
                    self.box_move_reward = 3
                else:
                    self.box_move_reward = -0.5
                # update box position and symbol for old location
                self.level[self._box_loc[move_box][0]][self._box_loc[move_box][1]] = '.'
                self._box_loc[move_box] = [new_box_position[0], new_box_position[1]]
                # update agent position
                if self._agent_loc in self._goal_loc:
                    self.level[self._agent_loc[0]][self._agent_loc[1]] = "G"  # agent moving off of goal
                else:
                    self.level[self._agent_loc[0]][self._agent_loc[1]] = "."  # agent moving
                self._agent_loc = new_position  # update position
                self.level[self._agent_loc[0]][self._agent_loc[1]] = "A"  # update symbol
                return True  # move was successful
        else:  # if where we want to go is an empty tile, go there
            # update agent position
            if self._agent_loc in self._goal_loc:
                self.level[self._agent_loc[0]][self._agent_loc[1]] = "G"  # agent moving off of goal
            else:
                self.level[self._agent_loc[0]][self._agent_loc[1]] = "."  # agent moving
            self._agent_loc = new_position  # update position
            self.level[self._agent_loc[0]][self._agent_loc[1]] = "A"  # update symbol
            return True  # move was successful

    def show(self, action=None):
        """
        Prints the level row by row
        :return: None
        """
        if action is None:
            print(f"The beginning...")
        else:
            print(f"Step Number: {self.step_count} | Action: {action_info[action]}")
        self.level_list_to_str()
        for row in self.level:
            print(row)
        print()

    def level_str_to_list(self):
        """
        Converts all rows of the level to lists for easy value assigning
        :return:
        """
        for i in range(len(self.level)):
            self.level[i] = list(self.level[i])

    def level_list_to_str(self):
        """
        Converts list level to string for easy printing
        :return:
        """
        for row in range(len(self.level)):
            string_row = ''
            for element in self.level[row]:
                string_row += element
            self.level[row] = string_row

    def _get_prob(self, state, action, state_prime):
        """
        Transition function. If the agent and any adjacent box move in the direction specified by "action," the probability is 1. Otherwise, probability is 0.
        :param state: s_t
        :param action: current action
        :param state_prime: s_t+1
        :return:
        """
        cur_agent, cur_boxes, cur_goals = state[0], state[1], state[2]
        new_agent = [sum(x) for x in zip(cur_agent, coord_dict[action])]

        if new_agent in cur_boxes and new_agent == state_prime[0]:
            index = cur_boxes.index(new_agent)
            new_box = [sum(x) for x in zip(cur_boxes[index], coord_dict[action])]
            if state_prime[1][index] == new_box:
                return 1
            else:
                return 0
        elif new_agent == state_prime[0] and cur_boxes == state_prime[1]:  # if the agent doesn't move any boxes and move is legit
            return 1
        else:
            return 0

    def count_boxes(self):
        """
        Counts the number of boxes on goals
        :return: the count of boxes on goal tiles
        """
        return sum(loc in self._box_loc for loc in self._goal_loc)

    def check_boxes(self):
        """
        Check if all boxes are on goal tiles
        :return: Bool stating if boxes are all on goal tiles or not
        """
        return True if sorted(self._box_loc) == sorted(self._goal_loc) else False

    def check_done(self):
        """
        Checks if we won the game or if we've run out of steps
        :return: Bool stating if we end the game or not
        """
        return True if self.check_boxes() or self.step_count == self.max_steps else False

    def reset(self, seed=None, options=None):
        """
        Resets the environment to default values
        """
        super().reset(seed=seed)
        self.level, self.locations = self.base_level.copy(), self.base_locations.copy()
        self._agent_loc = self.base_locations["Agent"]
        self._box_loc = self.base_locations["Box"]
        self._goal_loc = self.base_locations["Goal"]
        self._wall_loc = self.base_locations["Wall"]
        self.step_count = 0
        return self._get_obs(), self._get_info()


class SemiGradientSarsa:
    def __init__(self, env, alpha, eps):
        """
        initialize SARSA
        """
        self.env = env
        self.alpha = alpha
        self.epsilon = eps

    def e_greedy(self, state, weights):
        """
        performs e-greedy action selection based on epsilon
        """
        check = np.random.random()

        if check < self.epsilon:
            max_a = 0
            max_a_v = 0
            for action in range(self.env.action_space.n):
                v = self.q_hat(state, action, weights)
                if v > max_a_v:
                    max_a = action
                    max_a_v = v
            return max_a
        else:
            return np.random.randint(self.env.action_space.n)

    def q_hat(self, state, action, weight):
        """
        Calculate the q_hat vector for SARSA based on the input state, action, and weight
        """
        return np.dot(self.x_s_a(state, action), weight)

    def x_s_a(self, state, action):
        """
        finds the X vector for a specific state -> incorporates centroid of boxes and agent location
        """
        x = [i[0] for i in state['boxes']]  # all x values for centroid calculation
        y = [i[1] for i in state['boxes']]  # all y values for centroid calculation
        return np.array([1, state['agent'][0], state['agent'][1], sum(x) / len(state['boxes']), sum(y) / len(state['boxes'])])  # feature vector

    def train(self, episodes):
        """
        Train the SARSA algorithm
        :param episodes: Number of epochs
        :return: A list containing the average reward of each training epoch
        """
        weights = np.zeros(5)
        total_rewards = []
        print_count = 0
        for i in range(episodes):
            rewards = []
            state, info = self.env.reset()
            action = self.e_greedy(state, weights)
            done = False
            while not done:
                state_prime, reward, done, _, _ = self.env.step(action)
                if done:
                    weights += self.alpha * (reward - self.q_hat(state, action, weights)) * self.x_s_a(state, action)
                else:
                    action_prime = self.e_greedy(state_prime, weights)
                    vector = self.x_s_a(state, action)
                    weights += self.alpha * (reward + 0.9 * self.q_hat(state_prime, action_prime, weights) - self.q_hat(state, action, weights)) * vector
                    state = state_prime
                    action = action_prime
                rewards.append(reward)
            total_rewards.append(sum(rewards) / len(rewards))
            if print_count == int(episodes/10):
                print(f'---- Episode: {i}')
                print_count = 0
            print_count += 1
        return total_rewards, weights

    def load_policy(self, state, weight) -> int:
        """
        Given a state and the weight vector received after training, return the best action by taking argmax of q values for each action
        """
        return self.e_greedy(state, weight)


coord_dict = {
    0: [0, -1],
    1: [0, 1],
    2: [1, 0],
    3: [-1, 0]
}  # dictionary containing change in coords based on action

action_info = {
    0: 'move left',
    1: 'move right',
    2: 'move down',
    3: 'move up',
}  # dictionary describing actions

if __name__ == '__main__':
    sokoban = SokobanEnvironment('Levels/level_1.txt')  # change to level_0.txt to check 6-box map
    sokoban.show()
    a = 0.1
    e = 0.7
    sarsa = SemiGradientSarsa(env=sokoban, alpha=a, eps=e)
    episodes = 100000
    results, policy = sarsa.train(episodes)
    # plotting
    plt.plot(range(1, len(results) + 1), results)
    plt.title(f"SARSA Sokoban | alpha = {a} | Epsilon = {e}")
    plt.xlabel("Episode number")
    plt.ylabel("Average reward")
    if SAVE_FIG:
        plt.savefig(f"pres_centroid_{a}_e{e}_{episodes}.svg")  # save figure
    plt.show()
    print(f'Policy after training: {policy}')
    # load policy and find best action for random state
    test_state = {'agent': [2, 2], 'boxes': [[2, 5]]}
    best_action = sarsa.load_policy(state=test_state, weight=policy)
    print(f"Load policy and find best action for state: {test_state}")
    print(f"The best action on the test state outlined above is: {best_action} ({action_info[best_action]})")
