
import numpy as np
import ast
import math
import random
from sympy.combinatorics.permutations import Permutation
from collections import defaultdict
from queue import PriorityQueue


class Board:
    width = 3
    height = 3
    values = list(range(width * height))
    goal = None

    def __init__(self, width=3,height=3,board=None, goal=None):
        if width is not None:
            self.width=width
        if height is not None:
            self.heigth=height
        self.board = np.zeros((self.width, self.height))
        if board is not None:
            self.board = board
        if goal is not None:
            np.array(ast.literal_eval(goal))
        else:
            self.goal = self.goal()

    # returns the possible states of a board folowing an allowed movement
    def next_states(self):
        x0, y0 = self.find_position(0)
        boards = []
        possible_state = [(x0 - 1, y0), (x0, y0 - 1),
                          (x0, y0 + 1), (x0 + 1, y0)]
        for state in possible_state:
            if self.valid_position(state):
                board = self.board.copy()
                # on permute!
                board[state[0]][state[1]
                                ], board[x0][y0] = board[x0][y0], board[state[0]][state[1]]
                boards.append(self.__class__(board=board))
        return boards

    def can_be_solved(self):
        """
        verify that this can be solved
        """
        permutation_parity = Permutation(self.board.flatten()).signature()
        empty_position = self.find_position(0)
        distance = empty_position[0] + empty_position[1]
        empty_parity = 1 if (distance % 2 == 0) else -1
        return empty_parity == permutation_parity

    def display(self):
        for i in range(self.height):
            for j in range(self.width):
                print(str(self.board[i][j]), end=" | ")
            print()

    # checks if a given position is valid or not in a board : (4,5) is not a valid position in 3*3 board

    def valid_position(self, position):
        return (self.width > position[0] and self.height > position[1]
                and position[0] >= 0 and position[1] >= 0)

    # return the position of a value (x,y) if existed
    def find_position(self, value, goal=False):
        x, y = 0, 0
        # we look linside the goal array if specified otherwise we look in the board array
        if goal:
            x, y = np.where(self.goal == value)
        else:
            x, y = np.where(self.board == value)
        return (x[0], y[0])

    # defines randomly the board
    def random(self):
        choices = random.sample(set(self.values), len(self.values))
        choices = np.array(choices)
        choices = choices.reshape((self.width, self.height))

        res = Board()
        res.board = choices
        return res

    # sets the board to be the goal

    def goal(self):
        goal = np.array(self.values)
        goal = goal.reshape(self.width, self.height)
        return goal

    # somme des place eronnée

    def heuristic1(self, current):
        total = 0
        for case in np.nditer(current.board):
            if self.find_position(case, goal=True) != current.find_position(case):
                total += 1
        return total

    # somme de distance a la place goal

    def heuristic2(self, current):
        total = 0
        for index, case in np.ndenumerate(current.board):
            correct_position = self.find_position(case, goal=True)
            total += abs(index[0] - correct_position[0]) + \
                abs(index[1] - correct_position[1])
        return total

    def __lt__(self, other):
        if (self.__eq__(other) == False):
            return True
        return False

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __str__(self):
        res = ""
        for row in self.board:
            for value in row:
                res += (str(value) if value != 0 else ' ') + " "
            res += "\n"
        return res

    def __repr__(self):
        return "\n" + self.__str__()

    def __hash__(self):
        return hash(str(self.board))

    def is_goal(self):
        return np.array_equal(self.board, self.goal)
# ------------


def reconstract_path(previous_node_map, current):
    total_path = [current]
    while current in previous_node_map.keys():
        current = previous_node_map[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def solve(start,
          heuristic, distance=lambda state1, state2: 1):
    if not start.can_be_solved():
        print("impossible to solve !!")
        return None
    cadidates_queue = PriorityQueue()
    cadidates_queue.put((heuristic(start), start))
    previous_node_map = {}
    # the cost of the cheapest path from start to n currently known
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    # For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess
    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = heuristic(start)

    current = cadidates_queue.get()[1]
    while current:  # len openList not 0

        if current.is_goal():
            #print(g_score[current])  # minimal distance  between start and goal
            return reconstract_path(previous_node_map, current)

        for candidate_state in current.next_states():
            # the score from start to candidate through current
            cadidate_score = g_score[current] + \
                distance(current, candidate_state)
            # This path to candidate is better than any previous one so we choose it
            if cadidate_score < g_score[candidate_state]:
                previous_node_map[candidate_state] = current
                g_score[candidate_state] = cadidate_score
                f_score[candidate_state] = g_score[candidate_state] + \
                    heuristic(candidate_state)
                cadidates_queue.put(
                    (f_score[candidate_state], candidate_state))
        current = cadidates_queue.get()[1]
    return None

#--------------------------------------#
width=3
height=3
board = Board(width=width,height=height)
board = board.random()
print("is this solvable ? 🐉 =",board.can_be_solved())
board.display()
print("👾👾 Heuristic 2 👾👾")

if resulth2 := solve(board, board.heuristic2):
    print("nbr of steps 🤯", len(resulth2))
    print("\n Steps : ", resulth2)

print("👾👾 Heuristic 1 👾👾")
if resulth1 := solve(board, board.heuristic1):
    print("nbr of steps 🤯", len(resulth1))
    print("\n Steps : ", resulth1)

