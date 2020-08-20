from collections import deque
import time
import os
import numpy as np
from Objects.Environments.pyFastSimEnv.DefaultNav_Env import SimpleNavEnv
import pyfastsim as fs
from ABC.Environments import ParameterizedEnvironment, EnvironmentFactory
from Parameters import Configuration


class CollectBall(ParameterizedEnvironment):
    """
    2 Wheeled robot inside a maze, collecting balls and dropping them into a goal.
    The environment is an additional layer to pyFastSim.

    Default observation space is Box(10,) meaning 10 dimensions continuous vector
        1-3 are lasers oriented -45:0/45 degrees
        4-5 are left right bumpers
        6-7 are light sensors with angular ranges of 50 degrees, sensing balls represented as sources of light.
        8-9 are light sensors with angular ranges of 50 degrees, sensing goal also represented as a source of light.
        10 is indicating if a ball is held
    (edit the xml configuration file in ./pyFastSimEnv if you want to change the sensors)

    Action space is Box(3,) meaning 3 dimensions continuous vector, corresponding to the speed of the 2 wheels, plus
    a 'grabbing value', to indicate whether or not the robot should hold or release a ball. (<0 release, >0 hold)

    x,y are by default bounded in [0, 600].

    Environment mutation corresponds to a translation of the balls + translation and rotation of the initial position
    of the robot at the start of one episode.
    """

    def get_weights(self):
        w = list()
        for i in self.init_balls:
            w.append(i[0])
            w.append(i[1])
        w.append(self.init_pos[0])
        w.append(self.init_pos[1])
        w.append(self.init_pos[2])
        return w

    def set_weights(self, weights):
        tups = list()
        for w in range(0, len(weights)-3, 2):
            tups.append((weights[w], weights[w+1]))
        p0 = weights[-3]
        p1 = weights[-2]
        p2 = weights[-1]
        self.init_balls = tups
        self.init_pos = (p0, p1, p2)

        posture = fs.Posture(*self.env.initPos)
        self.env.robot.set_pos(posture)

    def __init__(self, mut_std=5.0, nb_ball=6, ini_pos=(100, 500, 45)):
        self.env = SimpleNavEnv(os.path.dirname(__file__) + "/pyFastSimEnv/LS_maze_hard.xml")
        self.env.reset()

        self.mut_std = mut_std

        self.env.initPos = ini_pos
        posture = fs.Posture(*ini_pos)
        self.env.robot.set_pos(posture)

        for i in range(10):
            if self.check_validity():
                break
            else:
                self.env.initPos = ((self.env.initPos[0] + np.random.uniform(-10, 10)) % 590 + 5,
                                    (self.env.initPos[1] + np.random.uniform(-10, 10)) % 590 + 5,
                                    self.env.initPos[2])
                posture = fs.Posture(*self.env.initPos)
                self.env.robot.set_pos(posture)

        self.init_pos = self.env.get_robot_pos()
        self.ball_held = -1
        self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])

        self.init_balls = [(self.env.get_robot_pos()[0] + 60 * np.cos((2*np.pi) * i/nb_ball),
                            self.env.get_robot_pos()[1] + 60 * np.sin((2*np.pi) * i/nb_ball)) for i in range(nb_ball)]
        self.balls = self.init_balls.copy()

        self.windows_alive = False

        self.proximity_threshold = 10.0  # min distance required to catch or release ball

    def check_validity(self):
        """
        Take a step in a few directions to see if the robot is stuck or not.
        """
        self.env.reset()
        init_state = self.env.get_robot_pos()

        for i in range(4):
            state, reward, done, info = self.env.step((1, 1))

            if abs(info["robot_pos"][0] - init_state[0]) > 0.2 or abs(info["robot_pos"][1] - init_state[1]) > 0.2:
                posture = fs.Posture(*self.env.initPos)
                self.env.robot.set_pos(posture)
                return True
            new_pos = (self.env.initPos[0], self.env.initPos[1], i*90.0)
            posture = fs.Posture(*new_pos)
            self.env.robot.set_pos(posture)
        return False

    def add_balls(self):
        self.env.map.clear_illuminated_switches()
        self.env.map.add_illuminated_switch(fs.IlluminatedSwitch(1, 8, self.init_pos[0], self.init_pos[1], True))
        for x, y in self.balls:
            self.env.map.add_illuminated_switch(fs.IlluminatedSwitch(0, 8, x, y, True))

    def catch(self):
        if self.ball_held == -1:
            for i, (x, y) in zip(range(len(self.balls)), self.balls):
                if np.sqrt((self.pos[0] - x)**2 + (self.pos[1] - y)**2) < self.proximity_threshold:
                    self.ball_held = i
                    self.balls.remove(self.balls[i])
                    self.add_balls()
                    return 0.1
        return 0.0

    def release(self):
        if self.ball_held != -1:
            self.ball_held = -1
            if np.sqrt((self.pos[0] - self.init_pos[0])**2 + (self.pos[1] - self.init_pos[1])**2) \
               < self.proximity_threshold:
                return 1.0
        return 0.0

    def __call__(self, agent, render=False, use_state_path=False, max_steps=12000, exceed_reward=0):
        self.balls = self.init_balls.copy()
        self.add_balls()
        if render and not self.windows_alive:
            self.env.enable_display()
        state = self.env.reset()
        state.append(0.0)
        if len(agent.choose_action(state)) != 3:
            return AssertionError("The current agent returned an action of length != 3. Aborting.")
        done = False

        fitness = 0.0

        is_stuck = deque()

        path = list()
        count = 0
        while not done:
            if render:
                self.env.render()
                time.sleep(0.01)
            action = agent.choose_action(state)
            holding = action[2] > 0

            state, reward, done, info = self.env.step((action[0]*2.0, action[1]*2.0))
            state.append(1.0 if self.ball_held != -1 else 0.0)

            self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])

            reward = 0.0  # default reward is distance to goal

            if holding:
                reward += self.catch()

            if not holding:
                reward += self.release()

            if count % 50 == 0 and count > 0:
                if np.array(is_stuck).std() < 1:
                    break

            if count % 50 == 0 and count > 0:
                path.append(self.pos[0])
                path.append(self.pos[1])

            if len(is_stuck) == 100:
                is_stuck.popleft()
            is_stuck.append(state)

            fitness += reward
            count += 1
            if count > max_steps:
                fitness += exceed_reward
                break
        for i in range(20 - len(path)):
            path.append(-1)

        return Configuration.metric(agent, self, fitness, path)

    def get_child(self):
        new_init_pos = ((self.init_pos[0] + np.random.normal(0, self.mut_std)) % 560 + 20,
                        (self.init_pos[1] + np.random.normal(0, self.mut_std)) % 560 + 20,
                        (self.init_pos[2] + np.random.normal(0, self.mut_std)) % 360)
        new_env = CollectBall(self.mut_std, ini_pos=new_init_pos)
        new_balls = list()
        for b in self.init_balls:
            # We try to avoid getting too close to the border
            new_balls.append(((b[0] + np.random.normal(0, self.mut_std)) % 560 + 20,
                              (b[1] + np.random.normal(0, self.mut_std)) % 560 + 20))
        new_env.init_balls = new_balls
        return new_env

    def crossover(self, other):
        new_init_pos = self.init_pos if np.random.uniform(0, 1) < 0.5 else other.init_pos
        new_env = CollectBall(self.mut_std, ini_pos=new_init_pos)
        new_balls = list()
        if len(self.init_balls) >= len(other.init_balls):
            for i in range(len(other.init_balls)):
                new_balls.append(self.init_balls[i] if np.random.uniform(0, 1) < 0.5 else other.init_balls[i])
            new_balls += self.init_balls[len(other.init_balls):].copy()
        else:
            for i in range(len(self.init_balls)):
                new_balls.append(self.init_balls[i] if np.random.uniform(0, 1) < 0.5 else other.init_balls[i])
            new_balls += other.init_balls[len(other.init_balls):].copy()
        new_env.init_balls = new_balls
        return new_env

    def __getstate__(self):
        dic = dict()
        dic["Balls"] = self.init_balls
        dic["Std"] = self.mut_std
        dic["Init_pos"] = self.init_pos
        return dic

    def __setstate__(self, state):
        self.__init__(state["Std"], ini_pos=state["Init_pos"])
        self.init_balls = state["Balls"]

    def __del__(self):
        self.env.close()

    def a_star_complexity(self):
        total = 0
        for b in self.init_balls:
            goal = np.array((b[0], b[1]), dtype=int)
            g = np.array([self.init_pos[0], self.init_pos[1]], dtype=int)
            path = self.A_star(g, goal, CollectBall.distance_h)
            if path == -1:
                return -1
            total += len(path)
        return total

    def a_star_render(self, pos=(40, 200)):
        b = np.array(pos, dtype=int)
        g = np.array([self.init_pos[0], self.init_pos[1]], dtype=int)
        p = self.A_star(g, b, CollectBall.distance_h)
        self.env.enable_display()

        for i in range(len(p)):
            new_pos = (p[i][0], p[i][1], 90.0)
            posture = fs.Posture(*new_pos)
            self.env.robot.set_pos(posture)
            self.env.render()
            time.sleep(0.01)
        return p

    def A_star(self, start, end, h):
        open_set = [(start[0], start[1])]

        came_from = dict()
        g_score = np.ones((600, 600)) * 1e6
        g_score[start[0], start[1]] = 0

        f_score = np.ones((600, 600)) * 1e6
        f_score[start[0], start[1]] = h(start, end)

        while len(open_set) > 0:
            current = None
            best = float("inf")
            for k in open_set:
                if f_score[k[0], k[1]] < best:
                    best = f_score[k]
                    current = k

            if abs(current[0] - end[0]) <= 2 and abs(current[1]-end[1]) <= 2:
                return CollectBall.reconstruct_path(came_from, current)

            open_set.remove(current)
            neighbors = [current + np.array([0, 2]),
                         current + np.array([0, -2]),
                         current + np.array([2, 0]),
                         current + np.array([-2, 0])]
            for n in neighbors:
                if n[0] >= 600 or n[0] < 0 or n[1] >= 600 or n[1] < 0 or \
                        self.env.map.get_real(n[0], n[1]) == self.env.map.status_t.obstacle:
                    continue
                new_g_score = g_score[current[0], current[1]] + 1
                if new_g_score < g_score[n[0], n[1]]:
                    came_from[(n[0], n[1])] = current
                    g_score[n[0], n[1]] = new_g_score
                    f_score[n[0], n[1]] = g_score[n[0], n[1]] + h(n, end)
                    if (n[0], n[1]) not in open_set:
                        open_set.append((n[0], n[1]))
        return -1

    @staticmethod
    def reconstruct_path(came_from, current):
        total_path = [current]
        while (current[0], current[1]) in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    @staticmethod
    def distance_h(a, b):
        return np.linalg.norm(a-b)


class CollectBallFactory(EnvironmentFactory):

    def __init__(self, mut_std=35.0, ini_pos=(80, 480, 45), nb_balls=8):
        self.mut_std = mut_std
        self.ini_pos = ini_pos
        self.nb_balls = nb_balls

    def new(self):
        return CollectBall(mut_std=self.mut_std, ini_pos=self.ini_pos, nb_ball=self.nb_balls)
