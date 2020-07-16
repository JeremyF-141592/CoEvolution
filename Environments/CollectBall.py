from Environments.pyFastSimEnv.DefaultNav_Env import SimpleNavEnv
import pyfastsim as fs
import numpy as np
import time
from Templates.Environments import Environment, EnvironmentFactory
from Parameters import Configuration
import os


class CollectBall(Environment):
    """
    Observation space is Box(5,) meaning 5 dimensions continuous vector
        1-3 are lasers oriented -45:0/45 degrees
        4-5 are left right bumpers
    (edit the xml configuration file if you want to change the sensors)

    Action space is Box(2,) meaning 2 dimensions continuous vector, corresponding to the speed of the 2 wheels

    x,y are bounded [0, 600]
    """

    def __init__(self, new_ball_probability=0.2, mut_std=5.0):
        self.env = SimpleNavEnv(os.path.dirname(__file__) + "/pyFastSimEnv/LS_maze_hard.xml")
        self.env.reset()

        self.new_prob = new_ball_probability
        self.mut_std = mut_std

        self.init_pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])
        self.ball_held = -1
        self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])

        self.balls = [(self.env.get_robot_pos()[0] + 80, self.env.get_robot_pos()[1] + 80)]

        self.windows_alive = False

        self.proximity_threshold = 2.0  # min distance required to catch or release ball

    def add_balls(self):
        self.env.map.clear_illuminated_switches()
        for x, y in self.balls:
            self.env.map.add_illuminated_switch(fs.IlluminatedSwitch(0, 8, x % 580 + 20, y % 580 + 20, True))

    def catch(self):
        if self.ball_held == -1:
            for i, (x, y) in zip(range(len(self.balls)), self.balls):
                if np.sqrt((self.pos[0] - x)**2 + (self.pos[1] - y)**2) < self.proximity_threshold:
                    self.ball_held = i
                    return 100
        return 0

    def release(self):
        if self.ball_held != -1 and \
           np.sqrt((self.pos[0] - self.init_pos[0])**2, (self.pos[1] - self.init_pos[1])**2) < self.proximity_threshold:
            self.ball_held = -1
            return 100
        return 0

    def __call__(self, agent, render=False, use_state_path=False, max_steps=2000, exceed_reward=0):
        self.add_balls()
        if render and not self.windows_alive:
            self.env.enable_display()
        state = self.env.reset()
        done = False

        fitness = 0
        path = list()
        count = 0
        while not done:
            if render:
                self.env.render()
                time.sleep(0.01)

            action = agent.choose_action(state)
            state, reward, done, info = self.env.step(action)
            self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])
            if use_state_path:
                path.append(state)

            reward = 0  # default reward is distance to goal
            reward += self.catch()
            reward += self.release()

            fitness += reward
            count += 1
            if count > max_steps:
                fitness += exceed_reward
                break
        return Configuration.metric(agent, self, fitness, path)

    def get_child(self):
        new_env = CollectBall(self.new_prob, self.mut_std)
        new_balls = list()
        for b in self.balls:
            new_balls.append((b[0] + np.random.normal(0, self.mut_std),
                              b[1] + np.random.normal(0, self.mut_std)))
        if np.random.uniform(0, 1) < self.new_prob:
            new_balls.append((self.balls[-1][0] + np.random.normal(0, self.mut_std),
                              self.balls[-1][1] + np.random.normal(0, self.mut_std)))
        new_env.balls = new_balls
        return new_env

    def crossover(self, other):
        new_env = CollectBall(self.new_prob, self.mut_std)
        new_balls = list()
        if len(self.balls) >= len(other.balls):
            for i in range(len(other.balls)):
                new_balls.append(self.balls[i] if np.random.uniform(0, 1) < 0.5 else other.balls[i])
            new_balls += self.balls[len(other.balls):].copy()
        else:
            for i in range(len(self.balls)):
                new_balls.append(self.balls[i] if np.random.uniform(0, 1) < 0.5 else other.balls[i])
            new_balls += other.balls[len(other.balls):].copy()
        new_env.balls = new_balls
        return new_env

    def __getstate__(self):
        dic = dict()
        dic["Balls"] = self.balls
        dic["NewProb"] = self.new_prob
        dic["Std"] = self.mut_std
        return dic

    def __setstate__(self, state):
        self.__init__(state["NewProb"], state["Std"])
        self.balls = state["Balls"]

    def __del__(self):
        self.env.close()


class CollectBallFactory(EnvironmentFactory):

    def __init__(self, new_ball_probability=0.2, mut_std=25.0):
        self.new_ball_probability = new_ball_probability
        self.mut_std = mut_std

    def new(self):
        return CollectBall(self.new_ball_probability, self.mut_std)
