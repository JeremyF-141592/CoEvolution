from Environments.pyFastSimEnv.DefaultNav_Env import SimpleNavEnv
import pyfastsim as fs
import numpy as np
import time
from Templates.Environments import Environment, EnvironmentFactory
from Parameters import Configuration
import os


class CollectBall(Environment):
    """
    2 Wheeled robot inside a maze, collecting balls and dropping them into a goal.
    The environment is an additional layer to pyFastSim.

    Default observation space is Box(10,) meaning 10 dimensions continuous vector
        1-3 are lasers oriented -45:0/45 degrees
        4-5 are left right bumpers
        6-7 is a light sensor with an angular range of 50 degrees, sensing balls represented as sources of light.
        8-9 is a light sensor with an angular range of 50 degrees, sensing goal also represented as a source of light.
        10 is the grabbing value
    (edit the xml configuration file if you want to change the sensors)

    Action space is Box(3,) meaning 3 dimensions continuous vector, corresponding to the speed of the 2 wheels, plus
    a 'grabbing value', to indicate whether or not the robot should hold or release a ball. (<0 release, >0 hold)

    x,y are by default bounded in [0, 600].

    Environment mutation corresponds to a translation of the balls + translation and rotation of the initial position
    of the robot at the start of one episode.
    """

    def __init__(self, mut_std=5.0, nb_ball=6, ini_pos=(100, 500, 45)):
        self.env = SimpleNavEnv(os.path.dirname(__file__) + "/pyFastSimEnv/LS_maze_hard.xml")
        self.env.reset()

        self.mut_std = mut_std

        self.env.initPos = ini_pos
        posture = fs.Posture(*ini_pos)
        self.env.robot.set_pos(posture)

        self.init_pos = self.env.get_robot_pos()
        self.ball_held = -1
        self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])

        self.balls = [(self.env.get_robot_pos()[0] + 60 * np.cos((2*np.pi) * i/nb_ball),
                       self.env.get_robot_pos()[1] + 60 * np.sin((2*np.pi) * i/nb_ball)) for i in range(nb_ball)]

        self.windows_alive = False

        self.proximity_threshold = 10.0  # min distance required to catch or release ball

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
                    return 10.0
        return 0.0

    def release(self):
        if self.ball_held != -1:
            self.ball_held = -1
            if np.sqrt((self.pos[0] - self.init_pos[0])**2 + (self.pos[1] - self.init_pos[1])**2) \
                   < self.proximity_threshold:
                return 100.0
        return 0.0

    def __call__(self, agent, render=False, use_state_path=False, max_steps=2000, exceed_reward=0):
        self.add_balls()
        if render and not self.windows_alive:
            self.env.enable_display()
        state = self.env.reset()
        state.append(0.0)
        if len(agent.choose_action(state)) != 3:
            return AssertionError("The current agent returned an action of length != 3. Aborting.")
        done = False

        fitness = 0.0
        path = list()
        count = 0
        while not done:
            if len(self.balls) == 0:
                break
            if render:
                self.env.render()
                time.sleep(0.01)

            action = agent.choose_action(state)
            holding = action[2] > 0

            state, reward, done, info = self.env.step(action[:2])
            state.append(action[2])
            self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])

            reward = -0.1  # default reward is distance to goal

            if holding:
                reward += self.catch()

            if not holding:
                reward += self.release()

            if use_state_path:
                path.append(state)

            fitness += reward
            count += 1
            if count > max_steps:
                fitness += exceed_reward
                break
        return Configuration.metric(agent, self, fitness, path)

    def get_child(self):
        new_init_pos = (self.init_pos[0] + np.random.normal(0, self.mut_std),
                        self.init_pos[1] + np.random.normal(0, self.mut_std),
                        (self.init_pos[2] + np.random.normal(0, self.mut_std)) % 360)
        new_env = CollectBall(self.mut_std, ini_pos=new_init_pos)
        new_balls = list()
        for b in self.balls:
            # We try to avoid getting to close to the border
            new_balls.append((b[0] + np.random.normal(0, self.mut_std) % 580 + 10,
                              b[1] + np.random.normal(0, self.mut_std) % 580 + 10))
        new_env.balls = new_balls
        return new_env

    def crossover(self, other):
        new_init_pos = self.init_pos if np.random.uniform(0, 1) < 0.5 else other.init_pos
        new_env = CollectBall(self.mut_std, ini_pos=new_init_pos)
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
        dic["Std"] = self.mut_std
        dic["Init_pos"] = self.init_pos
        return dic

    def __setstate__(self, state):
        self.__init__(state["Std"], ini_pos=state["Init_pos"])
        self.balls = state["Balls"]

    def __del__(self):
        self.env.close()


class CollectBallFactory(EnvironmentFactory):

    def __init__(self, mut_std=25.0, ini_pos=(80, 480, 45), nb_balls=6):
        self.mut_std = mut_std
        self.ini_pos = ini_pos
        self.nb_balls = nb_balls

    def new(self):
        return CollectBall(mut_std=self.mut_std, ini_pos=self.ini_pos, nb_ball=self.nb_balls)
