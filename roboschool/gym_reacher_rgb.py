from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym import error, spaces
import numpy as np
import pygame
import os, sys
from profiler import Profiler

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))



class RoboschoolReacherRGB(RoboschoolMujocoXmlEnv):
    def __init__(self, display=True):
        RoboschoolMujocoXmlEnv.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=9)
        low = np.ones([self.VIDEO_H,self.VIDEO_W,3]) * -255
        high = np.ones([self.VIDEO_H,self.VIDEO_W,3]) * 255
        self.metadata = {"video": gym.spaces.Box(low, high)}
        force = 0.3
        self._action_set = np.array(((force, 0),
                            (-force, 0),
                            (0, force),
                            (0, -force),
                            (0, 0)), dtype=float)
        self._action_set_keys = [0, 1, 2, 3, 4]
        self.display = display
        self.screen = None
        self.transpose = True
            
        if self.display:
            if self.transpose:
                self.video_size = self.VIDEO_W, self.VIDEO_H
            else:
                self.video_size = self.VIDEO_H, self.VIDEO_W
            self.screen = pygame.display.set_mode(self.video_size)
            self.clock = pygame.time.Clock()

        #self.action_space = spaces.Discrete(len(self._action_set))

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    TARG_LIMIT = 0.27
    def robot_specific_reset(self):
        target_pos_1 = [-0.2, -0.1]
        target_pos_2 = [-0.2, 0.1]
        ## Reset at fixed location
        self.jdict["target_x"].reset_current_position(target_pos_2[0], 0)
        self.jdict["target_y"].reset_current_position(target_pos_2[1], 0)
        self.jdict["target_x_2"].reset_current_position(target_pos_1[0], 0)
        self.jdict["target_y_2"].reset_current_position(target_pos_1[1], 0)
        self.fingertip = self.parts["fingertip"]
        self.target    = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint   = self.jdict["joint1"]
        ## Reset at upper half of the table
        #self.central_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        #self.elbow_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.central_joint.reset_current_position(self.np_random.uniform( low=-np.pi/2, high=np.pi/2 ), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform( low=-np.pi/2, high=np.pi/2 ), 0)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        self.central_joint.set_motor_torque( 0.05*float(np.clip(a[0], -1, +1)) )
        self.elbow_joint.set_motor_torque( 0.05*float(np.clip(a[1], -1, +1)) )

    def calc_state(self):
        theta,      self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
            ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)

    def _step(self, a):
        assert(not self.scene.multiplayer)
        #a = self._action_set[a]
        self.apply_action(a)
        self.scene.global_step()

        motor_state = self.calc_state()  # sets self.to_target_vec
        aux_state = np.array([motor_state[-3], motor_state[-1]])
        state = self._render("rgb_array", False)

        potential_old = self.potential
        self.potential = self.calc_potential()

        electricity_cost = (
            -0.10*(np.abs(a[0]*self.theta_dot) + np.abs(a[1]*self.gamma_dot))  # work torque*angular_velocity
            -0.01*(np.abs(a[0]) + np.abs(a[1]))                                # stall torque require some energy
            )
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        self.frame  += 1
        done = np.linalg.norm(self.to_target_vec) < 0.02
        #print(np.linalg.norm(self.to_target_vec), done)
        if done:
            self.done   += 1
        #self.done   += 0
        self.reward += sum(self.rewards)
        self.HUD(state, a, False)
        if self.display:
            display_arr(self.screen, state, transpose=self.transpose, video_size=self.video_size)
            pygame.display.flip()
            self.clock.tick(100)
        return state, sum(self.rewards), done, {'motor': motor_state, "aux": aux_state}

    def _reset(self):
        motor_state = RoboschoolMujocoXmlEnv._reset(self)
        aux_state = np.array([motor_state[-3], motor_state[-1]])
        return self._render("rgb_array", False), {'motor': motor_state, "aux": aux_state}

    def camera_adjust(self):
        x, y, z = self.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        #self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
        self.camera.move_and_look_at(0.0, 0.0, np.pi/8, 0, 0, 0)

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'JOINT_A_PLUS':      ord('w'),
            'JOINT_A_MINUS':     ord('s'),
            'JOINT_B_PLUS':      ord('a'),
            'JOINT_B_MINUS':     ord('d')
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id
        print(keys_to_action)
        return keys_to_action

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set_keys]


ACTION_MEANING = {
    0 : "JOINT_A_PLUS",
    1 : "JOINT_A_MINUS",
    2 : "JOINT_B_PLUS",
    3 : "JOINT_B_MINUS",
    4 : "IDLE"
}

class RoboschoolReacherRGB_infinite(RoboschoolReacherRGB):
    def _step(self, a):
        assert(not self.scene.multiplayer)
        #a = self._action_set[a]
        self.apply_action(a)
        self.scene.global_step()

        motor_state = self.calc_state()  # sets self.to_target_vec
        aux_state = np.array([motor_state[-3], motor_state[-1]])
        state = self._render("rgb_array", False)

        potential_old = self.potential
        self.potential = self.calc_potential()

        electricity_cost = (
            -0.10*(np.abs(a[0]*self.theta_dot) + np.abs(a[1]*self.gamma_dot))  # work torque*angular_velocity
            -0.01*(np.abs(a[0]) + np.abs(a[1]))                                # stall torque require some energy
            )
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        self.frame  += 1
        self.done   += 0
        #self.done   += 0
        self.reward += sum(self.rewards)

        ## Main speed bottleneck
        #self.HUD(state, a, False)
        if self.display:
            display_arr(self.screen, state, transpose=self.transpose, video_size=self.video_size)
            pygame.display.flip()
            self.clock.tick(100)
        return state, sum(self.rewards), False, {'motor': motor_state, "aux": aux_state}



class RoboschoolReacherRGB_red(RoboschoolReacherRGB):
    def __init__(self):
        RoboschoolMujocoXmlEnv.__init__(self, 'reacher_red.xml', 'body0', action_dim=2, obs_dim=9)
        low = np.ones([self.VIDEO_H,self.VIDEO_W,3]) * -255
        high = np.ones([self.VIDEO_H,self.VIDEO_W,3]) * 255
        self.metadata = {"video": gym.spaces.Box(low, high)}
        force = 0.3
        self._action_set = np.array(((force, 0),
                            (-force, 0),
                            (0, force),
                            (0, -force),
                            (0, 0)), dtype=float)
        self._action_set_keys = [0, 1, 2, 3, 4]
        #self.action_space = spaces.Discrete(len(self._action_set))

        TARG_LIMIT = 0.27
    def robot_specific_reset(self):
        target_pos_1 = [-0.2, -0.1]
        target_pos_2 = [-0.2, 0.1]
        ## Reset at fixed location
        self.jdict["target_x"].reset_current_position(target_pos_1[0], 0)
        self.jdict["target_y"].reset_current_position(target_pos_1[1], 0)
        self.jdict["target_x_2"].reset_current_position(target_pos_2[0], 0)
        self.jdict["target_y_2"].reset_current_position(target_pos_2[1], 0)
        self.fingertip = self.parts["fingertip"]
        self.target    = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint   = self.jdict["joint1"]
        ## Reset at upper half of the table
        #self.central_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        #self.elbow_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.central_joint.reset_current_position(self.np_random.uniform( low=-np.pi/2, high=np.pi/2 ), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform( low=-np.pi/2, high=np.pi/2 ), 0)


class RoboschoolReacherRGB_green(RoboschoolReacherRGB):
    def __init__(self):
        RoboschoolReacherRGB.__init__(self)

        TARG_LIMIT = 0.27
    def robot_specific_reset(self):
        target_pos_1 = [-0.2, -0.1]
        target_pos_2 = [-0.2, 0.1]
        ## Reset at fixed location
        self.jdict["target_x"].reset_current_position(target_pos_2[0], 0)
        self.jdict["target_y"].reset_current_position(target_pos_2[1], 0)
        self.jdict["target_x_2"].reset_current_position(target_pos_1[0], 0)
        self.jdict["target_y_2"].reset_current_position(target_pos_1[1], 0)
        self.fingertip = self.parts["fingertip"]
        self.target    = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint   = self.jdict["joint1"]
        ## Reset at upper half of the table
        #self.central_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        #self.elbow_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.central_joint.reset_current_position(self.np_random.uniform( low=-np.pi/2, high=np.pi/2 ), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform( low=-np.pi/2, high=np.pi/2 ), 0)
