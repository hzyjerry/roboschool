#DEMO_ROOT = "/home/zhiyang/Desktop/intention/reacher/rl_demo_0"
DEMO_ROOT = "/home/zhiyang/Desktop/intention/reacher/rl_demo_1"



import gym, roboschool
import numpy as np

from roboschool.play import play
play(gym.make("RoboschoolReacherPlay-v1"))


import gym, roboschool
import numpy as np

from roboschool.play_policy import play
play(gym.make("RoboschoolReacherRGB_red-v1"), save_video=False, demo_root="/home/zhiyang/Desktop/intention/reacher/rl_demo_0")


import gym, roboschool
import numpy as np

from roboschool.play_policy import play
play(gym.make("RoboschoolReacherRGB_green-v1"), save_video=True, demo_root="/home/zhiyang/Desktop/intention/reacher/rl_demo_0")




import gym, roboschool
import numpy as np

from roboschool.play_policy import play
play(gym.make("RoboschoolReacherRGB-v1"), save_video=False, demo_root="/home/zhiyang/Desktop/intention/reacher/rl_demo_0")



import gym, roboschool
import numpy as np

from roboschool.play_policy import play
play(gym.make("RoboschoolReacherLine-v1"))
