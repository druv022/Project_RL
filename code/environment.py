import gym
import time
import numpy as np
"""
	Acrobot-v1: 
		
		Action space Discrete(3,). The action is either applying +1, 0 or -1 torque on the joint between
    	the two pendulum links

    	State space Box(6,). The state consists of the sin() and cos() of the two rotational joint
    	angles and the joint angular velocities : [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]
	
	CartPole-v1:
		
		Action Discrete(2). The action is either 0 Push cart to the left or 1 Push cart to the right.
		
		State is Box(4,). State consites of [Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip]

	MountainCar-v0:

		Action Discrete(3).
		State Box(2,) 
"""
def get_env(name):

	if name == "Acrobot-v1":
		
		env = gym.envs.make(name)
		#get state /action space length for our QNetwork
		space = (6, 3) #tuple of (state, action)

		return env, space

	elif name == "CartPole-v1":
		env = gym.envs.make(name)
		space = (4, 2)

		return env, space
	elif name == "MountainCar-v0":
		env = gym.envs.make(name)
		space = (2, 3)

		return env, space
	elif name == 'LunarLander-v2':
		env = gym.envs.make(name)
		space = (8,4)
		return env, space
	else:
		env = gym.envs.make(name)
		space = (2, 3)

		return env, space

	return 0 

if __name__=="__main__":
	env, _ = get_env("Acrobot-v1")
	a = env.action_space.sample()
	s = env.reset()
	print(env.action_space,s)