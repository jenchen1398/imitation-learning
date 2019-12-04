import cv2
import torch
from ddpg_bipedal.ddpg_agent import Agent
import gym

class BipedalWalker2:
	def __init__(self, render=True, actor_path=None, critic_path=None):
		self.actor_path = actor_path
		self.critic_path = critic_path
		self.env = gym.make('BipedalWalker-v2')
		self.env.reset()

		self.agent = Agent(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.shape[0], random_seed=10)
		if actor_path:
			self.agent.actor_local.load_state_dict(torch.load(actor_path))
		if critic_path:
			self.agent.critic_local.load_state_dict(torch.load(critic_path))


	def save_img(self, i, path, save_crop=False, get_crop=True):
		arr = self.env.render(mode='rgb_array')
		crop_arr = arr[120:-10,0:300]
		cv2.imwrite(path + "img_{}.png".format(i), arr)
		return crop_arr
	
	def move_bipedal(self, action):
		next_state, reward, done, _ = self.env.step(action)
		return next_state, done

	def close(self):
		self.env.close()
