from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.classic_control import Continuous_MountainCarEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm



class GoalMountainCar(gym.Wrapper):

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        ag = np.array(self.env.state)
        g = np.array([self.env.goal_position, self.env.goal_velocity])
        state = {'observation': state, 'achieved_goal': ag, 'desired_goal': g}
        return state

    def compute_reward(self, achieved_goal, desired_goal, info):
        shape = False
        dense = 100*((math.sin(3*achieved_goal[0]) * 0.0025 + 0.5 * achieved_goal[1] * achieved_goal[1]) - (math.sin(3*desired_goal[0]) * 0.0025 + 0.5 * desired_goal[1] * desired_goal[1])) 
        if achieved_goal[0] != desired_goal[0]:
            return -1 if not shape else dense
        else:
            return 0 if achieved_goal[0] >= desired_goal[0] else (-1 if not shape else dense)

    def step(self, action):
        state, _, done, info = super().step(action)
        ag = np.array(self.env.state)
        g = np.array([self.env.goal_position, self.env.goal_velocity])
        reward = self.compute_reward(ag, g, info)
        state = {'observation': state, 'achieved_goal': ag, 'desired_goal': g}
        info['is_success'] = reward==0
        return state, reward, done, info

class GoalMountainCarContinuous(gym.Wrapper):

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        ag = np.array(self.env.state)
        g = np.array([self.env.goal_position, self.env.goal_velocity])
        state = {'observation': state, 'achieved_goal': ag, 'desired_goal': g}
        return state

    def compute_reward(self, achieved_goal, desired_goal, info):
        shape = False
        dense = 100*((math.sin(3*achieved_goal[0]) * 0.0025 + 0.5 * achieved_goal[1] * achieved_goal[1]) - (math.sin(3*desired_goal[0]) * 0.0025 + 0.5 * desired_goal[1] * desired_goal[1])) 
        if achieved_goal[0] != desired_goal[0]:
            return -1 if not shape else dense
        else:
            return 0 if achieved_goal[0] >= desired_goal[0] else (-1 if not shape else dense)


    def step(self, action):
        state, _, done, info = super().step(action)
        ag = np.array(self.env.state)
        g = np.array([self.env.goal_position, self.env.goal_velocity])
        reward = self.compute_reward(ag, g, None)
        state = {'observation': state, 'achieved_goal': ag, 'desired_goal': g}
        info['is_success'] = ag[0] == g[0]
        return state, reward, done, info

# GoalMountainCarContinuous(gym.make("MountainCarContinuous-v0"))
# GoalMountainCar(gym.make(MountainCar-v0))

'''


# class DoorWrapper(Wrapper):
#     """
#     Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
#     found in the gym.core module
#     Args:
#         env (MujocoEnv): The environment to wrap.
#         keys (None or list of str): If provided, each observation will
#             consist of concatenated keys from the wrapped environment's
#             observation dictionary. Defaults to robot-state and object-state.
#     Raises:
#         AssertionError: [Object observations must be enabled if no keys]
#     """

#     def __init__(self, env, keys=None):
#         # Run super method
#         super().__init__(env=env)
#         # Create name for gym
#         robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
#         self.name = robots + "_" + type(self.env).__name__

#         # Get reward range
#         self.reward_range = (0, self.env.reward_scale)

#         if keys is None:
#             assert self.env.use_object_obs, "Object observations need to be enabled."
#             keys = ["object-state"]
#             # Iterate over all robots to add to state
#             for idx in range(len(self.env.robots)):
#                 keys += ["robot{}_robot-state".format(idx)]
#         self.keys = keys

#         # Gym specific attributes
#         self.env.spec = None
#         self.metadata = None
#         self.goal = np.array([.3])

#         # set up observation and action spaces
#         flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
#         self.obs_dim = flat_ob.size
#         high = np.inf * np.ones(self.obs_dim)
#         low = -high
#         self.observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(low=low, high=high), "achieved_goal": gym.spaces.Box(low=np.zeros(1), high=np.ones(1), shape=(1,)), "desired_goal": gym.spaces.Box(low=np.zeros(1), high=np.ones(1), shape=(1,))})
#         low, high = self.env.action_spec
#         self.action_space = gym.spaces.Box(low=low, high=high)

#     def _flatten_obs(self, obs_dict, verbose=False):
#         """
#         Filters keys of interest out and concatenate the information.
#         Args:
#             obs_dict (OrderedDict): ordered dictionary of observations
#             verbose (bool): Whether to print out to console as observation keys are processed
#         Returns:
#             np.array: observations flattened into a 1d array
#         """
#         ob_lst = []
#         for key in obs_dict:
#             if key in self.keys:
#                 if verbose:
#                     print("adding key: {}".format(key))
#                 ob_lst.append(obs_dict[key])
#         return np.concatenate(ob_lst)

#     def reset(self):
#         """
#         Extends env reset method to return flattened observation instead of normal OrderedDict.
#         Returns:
#             np.array: Flattened environment observation space after reset occurs
#         """
#         ob_dict = self.env.reset()
#         state = self._flatten_obs(ob_dict)
#         ag = np.array(self.env.sim.data.qpos[self.env.hinge_qpos_addr])
#         g = self.goal
#         return {'observation': state, 'achieved_goal': ag, 'desired_goal': g}

#     def step(self, action):
#         """
#         Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.
#         Args:
#             action (np.array): Action to take in environment
#         Returns:
#             4-tuple:
#                 - (np.array) flattened observations from the environment
#                 - (float) reward from the environment
#                 - (bool) whether the current episode is completed or not
#                 - (dict) misc information
#         """
#         ob_dict, reward, done, info = self.env.step(action)
#         state = self._flatten_obs(ob_dict)
#         ag = np.array(self.env.sim.data.qpos[self.env.hinge_qpos_addr])
#         g = self.goal
#         ob_dict = {'observation': state, 'achieved_goal': ag, 'desired_goal': g}
#         return ob_dict, reward, done, info

#     def seed(self, seed=None):
#         """
#         Utility function to set numpy seed
#         Args:
#             seed (None or int): If specified, numpy seed to set
#         Raises:
#             TypeError: [Seed must be integer]
#         """
#         # Seed the generator
#         if seed is not None:
#             try:
#                 np.random.seed(seed)
#             except:
#                 TypeError("Seed must be an integer type!")

#     def compute_reward(self, achieved_goal, desired_goal, info):
#         return 1 if achieved_goal[0] > desired_goal[0] else 0

env = suite.make(
                "Door",
                robots="Sawyer",                # use Sawyer robot
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=False,              # make sure we can render to the screen
                reward_shaping=False,            # use dense rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
'''

def experiment(variant):
    expl_env = NormalizedBoxEnv(Continuous_MountainCarEnv())
    eval_env = NormalizedBoxEnv(Continuous_MountainCarEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
