import argparse
import logging
import os
import sys

import gym
from gym.spaces import Discrete, Box
import gym.wrappers
import pybullet_envs
import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers

import chainerrl
from chainerrl import experiments
from chainerrl import misc
from chainerrl.wrappers import atari_wrappers
from chainerrl.replay_buffers import replay_buffer

# from expert_dataset import ExpertDataset
# from observation_wrappers import (
#     generate_pov_converter, generate_pov_with_compass_converter,
#     generate_unified_observation_converter)
# from action_wrappers import (
#     generate_discrete_converter, generate_continuous_converter,
#     generate_multi_dimensional_softmax_converter)
from agent.behavioral_cloning import BehavioralCloning as BC
from network.bc_network import ContinuousBCNet
from expert_data_set import ExpertDataset
# from distribution import MultiDimensionalSoftmaxDistribution

class NormalizeActionSpace(gym.ActionWrapper):
    """Normalize a Box action space to [-1, 1]^n."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(env.action_space.low),
            high=np.ones_like(env.action_space.low),
        )

    def action(self, action):
        # action is in [-1, 1]
        action = action.copy()

        # -> [0, 2]
        action += 1

        # -> [0, orig_high - orig_low]
        action *= (self.env.action_space.high - self.env.action_space.low) / 2

        # -> [orig_low, orig_high]
        return action + self.env.action_space.low

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HopperBulletEnv-v0')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files. If it does not exist, it will be created.')
    parser.add_argument('--expert', type=str, default='demo',
                        help='Path storing expert trajectories.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--eval-interval', type=int, default=10000,
                        help='Interval in timesteps between evaluations.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load-demo', type=str, default='demo/HopperBulletEnv-v0/',
                        help='Directory to load replay buffer of demo from.')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--logging-level', type=int, default=10, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation.')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate.')
    parser.add_argument('--entropy-coef', type=float, default=0)
    parser.add_argument('--action-wrapper', type=str, default='continuous',
                        # choices=['discrete', 'continuous', 'multi-dimensional-softmax'])
                        choices = ['continuous'])
    parser.add_argument('--training-dataset-ratio', type=float, default=0.7,
                        help='ratio of training dataset on behavioral cloning between (0, 1)')
    parser.add_argument('--render', action='store_true', default=False)

    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level)

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv, time_format='%Y%m%dT%H%M%S')
    print('Output files are saved in {}'.format(args.outdir))

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    process_seed = args.seed
    assert process_seed < 2 ** 32

    def make_env(test):
        env = gym.make(args.env)
        # Unwrap TimiLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)

        if isinstance(env.observation_space, Box):
            # Cast observations to float32 because our model uses float32
            env = chainerrl.wrappers.CastObservationToFloat32(env)
        else:
            env = atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(args.env, max_frames=None),
                episode_life=not test,
                clip_rewards=not test)

        if isinstance(env.action_space, Box):
            # Normalize action space to [-1, 1]^n
            env = NormalizeActionSpace(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    action_space = env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    network = ContinuousBCNet(action_space)
    opt = optimizers.Adam(args.lr)
    opt.setup(network)

    agent = BC(network, opt, minibatch_size=1024,
               entropy_coef=args.entropy_coef, action_wrapper=args.action_wrapper)

    rbuf_demo = replay_buffer.ReplayBuffer(5 * 10 ** 5)
    assert len(args.load_demo) > 0
    rbuf_demo.load(os.path.join(args.load_demo, 'replay'))
    assert isinstance(rbuf_demo, replay_buffer.ReplayBuffer)

    expert_data = ExpertDataset(rbuf_demo, len(rbuf_demo))

    # separate train data and validation data
    all_obs = []
    all_action = []
    for i in range(expert_data.size):
        obs, action, _, _, _ = expert_data.sample()
        all_obs.append(obs)
        all_action.append(action)
    all_obs = np.array(all_obs)
    all_action = np.array(all_action)

    num_train_data = int(expert_data.size * args.training_dataset_ratio)

    train_obs = all_obs[:num_train_data]
    train_acs = all_action[:num_train_data]
    validate_obs = all_obs[num_train_data:]
    validate_acs = all_action[num_train_data:]

    agent.train(train_obs, train_acs, validate_obs, validate_acs)

    if args.demo:
        eval_stats = experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
        logging.logger.info('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()