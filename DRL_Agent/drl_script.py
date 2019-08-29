from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DRL_Agent.BiddingSimulator import BiddingSimulator
from DRL_Agent.agent_def import DeepLearningAgent
from argparse import ArgumentParser


def run_experiment(env, agent, number_of_steps):
    mean_reward = 0.
    reward_per_episode = []
    rewards = []
    actions = []
    states_df = pd.DataFrame(columns=['sessions', 'CPC', 'CVR', 'CTR', 'conversion', 'avg_pos',
                                      'impressions', 'search_imp_share', 'action', 'action_value'])

    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0

    next_state = np.empty((len(states_df.columns)))
    next_state.fill(np.nan)
    state_dict = dict(zip(states_df.columns, next_state.squeeze()))
    state_dict['action'] = action
    state_dict['action_value'] = env._actions[action]
    states_df = states_df.append(state_dict, ignore_index=True)
    accum_reward = 0
    for i in range(number_of_steps):
        reward, discount, next_state = env.step(action)
        rewards.append(reward)
        if discount == 0:
            reward_per_episode.append(accum_reward)
            reward = 0
        else:
            accum_reward += reward
        action = agent.step(reward, discount, next_state)
        actions.append(action)
        mean_reward += (reward - mean_reward) / (i + 1.)

        state_dict = dict(zip(states_df.columns, next_state.squeeze()))
        state_dict['action'] = action
        state_dict['action_value'] = env._actions[action]
        states_df = states_df.append(state_dict, ignore_index=True)

    return mean_reward, reward_per_episode, rewards, states_df


def smooth(array, smoothing_horizon=100., initial_value=0.):
    smoothed_array = []
    value = initial_value
    b = 1. / smoothing_horizon
    m = 1.
    for x in array:
        m *= 1. - b
        lr = b / (1 - m)
        value += lr * (x - value)
        smoothed_array.append(value)
    return np.array(smoothed_array)


def main(a):
    a.predictors = a.predictors.strip('[]').split(',')
    a.targets = a.targets.strip('[]').split(',')
    df = pd.read_csv(a.csv_filename, index_col=a.date_field_name, parse_dates=[a.date_field_name])
    models_df = pd.read_csv(a.env_report)
    report_df = pd.DataFrame(columns=['keyword', 'device_type', 'reward', 'agent_reward', 'loss_min',
                                      'min_at_step', 'loss_max', 'max_at_step'])

    if a.keyword is not None:
        models_df = models_df[(models_df[a.k_field] == a.keyword)]

    for model_idx, model_row in models_df.iterrows():
        kwd = model_row[a.k_field]
        dev_type = model_row[a.d_field]
        if model_row['time_steps'] >= 32:
            kwd_df = df[(df[a.k_field] == kwd) & (df[a.d_field] == dev_type)].copy()
            del kwd_df[a.k_field], kwd_df[a.d_field]

            simulator = BiddingSimulator(kwd_df, a.action_col, a.predictors, a.targets, model_row['model_file'],
                                         episode_length=30)
            dla = DeepLearningAgent(8, 100, simulator.get_observation(), int(kwd_df.shape[0] * 0.5),
                                    int(kwd_df.shape[0] * 0.1))
            print("Running training for keyword '{}' and device type '{}'...".format(kwd, dev_type))
            start_time = datetime.now()
            agent_reward, episode_reward, rewards_list, generated_states = run_experiment(simulator, dla, kwd_df.shape[0] * 5)
            # Model saving
            model_name = '{}tf_{}_{}.ckpt'.format(a.agents_dir, kwd, dev_type)
            dla.q_model.save(model_name)
            elapsed_time = datetime.now() - start_time
            print("Training completed in '{}'".format(elapsed_time))
            losses_array = np.array(dla._losses)
            # Loss graph saving
            plt.xlabel('Training step')
            plt.title('Loss')
            plt.plot(smooth(losses_array))
            plt.savefig('{}graphs/loss_{}_{}.png'.format(a.agents_dir, kwd, dev_type))
            plt.gcf().clear()
            # Reward graph saving
            plt.xlabel('Training step')
            plt.title('Net Profit')
            plt.plot(smooth(rewards_list))
            plt.savefig('{}graphs/reward_{}_{}.png'.format(a.agents_dir, kwd, dev_type))
            plt.gcf().clear()

            generated_states.to_csv('{}gen_states_{}_{}.csv'.format(a.agents_dir, kwd, dev_type), index=False)
            report_df = report_df.append({'keyword': kwd, 'device_type': dev_type, 'reward': agent_reward,
                                          'agent_reward': max(episode_reward),
                                          'loss_min': losses_array.min(), 'min_at_step': losses_array.argmin(),
                                          'loss_max': losses_array.max(), 'max_at_step': losses_array.argmax()},
                                         ignore_index=True)
            del kwd_df

    return report_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--csv", dest="csv_filename",
                        help="Features path and filename", metavar="CSV")
    parser.add_argument("-r", "--report", dest="report_filename",
                        help="CSV to report values on agents performance", metavar="Report CSV")
    parser.add_argument("-mr", "--env_report", dest="env_report",
                        help="Environment report with source directory", metavar="Models Report CSV")
    parser.add_argument("-d", "--date_field", dest="date_field_name",
                        help="Name of the date field to pick the time series from", metavar="Date field")
    parser.add_argument("-p", "--predictor_fields", dest="predictors",
                        help="Predictor column indexes", metavar="Predictor columns")
    parser.add_argument("-a", "--action_column", dest="action_col",
                        help="Action column name", metavar="Action column")
    parser.add_argument("-y", "--target_fields", dest="targets",
                        help="Target column indexes", metavar="Target columns")
    parser.add_argument("-k", "--keyword_field", dest="k_field",
                        help="Keyword field name", metavar="Keyword")
    parser.add_argument("-dt", "--device_type_field", dest="d_field",
                        help="Device type field name", metavar="Device type")
    parser.add_argument("-ad", "--agents_dir", dest="agents_dir",
                        help="Source directory for trained agents", metavar="Agents directory")
    parser.add_argument("-kwd", "--single_keyword", dest="keyword",
                        help="Analysis on a single keyword", metavar="Keyword analysis")
    args = parser.parse_args()
    global_start_time = datetime.now()
    report = main(args)
    global_elapsed_time = datetime.now() - global_start_time
    print("Total training time in '{}'".format(global_elapsed_time))
    report.to_csv(args.report_filename, index=False)
