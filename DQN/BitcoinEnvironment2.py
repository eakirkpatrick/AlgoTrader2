import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

MAX_TRADING_SESSION = 4000

class BitcoinEnvironment(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}
    viewer = None
    scaler = preprocessing.MinMaxScaler()

    def __init__(self, df, lookback_window=60, commission=0.00075, initial_balance=10000, serial=False):
        super(BitcoinEnvironment, self).__init__()

        self.df = df
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial

        self.action_space = spaces.MultiDiscrete([3,10])

        self.observation_space = spaces.Box(low=0, high=1, shape=(10, lookback_window + 1), dtype=np.float16)


    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0

        self._reset_session()

        self.account_history = np.repeat([
        [self.net_worth],
        [0],
        [0],
        [0],
        [0]
        ], self.lookback_window + 1, axis=1)

        self.trades = []

        return self._next_observation()

    def _reset_session(self):
        self.current_step = 0

        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window - 1
            self.frame_start = self.lookback_window
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(self.lookback_window, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - self.lookback_window : self.frame_start + self.steps_left]

    def _next_observation(self):
        self.end = self.current_step + self.lookback_window + 1

        obs = np.array([
        self.active_df['OpenPrice'].values[self.current_step: self.end],
        self.active_df['High'].values[self.current_step: selfend],
        self.active_df['Low'].values[self.current_step: self.end],
        self.active_df['Close'].values[self.current_step: self.end],
        self.active_df['Volume'].values[self.current_step: self.end]
        ])

        scaled_history = self.scaler.fit_transform(self.account_history)

        obs = np.append(obs, scaled_history[:, -(self.lookback_window + 1):], axis=0)

        return obs

    def _get_current_price(self):
        return self.df.loc[self.current_step, 'Close']

    def step(self, action):
        current_price = self._get_current_price() + 0.01
        self._take_action(action, current_price)
        self.steps_left -= 1
        self.current_step += 1

        if self.steps_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            self._reset_session()

        obs = self._next_observation()
        reward = self.net_worth
        done = self.net_worth <= 0

        return obs, reward, done, {}

    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type < 1:
            btc_bought = self.balance / current_price * amount
            cost = btc_bought * current_price * (1 + self.commission)
            self.btc_held += btc_bought
            self.balance -= cost

        elif action_type < 2:
            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price * (1 - self.commission)
            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({
            'step': self.frame_start + self.current_step,
            'amount': btc_sold if btc_sold > 0 else btc_bought,
            'total': sales if btc_sold > 0 else cost,
            'type': "sell" if btc_sold > 0 else "buy"
            })

        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history, [
        [self.net_worth],
        [btc_bought],
        [cost],
        [btc_sold],
        [sales]
        ], axis=1)

    def render(self, mode='human', **kwargs):
        print('Net Worth: ', self.net_worth)

df = pd.read_csv('thisData.csv', index_col=0)

slice_point = 4000
train_df = df[:slice_point]
test_df = df[slice_point:]

train_env = DummyVecEnv([lambda: BitcoinEnvironment(train_df,
                         commission=0, serial=False)])
test_env = DummyVecEnv([lambda: BitcoinEnvironment(test_df,
                        commission=0, serial=True)])

model = PPO2(MlpPolicy,
             train_env,
             verbose=1,
             tensorboard_log="./tensorboard/")
model.learn(total_timesteps=4000)

state = None
buy = []
sell = []
tradesEnv = []
rewards = []
obs = test_env.reset()
for i in range(len(test_df)):
    action, state = model.predict(obs)
    if action[0] == 0:
        buy.append(test_env.end)
    elif action[0] == 1:
        sell.append(test_env.end)

    tradesEnv.append(obs[3, test_env.end])

    obs, reward, done, info = test_env.step([action[0]])


    rewards.append(reward)

    test_env.render(mode='human')

plt.plot(test_df['OpenTime'], tradesEnv, '-', buy, tradesEnv, 'g^', sell, tradesEnv, 'bs')
plt.show()
                                                                                                                                                                                          157,23        92%

                                                                                                                                                                                          103,22        46%

                                                                                                                                                                                          1,10          Top
