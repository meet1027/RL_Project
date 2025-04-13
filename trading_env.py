import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, stock_dim, hmax, initial_amount, num_stock_shares,
                buy_cost_pct, sell_cost_pct, reward_scaling, tech_indicator_list,
                turbulence_threshold=None, risk_indicator_col="turbulence",
                make_plots=False, print_verbosity=10, day=0, initial=True,
                previous_state=[], model_name="", mode="", iteration=""):
        
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Action/Observation Space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.tech_indicator_list)+2*self.stock_dim+1,))
        
        # Initialize state
        self.data = self._get_multi_stock_data()
        self.state = self._initiate_state()
        self.reset()

        # Tracking variables
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.cost = 0
        self.trades = 0
        self.turbulence = 0
        self.episode = 0

    def _sell_stock(self, index, action):
        if self.state[index + 2*self.stock_dim + 1] != True:
            if self.state[index + self.stock_dim + 1] > 0:
                sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                
                self.state[0] += sell_amount
                self.state[index + self.stock_dim + 1] -= sell_num_shares
                self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1
                return sell_num_shares
            else:
                return 0
        else:
            return 0

    def _buy_stock(self, index, action):
        if self.state[index + 2*self.stock_dim + 1] != True:
            available_amount = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
            buy_num_shares = min(available_amount, action)
            
            buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
            self.state[0] -= buy_amount
            self.state[index + self.stock_dim + 1] += buy_num_shares
            self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
            self.trades += 1
            return buy_num_shares
        else:
            return 0

    def calculate_reward(self):
        current_total_asset = self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim+1)]) * 
            np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
        )
        
        df_total_value = pd.DataFrame(self.asset_memory, columns=["account_value"])
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change()
        
        risk_component = -np.std(df_total_value["daily_return"]) * 100 if len(self.asset_memory) > 1 else 0
        cost_component = -self.cost * 0.1
        return_component = (current_total_asset - self.asset_memory[-2])/self.asset_memory[-2] if len(self.asset_memory) > 1 else 0
        
        return (return_component * 0.5 + risk_component * 0.2 + cost_component * 0.1) * self.reward_scaling

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}
        
        else:
            actions = actions * self.hmax
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim+1)]) *
                np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
            )
            
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])
            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self._get_multi_stock_data()  
            self.state = self._update_state()
            
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim+1)]) *
                np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
            )
            
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = self.calculate_reward()
            self.rewards_memory.append(self.reward)
            
            return self.state, self.reward, self.terminal, False, {}

    def reset(self, **kwargs):
        self.day = 0
        self.data = self._get_multi_stock_data()
        self.state = self._initiate_state()
        
        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1:(self.stock_dim+1)]) *
                np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)])
            )
            self.asset_memory = [previous_total_asset]
            
        self.cost = 0
        self.trades = 0
        self.turbulence = 0
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        
        return self.state, {}

    def _initiate_state(self):
        if self.initial:
            return [self.initial_amount] + \
                   self.data.close.values.tolist() + \
                   [0]*self.stock_dim + \
                   sum([[tech] for tech in self.data[self.tech_indicator_list].values.T.tolist()], [])
        else:
            return [self.previous_state[0]] + \
                   self.data.close.values.tolist() + \
                   self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)] + \
                   sum([[tech] for tech in self.data[self.tech_indicator_list].values.T.tolist()], []

    def _update_state(self):
        return [self.state[0]] + \
               self.data.close.values.tolist() + \
               list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
               sum([[tech] for tech in self.data[self.tech_indicator_list].values.T.tolist()], []

    def _get_multi_stock_data(self):
        return self.df.loc[self.day*self.stock_dim : (self.day+1)*self.stock_dim-1]

    def _get_date(self):
        return self.data.date.values[0]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
