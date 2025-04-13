from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

class RLAgent:
    def __init__(self, env):
        self.env = env
        self.model = None

    def train(self, total_timesteps=100_000):
        vec_env = make_vec_env(lambda: Monitor(self.env, '/'), n_envs=1)
        self.model = PPO("MlpPolicy", vec_env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)

    def optimize_hyperparameters(self, n_trials=10):
        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            gamma = trial.suggest_float("gamma", 0.8, 0.999)
            n_steps = trial.suggest_int("n_steps", 128, 2048, step=128)
            batch_size = trial.suggest_int("batch_size", 32, 1024, step=64)
            n_epochs = trial.suggest_int("n_epochs", 3, 20)
            ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
            clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

            vec_env = make_vec_env(lambda: Monitor(self.env, '/'), n_envs=1)
            model = PPO(
                "MlpPolicy", vec_env,
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                ent_coef=ent_coef,
                clip_range=clip_range,
                verbose=1
            )
            model.learn(total_timesteps=50_000)
            mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)
            return mean_reward

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
