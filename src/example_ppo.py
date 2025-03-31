import os
import numpy as np  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import gymnasium as gym  # type: ignore[import-untyped]
from stable_baselines3 import PPO  # type: ignore[import-untyped]
from stable_baselines3.common.env_util import make_vec_env  # type: ignore[import-untyped]

# Import Monitor for evaluation environment wrapping
from stable_baselines3.common.monitor import Monitor  # type: ignore[import-untyped]
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore[import-untyped]
from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore[import-untyped]
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import-untyped]
from typing import TypedDict
import time

plt.style.use(["science", "grid"])
# Register the environment
gym.register(
    id="MultiInv-v0",
    entry_point="example_env:MultiInvEnv",
)

# Environment parameters
n_locations = 4
starting_inventory = [20.0 for _ in range(n_locations)]
kwargs = {
    "n_locations": n_locations,
    "n_timesteps": 50,
    "starting_inventory": starting_inventory,
    "lead_times": [[2 for _ in range(n_locations)] for _ in range(n_locations)],
    "costs": {"unit": 1.0, "holding": 1.0, "shipment": 10.0},
    "unit_cost_param": 1.0,
    "mission_params": None,
    "random_demand_params": [
        {"missions": (8, 12), "distance": (5000.0, 5000.0), "aircraft": (7, 9)}
        for _ in range(n_locations)
    ],
    "random_demand_fixed_param": 1_000_000.0,
    "random_demand_dist_params": {"p": (0.01, 0.03)},
    "seed": 42,  # Fixed seed for reproducibility
    "scenario": "depot",
    "variance": 1.0,
}


class WrapperKwargs(TypedDict):
    norm_obs: bool
    norm_reward: bool
    clip_obs: float
    clip_reward: float


# Create output directories
os.makedirs("./ppo_training_logs", exist_ok=True)
os.makedirs("./ppo_inventory_plots", exist_ok=True)
os.makedirs("./ppo_models", exist_ok=True)
os.makedirs("./ppo_csv_data", exist_ok=True)


def get_inventory_dataframe(inventory_history, n_locations):
    """Convert inventory history to DataFrame for plotting and saving"""
    time_steps = list(range(len(inventory_history)))
    df = pd.DataFrame({"time": time_steps})
    for loc in range(n_locations):
        # Ensure history contains valid data before accessing index
        df[f"Location_{loc}"] = [
            inventory[loc] if len(inventory) > loc else None
            for inventory in inventory_history
        ]
    return df


def get_stockout_dataframe(stockouts):
    """Convert stockouts to DataFrame for plotting"""
    if not stockouts:
        # Return dataframe with correct columns even if empty
        return pd.DataFrame(columns=["location", "step"])
    return pd.DataFrame(stockouts)


def plot_inventory_levels(
    inventory_history, stockouts, n_locations, episode_num, total_timesteps
):
    """Plot inventory levels and save the figure"""
    inventory_df = get_inventory_dataframe(inventory_history, n_locations)
    stockout_df = get_stockout_dataframe(stockouts)

    # Save inventory data to CSV
    csv_filename = f"./ppo_csv_data/inventory_episode_{episode_num}_timesteps_{total_timesteps}.csv"
    inventory_df.to_csv(csv_filename, index=False)

    fig, axes = plt.subplots(
        nrows=(n_locations + 1) // 2,
        ncols=2,
        figsize=(15, 3 * ((n_locations + 1) // 2)),
    )
    axes = axes.flatten()

    for i in range(n_locations):
        ax = axes[i]
        # Plot inventory levels
        ax.step(
            inventory_df["time"],
            inventory_df[f"Location_{i}"],
            where="post",
            linewidth=2,
        )

        # Mark stockouts - Use 'step' column from stockout_df
        location_stockouts = stockout_df[stockout_df["location"] == i]
        if not location_stockouts.empty:
            unique_times = np.sort(location_stockouts["step"].unique())
            ax.scatter(
                unique_times,
                # Ensure we plot at 0 level on y-axis for stockouts
                [0] * len(unique_times),
                color="red",
                s=100,
                marker="X",
                label="Stockout",
            )

        # Fill between
        ax.fill_between(
            inventory_df["time"],
            inventory_df[f"Location_{i}"],
            step="post",
            alpha=0.3,
        )

        ax.set_title(f"Location {i}")
        ax.set_xlabel("Time Step")  # Changed label for clarity
        ax.set_ylabel("Inventory")
        ax.grid(True, alpha=0.3)
        # Only show legend if there are stockouts to avoid empty legend box
        if not location_stockouts.empty:
            ax.legend()
        else:  # Add placeholder legend entry if no stockouts
            # ax.plot([], [], " ", label="No Stockouts")
            ax.legend()

    # Hide any unused subplots
    for i in range(n_locations, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    # Updated title to reflect plot content more accurately
    plt.suptitle(
        f"Timesteps {total_timesteps}",
        fontsize=16,
    )
    plt.subplots_adjust(top=0.92)

    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(
        f"./ppo_inventory_plots/inventory_episode_{episode_num}_timesteps_{total_timesteps}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Custom callback structure (though we won't use its _on_step in manual eval)
class InventoryLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.inventory_histories = []
        self.stockout_events = []
        self.n_locations = 0  # Will be set later

    def set_n_locations(self, n_locations):
        self.n_locations = n_locations

    def _on_training_start(self) -> None:
        # Useful if using with model.learn and callbacks list
        pass

    def _on_step(self) -> bool:
        # This method is designed for the SB3 training loop (self.locals available)
        # We will log manually in the evaluation loop instead.
        # obs = self.locals.get("obs")
        # if isinstance(obs, dict): # Handle MultiInputPolicy case
        #      inventory = obs.get("inventory")
        #      if inventory is not None:
        #           # For VecEnv, obs might be batched, take the first one
        #           current_inv = inventory[0] if isinstance(inventory, np.ndarray) and inventory.ndim > 1 else inventory
        #           self.inventory_histories.append(current_inv.copy())
        #           for loc, inv in enumerate(current_inv):
        #                if inv <= 0: # Use <= 0 for safety
        #                     self.stockout_events.append({"location": loc, "step": self.num_timesteps})
        return True

    def reset_logs(self):
        self.inventory_histories = []
        self.stockout_events = []


def main():
    start_time = time.time()
    print("Starting training and evaluation process...")

    # Wrapper kwargs for VecNormalize
    wrapper_kwargs: WrapperKwargs = {
        "norm_obs": True,
        "norm_reward": True,  # Normalize reward during training
        "clip_obs": 10.0,
        "clip_reward": 10.0,
    }
    # Define keys to normalize
    norm_obs_keys = [
        "inventory",
        "expected_demand",
        "expected_demand_ratio",
        "pipeline_inventory",
    ]

    # Create training environment (vectorized)
    n_envs = 8
    env = make_vec_env(
        "MultiInv-v0",
        n_envs=n_envs,
        monitor_dir="./ppo_training_logs/",
        env_kwargs=kwargs,
    )
    env = VecNormalize(
        env,
        norm_obs=wrapper_kwargs["norm_obs"],
        norm_reward=wrapper_kwargs["norm_reward"],
        clip_obs=wrapper_kwargs["clip_obs"],
        clip_reward=wrapper_kwargs["clip_reward"],
        norm_obs_keys=norm_obs_keys,
        # Pass gamma, needed if norm_reward=True
        # gamma=0.99 # Assuming default gamma, or get from model later if needed
    )

    # --- CORRECTED EVALUATION ENVIRONMENT CREATION ---
    # Create a function to generate the monitored environment
    def make_eval_env():
        eval_core_env = gym.make("MultiInv-v0", **kwargs)
        # Wrap with Monitor to track episode stats correctly
        monitored_env = Monitor(eval_core_env)
        return monitored_env

    # Use DummyVecEnv for the single evaluation environment
    eval_env = DummyVecEnv([make_eval_env])

    # Path for normalization stats
    vecnormalize_path = "./ppo_training_logs/vecnormalize.pkl"

    # Load normalization stats if they exist and apply to eval_env
    eval_env = DummyVecEnv([make_eval_env])
    if os.path.exists(vecnormalize_path):
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        eval_env.norm_obs = False
        print("Evaluation environment normalization loaded.")
    else:
        # If stats don't exist (e.g., first run), create VecNormalize wrapper without loading
        # Note: This eval env won't be properly normalized until after the first save/load cycle
        print(
            "VecNormalize statistics not found. Creating new VecNormalize for evaluation (will not be normalized until first save)."
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=wrapper_kwargs["norm_obs"],
            norm_reward=False,  # Keep False for evaluation
            clip_obs=wrapper_kwargs["clip_obs"],
            clip_reward=wrapper_kwargs["clip_reward"],
            norm_obs_keys=norm_obs_keys,
            training=False,  # Ensure not training
        )
    # --- END CORRECTION ---

    # Define network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128, 128],
            vf=[128, 128, 128],
            # qf is not used by PPO, safe to remove or ignore
            # qf=[128, 128, 128]
        )
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        batch_size=256,  # Reduced batch size might help with smaller envs/memory
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=1,
        clip_range=0.25,  # Might explore 0.2
        learning_rate=0.0001,  # 1e-4 is often a good starting point
        tensorboard_log="./ppo_training_logs/tensorboard/",
        # Consider adding ent_coef for exploration, n_steps, n_epochs if needed
        # n_steps=1024 # Common PPO parameter
    )
    # Ensure gamma used in VecNormalize matches model gamma if reward normalization is ever turned on for eval
    # env.gamma = model.gamma
    # eval_env.gamma = model.gamma # Set gamma for eval_env too if norm_reward=True during eval

    total_timesteps = 0
    max_timesteps = 500_000
    increment = 50_000
    episode_num = 0  # Tracks training *iterations*, not environment episodes

    # Initialize the callback object for storing eval data
    inventory_callback = InventoryLoggingCallback()
    inventory_callback.set_n_locations(
        n_locations
    )  # Inform callback about env dimension

    while total_timesteps < max_timesteps:
        print(f"\n{'=' * 50}")
        print(f"Starting training iteration {episode_num + 1}")
        print(f"Current total timesteps: {total_timesteps}")
        print(f"Target total timesteps: {total_timesteps + increment}")
        print(f"{'=' * 50}")

        # Train for the next increment
        model.learn(
            total_timesteps=increment,
            reset_num_timesteps=False,
            tb_log_name=f"PPO_run_{episode_num}",
        )
        total_timesteps += increment
        episode_num += 1

        # Save model checkpoint and normalization statistics
        model_path = f"./ppo_models/ppo_multiinv_{total_timesteps}"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        # Save the normalization stats from the *training* environment
        env.save(vecnormalize_path)
        print(f"Normalization stats saved to {vecnormalize_path}")

        # --- Quick Evaluation using evaluate_policy (uses the corrected eval_env) ---
        # It's good practice to reload the normalization stats for the eval_env before evaluation
        # This ensures it uses the *latest* stats saved from the training env
        if os.path.exists(vecnormalize_path):
            print(
                f"Reloading normalization stats into eval_env from {vecnormalize_path}"
            )
            eval_env = VecNormalize.load(vecnormalize_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False  # Ensure reward norm remains off for eval
        else:
            print(
                "Warning: Cannot reload normalization stats for eval_env, file not found."
            )

        # Now evaluate the policy
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=5, deterministic=True
        )
        print(
            f"Quick Evaluation (evaluate_policy) - Mean Reward: {mean_reward:.4f}, Std Reward: {std_reward:.4f}"
        )

        # --- Detailed Evaluation with Manual Loop and Logging ---
        print("Starting detailed evaluation run for logging...")
        inventory_callback.reset_logs()  # Clear logs from previous iteration

        # Reset the evaluation environment
        try:
            obs = eval_env.reset()
            print(
                f"Initial normalized obs shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}"
            )
            initial_original_obs = eval_env.get_original_obs()
            print(
                f"Initial original obs structure (type {type(initial_original_obs)}): {initial_original_obs}"
            )
        except Exception as e:
            print(
                f"FATAL: Error during initial eval_env reset or get_original_obs: {e}"
            )
            # Exit or handle this critical failure
            return

        done = False
        num_steps = 0
        max_eval_steps = kwargs.get("n_timesteps", 100) + 10  # Safety margin

        while not done:
            if num_steps >= max_eval_steps:
                print(
                    f"Warning: Evaluation loop exceeded max_eval_steps ({max_eval_steps}) without termination signal."
                )
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            try:
                # Try to get inventory directly from the unwrapped environment
                raw_env = eval_env.venv.envs[0].env.unwrapped

                # Check if we can access inventory directly
                if hasattr(raw_env, "inventory"):
                    # Use the raw inventory value
                    current_inventory = raw_env.inventory
                    print(f"Direct inventory access: {current_inventory}")
                    inventory_callback.inventory_histories.append(
                        current_inventory.copy()
                    )

                    # Log stockouts using raw inventory
                    for loc, inv in enumerate(current_inventory):
                        if inv <= 0:
                            inventory_callback.stockout_events.append(
                                {"location": loc, "step": num_steps}
                            )
                else:
                    # Fall back to unnormalized observation if direct access isn't available
                    # Keep your existing normalized/unnormalized code as fallback
                    print(
                        "Using observation-based inventory (no direct access available)"
                    )
                    unnormalized_obs_dict = eval_env.unnormalize_obs(obs)

                    # Continue with your existing code for processing unnormalized observations
                    # ...
                # --- End Validation ---

                # --- SQUEEZE and Logging ---
                inventory_np = np.array(current_inventory)

                # --->>> CRITICAL FIX: Remove singleton dimensions <<<---
                inventory_squeezed = np.squeeze(inventory_np)
                # Now inventory_squeezed should have shape (n_locations,)

                # Check if squeezing resulted in a scalar (unexpected but possible if n_locations=1)
                if inventory_squeezed.ndim == 0:
                    inventory_final = np.array(
                        [inventory_squeezed.item()]
                    )  # Make it 1D array
                else:
                    inventory_final = inventory_squeezed

                # Ensure it's still a numpy array after potential scalar conversion
                if not isinstance(inventory_final, np.ndarray):
                    print(
                        f"Error Step {num_steps}: Inventory became non-array after squeeze/scalar check. Type: {type(inventory_final)}"
                    )
                    num_steps += 1
                    done = dones[0]
                    continue

                # Append the *squeezed* array copy
                inventory_callback.inventory_histories.append(inventory_final.copy())

                # Log stockouts using the squeezed array
                for loc, inv_scalar in enumerate(inventory_final):
                    # inv_scalar should now be a number
                    if inv_scalar <= 0:
                        inventory_callback.stockout_events.append(
                            {"location": loc, "step": num_steps}
                        )
                # --- End SQUEEZE and Logging ---

            # --- Keep Exception Handling as before ---
            except AttributeError as ae:
                print("--- AttributeError During Logging ---")
                print(
                    f"Step: {num_steps}\nError: {ae}\neval_env type: {type(eval_env)}"
                )
                break
            except Exception as e:
                print("--- Unexpected Error During Logging ---")
                print(
                    f"Step: {num_steps}\nException Type: {type(e)}\nArgs: {e.args}\nRepr: {repr(e)}"
                )
                try:
                    print(f"Normalized obs dict: {obs}")
                except Exception as inner_e:
                    print(f"Could not print normalized obs dict: {inner_e}")
                try:
                    print(f"Unnormalized dict: {unnormalized_obs_dict}")
                except NameError:
                    print("unnormalized_obs_dict not assigned.")
                except Exception as inner_e:
                    print(f"Could not print unnormalized_obs_dict: {inner_e}")
                print("--- End Unexpected Error Details ---")
                num_steps += 1
                done = dones[0]
                continue

            # --- CORRECTED DONE CHECK ---
            done = dones[0]
            # --- END CORRECTION ---

            # Increment step counter only if logging didn't 'continue'
            num_steps += 1
            if num_steps % 10 == 0:
                print(f"Step {num_steps}, done flag: {done}")
                if hasattr(raw_env, "_elapsed_steps"):
                    print(
                        f"Environment internal step counter: {raw_env._elapsed_steps}"
                    )

        if num_steps >= max_eval_steps and not done:
            print(
                f"Warning: Evaluation reached max steps ({max_eval_steps}) before environment signaled done."
            )
        else:
            print(f"Detailed evaluation finished after {num_steps} steps.")

        # --- PLOTTING AND SAVING LOGS ---
        # Check if any inventory history was actually logged
        if inventory_callback.inventory_histories:
            # Plot inventory levels using the collected data
            plot_inventory_levels(
                inventory_callback.inventory_histories,
                inventory_callback.stockout_events,
                n_locations,
                episode_num,  # Use training iteration number
                total_timesteps,
            )
            print(
                f"Inventory plot saved for iteration {episode_num}, total timesteps {total_timesteps}."
            )
            # Note: CSV saving is now handled inside plot_inventory_levels
            csv_filename = f"./ppo_csv_data/inventory_episode_{episode_num}_timesteps_{total_timesteps}.csv"
            print(f"Detailed inventory log saved to {csv_filename}")
        else:
            print(
                "Warning: No inventory history was logged during detailed evaluation. Skipping plot/CSV generation."
            )
        # --- END PLOTTING AND SAVING ---

        elapsed_time = (time.time() - start_time) / 60  # in minutes
        print(
            f"\nCompleted training and evaluation iteration {episode_num} ({total_timesteps} total timesteps)"
        )
        print(f"Time elapsed: {elapsed_time:.2f} minutes")
        if total_timesteps < max_timesteps:
            remaining_iterations = (max_timesteps - total_timesteps) / increment
            # Avoid division by zero if first iteration
            time_per_iteration = (
                elapsed_time / episode_num if episode_num > 0 else elapsed_time
            )
            estimated_remaining_time = remaining_iterations * time_per_iteration
            print(f"Estimated time remaining: {estimated_remaining_time:.2f} minutes")
        print(f"{'-' * 50}\n")


if __name__ == "__main__":
    main()
