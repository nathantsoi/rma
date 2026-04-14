"""Script to convert trained policies for deployment."""
from __future__ import annotations

from rma_utils.wandb_utils import load_wandb_policy

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert a checkpoint of an RL agent from RSL-RL to onnx for deployment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--estimator", action="store_true", default=False, help="Load estimator during conversion.")
parser.add_argument("--symmetric", action="store_true", default=False, help="Enforce value symmetry during training.")
parser.add_argument("--onnx", action="store_true", default=False, help="Export the policy as ONNX")
parser.add_argument("--jit", action="store_true", default=False, help="Export the policy as JIT.")
parser.add_argument("--alg", type=str, default="rma", help="Algorithm to use for training.")

parser.add_argument(
    "--wandb", 
    action="store_true", 
    default=False, 
    help="Resume with model from WandB."
)
parser.add_argument(
    "--wandb_run", 
    type=str, default="", 
    help="Run from WandB."
)
parser.add_argument(
    "--wandb_model", 
    type=str, 
    default="", 
    help="Model from WandB."
)
parser.add_argument("--actor_model", type=str, default=None, required=True, help="Path to actor model checkpoint.")


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.num_envs = 1
args_cli.video = False
args_cli.device = "cpu"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import traceback
from datetime import datetime
import yaml

import carb
import gymnasium as gym
from rma_utils.wandb_utils import pull_policy_from_wandb
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper
)
# from rsl_rl.runners import OnPolicyRunner
# from rsl_rl.runners import TactileOnPolicyRunner, VisionOnPolicyRunner
from rma_utils.exports import export_policy_as_onnx, export_policy_as_jit

from rma_tasks.rma.runners import DistillationRunner

# Add imports for the Runner classes (assuming they exist in rma_tasks.rma.runners or similar)
from rma_tasks.rma.runners import BasePolicyRunner  # Add this if not already imported

# Define Runner with a default fallback to avoid NameError
if args_cli.task == "RMA-Spot-v0":
    Runner = BasePolicyRunner
elif args_cli.task == "RMA2-Spot-v0":
    Runner = DistillationRunner
elif args_cli.task == "VRMA-Spot-v0":
    Runner = VrmaPolicyRunner
else:
    # Default fallback; adjust as needed
    Runner = BasePolicyRunner
    print("\033[93m[WARN] Unknown task; defaulting to BasePolicyRunner\033[0m")

import rma_tasks  # noqa: F401
import isaaclab_tasks

import torch

def pull_models_only():
    """Pull model files and configs from WandB without conversion."""
    
    # specify directory for downloaded models
    export_root_path = os.path.join("logs", "rma", "downloaded_models")
    export_root_path = os.path.abspath(export_root_path)
    print(f"[INFO] Downloading models to directory: {export_root_path}")
    # specify directory for this download session: {time-stamp}
    export_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir = os.path.join(export_root_path, export_dir)
    
    run_path = input(
        "Enter the weights and biases run path located on the Overview panel; i.e"
        " usr/Spot-Blind/abc123\n"
    )
    
    downloaded_count = 0
    while True:
        model_name = input(
            "\nEnter the name of the model file to download one at a time; i.e model_100.pt \n"
            + "Press Enter without a file name to finish.\n"
        )
        if model_name == "":
            break
        try:
            resume_path, env_cfg = pull_policy_from_wandb(export_dir, run_path, model_name)
            downloaded_count += 1
            print(f"\033[92m[INFO] Successfully downloaded {model_name} to {os.path.dirname(resume_path)}\033[0m")
            
            # Save env_cfg if it exists
            if env_cfg is not None:
                model_file_dir = os.path.dirname(resume_path)
                cfg_save_path = os.path.join(model_file_dir, "env_cfg.json")
                with open(cfg_save_path, "w") as fp:
                    json.dump(env_cfg, fp, indent=4)
                print(f"\033[92m[INFO] Saved env_cfg.json\033[0m")
        except Exception as e:
            print(
                f"\033[93m[WARN] Unable to download from Weights and Biases: {str(e)}\033[0m"
            )
    
    if downloaded_count > 0:
        print(f"\033[92m[INFO] Downloaded {downloaded_count} model(s) to {export_dir}\033[0m")
    else:
        print("\033[93m[INFO] No models were downloaded\033[0m")


def convert_policy():
    """Convert pytorch policies to onnx at different checkpoints."""
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # specify directory for logging experiments
    export_root_path = os.path.join("logs", "rma", agent_cfg.experiment_name)
    export_root_path = os.path.abspath(export_root_path)
    print(f"[INFO] Logging experiment in directory: {export_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    export_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        export_dir += f"_{agent_cfg.run_name}"
    export_dir = os.path.join(export_root_path, "exported", export_dir)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load the policy
    resume_path = ""
    env_cfg = None

    # load configuration
    policy_list = []
    while True:
        policy_load_type = input(
            "Select a policy loading option to add to the conversion queue: \n"
            + "(1) to convert using an explicit path to policy .pt file\n"
            + "(2) to convert policy from Weights and Biases\n"
        )
        # catch invalid input string
        if len(policy_load_type) > 1:
            continue
        match policy_load_type:
            case "1":
                while True:
                    policy_path = input(
                        "\nEnter the path of the policy model file you wish to convert one at a time\n"
                        + "Press Enter without a path to finish and convert all added policies.\n"
                    )
                    if policy_path == "":
                        break
                    resume_path = os.path.abspath(policy_path)
                    if os.path.exists(resume_path):
                        env_cfg = cli_args.load_local_cfg(resume_path)
                        policy_list.append(tuple([resume_path, env_cfg]))
                        print(f"\033[92m[INFO] added policy to conversion queue of length {len(policy_list)}\033[0m")
                    else:
                        print(
                            "\033[93m[WARN] Got invalid file path, unable to add selected file to conversion"
                            " queue!\033[0m"
                        )
                break
            case "2":
                run_path = input(
                    "Enter the weights and biases run path located on the Overview panel; i.e"
                    " usr/Spot-Blind/abc123\n"
                )
                while True:
                    model_name = input(
                        "\nEnter the name of the model file to download one at a time; i.e model_100.pt \n"
                        + "Press Enter again without a file name to finish and convert all policies in queue.\n"
                    )
                    if model_name == "":
                        break
                    try:
                        resume_path, env_cfg = pull_policy_from_wandb(export_dir, run_path, model_name)
                        
                        # Save env_cfg as YAML (consistent with WandB mode)
                        if env_cfg is not None:
                            model_file_dir = os.path.dirname(resume_path)
                            cfg_save_path = os.path.join(model_file_dir, "env_cfg.yaml")
                            with open(cfg_save_path, "w") as fp:
                                yaml.dump(env_cfg, fp, default_flow_style=False, indent=2)
                            print(f"\033[92m[INFO] Saved env_cfg.yaml\033[0m")

                        policy_list.append(tuple([resume_path, env_cfg]))
                        print(f"\033[92m[INFO] added policy to conversion queue of length {len(policy_list)}\033[0m")
                    except Exception:
                        print(
                            "\033[93m[WARN] Unable to download from Weights and Biases for conversion, is the path"
                            " and filename correct?\033[0m"
                        )
                break

    for idx in range(len(policy_list)):
        resume_path, env_cfg = policy_list[idx]
        model_file_dir = os.path.dirname(resume_path)
        model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # create runner from rsl-rl
        #ppo_runner = VisionOnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        #ppo_runner = TactileOnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

        teacher_path = None  # Initialize to avoid NameError
        if args_cli.actor_model == "wandb":
            # load configuration
            run_path = args_cli.wandb_run
            model_name = args_cli.wandb_model
            teacher_path, _ = load_wandb_policy(run_path, model_name, log_root_path)
            agent_cfg.teacher.checkpoint_path = teacher_path

        ppo_runner = Runner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        if teacher_path is not None:  # Only call if teacher_path is set
            ppo_runner.load_baseActor_policy(teacher_path)

        #obs = torch.zeros(1, 51, device=agent_cfg.device)
        #image_obs = torch.zeros(1, 2, 53, 30, device=agent_cfg.device)
        #print("actions: ",ppo_runner.alg.actor_critic(obs, image_obs))
        #print("actions: ",ppo_runner.alg.actor_critic.act_inference(obs))
        ppo_runner.load(resume_path, load_optimizer=False)

        export_model_dir = os.path.join(model_file_dir, f"{model_file_name}_deployment")
        os.makedirs(export_model_dir, exist_ok=True)  # Added exist_ok to avoid errors if dir exists
        print(f"[INFO]: Saving env config json file to {export_model_dir}")
        cfg_save_path = os.path.join(export_model_dir, "env_cfg.json")
        if env_cfg is not None:  # Ensure env_cfg exists before saving
            with open(cfg_save_path, "w") as fp:
                json.dump(env_cfg, fp, indent=4)
        if args_cli.onnx:
            print(f"[INFO]: Saving policy onnx file to {export_model_dir}")
            export_policy_as_onnx(ppo_runner.alg.policy, export_model_dir, filename=f"{model_file_name}.onnx")
        elif args.cli.jit:
            print(f"[INFO]: Saving policy jit file to {export_model_dir}")
            export_policy_as_jit(ppo_runner.alg.policy, normalizer=None, path=export_model_dir, filename=f"{model_file_name}.jit", alg=args.cli.alg)
    if len(policy_list) > 0:
        print(f"\033[92m[INFO] Exported {len(policy_list)} policy(ies) to {export_dir}\033[0m")


if __name__ == "__main__":
    # Select operation mode
    mode = input(
        "Select operation mode:\n"
        + "(1) Convert policies to ONNX/JIT\n"
        + "(2) Pull model files from WandB only (no conversion)\n"
    )
    
    if mode == "1":
        # run the main execution
        convert_policy()
    elif mode == "2":
        # just pull models without conversion
        pull_models_only()
    else:
        print("[ERROR] Invalid selection. Exiting.")
    
    # close sim app
    simulation_app.close()
