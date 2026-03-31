from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDistillationAlgorithmCfg
)


from rma_tasks.rma.wrappers import BasePolicyCfg

@configclass
class Rma1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "spot_rma"
    empirical_normalization = False
    store_code_state = False
    logger = "wandb"
    wandb_project = "Vision_RMA"
    obs_group = {"policy" : ["policy"], "priv_obs" : ["priv_obs"]}
    policy = BasePolicyCfg(
        class_name="BasePolicy",
        prev_step_size=48,
        z_size=8, # Size of embedding space for priv obs 
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class Rma2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "spot_rma"
    empirical_normalization = False
    store_code_state = False
    class_name = "DistillationRunner"
    logger = "wandb"
    wandb_project = "Vision_RMA"
    obs_group = {"policy" : ["policy"], "priv_obs" : ["priv_obs"], "history" : ["history"], "critic" : ["policy"]}
    teacher = BasePolicyCfg(
        class_name="BasePolicy",
        z_size=8, # Size of embedding space for priv obs 
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
    )
    policy = BasePolicyCfg(
        class_name="AdaptationModule",
        z_size=8, # Size of embedding space for priv obs 
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
