# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer

from rma_tasks.rma.modules import BasePolicy, AdaptationModule
import torch.nn.functional as F  # Put this at the very top of distillation.py if it isn't there

# teacher is really just the trained base policy + encoder being used to generate supervision for the adaptation module.
class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: AdaptationModule
    teacher: BasePolicy
    """The student teacher model."""

    def __init__(
        self,
        policy,
        teacher,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.teacher = teacher
        self.teacher.to(self.device)
        
        # Freeze teacher weights - teacher should not be updated during distillation
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # initialize the optimizer for the encoder
        self.optimizer = resolve_optimizer(optimizer)(self.policy.encoder.parameters(), lr=learning_rate)

        # initialize the transition
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self.policy.loaded_teacher = False

        # initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    # def act(self, obs):
    #     with torch.no_grad():
    #         teacher_latents = self.teacher.get_latents(obs).detach()
    #         teacher_action = self.teacher.act_inference(obs).detach()

    #         # Student latent (we can detach for storage; during update we will recompute with grad)
    #         student_latents = self.policy.get_latents(obs).detach()

    #         self.transition.latents = self.policy.get_latents(obs) # uses history
    #         privileged_action = self.policy.teacher.actor(actor_input)

    #     # # compute the actions
            
    #     # self.transition.actions = self.policy.act(self.transition.latents + obs).detach()
    #     # self.transition.teacher_latents = self.teacher.get_latents(obs).detach() # uses priv_obs
    #     # self.transition.observations = obs
    #     # return self.transition.actions

    #     # store in transition; don't mutate obs or try to add tensors to the tensordict
    #     self.transition.teacher_latents = teacher_latents
    #     self.transition.student_latents = student_latents
    #     self.transition.observations = obs
    #     # If you need the teacher action saved too (optional):
    #     self.transition.actions = teacher_action

    #     # return the action that will be executed in the env (teacher's deterministic action)
    #     return teacher_action

    def act(self, obs):
        actions = self.policy.act(obs)
        self.transition.actions = actions.detach()
        
        with torch.no_grad():
            self.transition.latents = self.policy.get_latents(obs).detach()
            
            priv_obs = self.teacher.get_encoder_obs(obs)
            actor_obs = self.teacher.get_actor_obs(obs)
            priv_input = self.teacher.encoder_obs_normalizer(priv_obs)
            teacher_latent = self.teacher.encoder(priv_input).detach()
            # print("Teacher Latent Sum:", teacher_latent[0].abs().sum().item())
            self.transition.teacher_latents = teacher_latent
            
            actor_input = torch.cat([actor_obs, teacher_latent], dim=-1)
            privileged_action = self.teacher.actor(actor_input)
            self.transition.privileged_actions = privileged_action.detach()
            
            # ==========================================
            # DIAGNOSTIC PRINTS - ADD THESE HERE
            # ==========================================
            # if self.transition.observations is None: # Just print once per step
            #     print("--- TEACHER VITALS ---")
            #     print("1. Actor Obs Sum: ", actor_obs[0].abs().sum().item())
            #     print("2. Priv Obs Sum:  ", priv_obs[0].abs().sum().item())
            #     print("3. Teacher Action:", privileged_action[0].abs().sum().item())
            #     print("----------------------")
            
        self.transition.observations = obs
        
        # Remember to keep returning the teacher's action!
        return self.transition.privileged_actions


    
    def process_env_step(self, obs, rewards, dones, extras):
        # update the normalizers
        self.policy.update_normalization(obs)

        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones

        # update latents in step
        self.transition.student_latents = self.policy.get_latents(obs).detach()
        self.transition.teacher_latents = self.teacher.get_latents(obs).detach()

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)


    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            for obs, _, privileged_actions, dones in self.storage.generator():

                # ==========================================
                # THE FIX: Calculate Teacher latents the exact 
                # same way we did in the act() function
                # ==========================================
                with torch.no_grad():
                    priv_obs = self.teacher.get_encoder_obs(obs)
                    # Apply the normalizer so the network isn't blinded!
                    if hasattr(self.teacher, "encoder_obs_normalizer") and self.teacher.encoder_obs_normalizer is not None:
                        priv_input = self.teacher.encoder_obs_normalizer(priv_obs)
                    else:
                        priv_input = priv_obs
                        
                    # Calculate the true teacher latent
                    teacher_latents = self.teacher.encoder(priv_input).detach()
                
                # Get student latents (from history observations) - gradients needed for training
                student_latents = self.policy.get_latents(obs) 

                # behavior cloning loss (MSE)
                # behavior_loss = self.loss_fn(student_latents, teacher_latents)
                # ==========================================
                # THE FIX: BOMB-PROOF THE LOSS
                # ==========================================
                # 1. Clamp the teacher's target just in case the buffer fed it garbage
                safe_teacher_latents = torch.clamp(teacher_latents, min=-20.0, max=20.0)
                
                # 2. Use Smooth L1 (Huber) Loss so Batch 1 doesn't explode the network
                behavior_loss = F.smooth_l1_loss(student_latents, safe_teacher_latents)

                # ==========================================
                # UPDATE VITALS DIAGNOSTIC
                # ==========================================
                if cnt == 0 and epoch == 0:
                    print("--- UPDATE VITALS ---")
                    print("1. Buffer Priv Obs Sum:     ", priv_obs[0].abs().sum().item())
                    print("2. Buffer Teacher Latent Sum:", teacher_latents[0].abs().sum().item())
                    print("3. Buffer Student Latent Sum:", student_latents[0].abs().sum().item())
                    print("4. Starting MSE Loss:       ", behavior_loss.item())
                    print("---------------------")

                # total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # gradient step
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.policy.encoder.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                loss = 0

        mean_behavior_loss /= cnt
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict
    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
        