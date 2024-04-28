import unittest, json, submitted, utils
import numpy as np
import torch
import itertools

from gradescope_utils.autograder_utils.decorators import weight

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        with open('tests/solutions_visible.json') as f:
            self.solution = json.load(f)

    @weight(5)
    def test_ppo_loss(self):        
        sol_this = self.solution['ts_ppo']
        obs_space_sz = sol_this['obs_sz']
        act_space_sz = sol_this['act_sz']
        batch_sz     = sol_this['batch_sz']
        obs          = torch.tensor(sol_this['obs'])
        action       = torch.tensor(sol_this['action'])
        old_logits   = torch.tensor(sol_this['old_logits'])
        return_      = torch.tensor(sol_this['return_'])
        fixed_init   = sol_this['fixed_init']
        ppo_loss_ref = sol_this['ppo_loss']

        policy = utils.SimpleReLuNetwork(obs_space_sz, act_space_sz,
                                        out_logsoftmax=True, fixed_init=fixed_init)
        ppo_loss = submitted.get_PPO_policy_gradient_loss(
            policy=policy,
            observation         = obs,
            old_logits          = old_logits,
            action              = action,
            return_or_advantage = return_,
            ppo_clip            = 0.1
        )
        self.assertEqual(type(ppo_loss), torch.Tensor, msg=f"Output of get_PPO_policy_gradient_loss should be a torch tensor, got {type(ppo_loss)}")
        self.assertEqual(ppo_loss.shape, torch.Size([]), msg=f"Output of get_PPO_policy_gradient_loss should be singleton tensor, got shape {ppo_loss.shape}")
        self.assertAlmostEqual(ppo_loss.item(),
                               ppo_loss_ref, places=4, msg=f"Output of get_PPO_policy_gradient_loss differend more than allowed from reference")

    
    @weight(5)
    def test_training_loop_ppo(self):
        sol_this = self.solution['ts_training_loop_ppo']
        env_target = sol_this['env_target']
        env_start =  sol_this['env_start']
        rollout_seed = np.array(sol_this['rollout_seed'])
        actor_state_dict        = {k:torch.tensor(v) for k, v in sol_this['actor_init'].items()}
        actor_state_dict_after  = {k:torch.tensor(v) for k, v in sol_this['actor_after'].items()}
        critic_state_dict       = {k:torch.tensor(v) for k, v in sol_this['critic_init'].items()}
        critic_state_dict_after = {k:torch.tensor(v) for k, v in sol_this['critic_after'].items()}
        losses_actor_ref = torch.tensor(sol_this['losses_actor'])
        losses_critic_ref = torch.tensor(sol_this['losses_critic'])
        final_rewards_ref = sol_this['final_rewards']

        env = utils.GridWorldPointTargetEnv(grid_size=10, dimensions=1,
                                            set_target=env_target, set_state=env_start,
                                            episode_length=4)
        actor = utils.SimpleReLuNetwork(2, 3, hidden_dims=[16], out_logsoftmax=True)
        actor.load_state_dict(actor_state_dict)
        critic = utils.SimpleReLuNetwork(2, 1, hidden_dims=[16])
        critic.load_state_dict(critic_state_dict)
        optimizer = torch.optim.SGD(itertools.chain(actor.parameters(), critic.parameters()), lr=1e-3)

        losses_actor, losses_critic, final_rewards, lr = submitted.train_policy_gradient(
            env=env,
            policy=actor, optimizer=optimizer,
            get_policy_gradient_loss=submitted.get_PPO_policy_gradient_loss,
            get_returns=submitted.get_returns,
            value_net=critic,
            get_advantages = submitted.get_advantages,
            get_value_net_loss = submitted.get_value_net_loss,
            critic_loss_multiplier = 1.0,
            rollouts=1,
            rollouts_before_training=1,
            training_epochs_per_rollout=1,
            minibatch_size=4,
            rollout_seed=rollout_seed
        )

        self.assertAlmostEqual(losses_actor[0].item(), losses_actor_ref.item(), places=4,
                                msg=f"Loss returned from training loop for actor differed more than allowed from reference")
        self.assertAlmostEqual(losses_critic[0].item(), losses_critic_ref.item(), places=4,
                                msg=f"Loss returned from training loop for critic differed more than allowed from reference")

        for k, v in actor.state_dict().items():
            ref = actor_state_dict_after[k]
            self.assertLessEqual(torch.max(torch.abs(v - ref)).item(), 0.01, 
                                 msg=f"After executing training step, found difference in actor weights more than allowed 0.01")
        for k, v in critic.state_dict().items():
            ref = critic_state_dict_after[k]
            self.assertLessEqual(torch.max(torch.abs(v - ref)).item(), 0.01, 
                                 msg=f"After executing training step, found difference in critic weights more than allowed 0.01")
