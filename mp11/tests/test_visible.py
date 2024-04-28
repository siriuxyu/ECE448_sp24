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
            
    @weight(9)
    def test_returns_calculation(self):        
        sol_this = self.solution['ts_returns']
        ROLLOUT_N = sol_this['rollout_n']
        REWARD_N = sol_this['reward_n']
        rewards = sol_this['rewards']
        returns_ref = sol_this['returns']

        rollout_buffer = utils.RolloutBuffer()
        dummy = torch.tensor([0])
        for reward_this in rewards:
            for i in reward_this[:-1]:
                rollout_buffer.add(action=dummy, logits=dummy, observation=dummy, terminated=False, reward=i)
            rollout_buffer.add(action=dummy, logits=dummy, observation=dummy, terminated=True, reward=reward_this[-1])
        rollout_buffer.finalize()
        returns_submitted = submitted.get_returns(rollout_buffer, discount_factor=0.5)
        self.assertEqual(type(returns_submitted), torch.Tensor, msg=f"Output of get_returns should be a torch tensor, got {type(returns_submitted)}")
        self.assertEqual(returns_submitted.shape, torch.Size([500,1]), msg=f"Output of get_returns should be shape batch x 1, got {returns_submitted.shape}")
        for i, (a, b) in enumerate(zip(returns_ref, returns_submitted.tolist())):
            self.assertAlmostEqual(a[0], b[0], places=4, msg=f"Output of get_returns not within 4 places of reference at index {i}")

    @weight(9)
    def test_vanilla_gradient_descent_loss(self):        
        sol_this = self.solution['ts_vpg']
        obs_space_sz = sol_this['obs_sz']
        act_space_sz = sol_this['act_sz']
        batch_sz     = sol_this['batch_sz']
        obs          = torch.tensor(sol_this['obs'])
        action       = torch.tensor(sol_this['action'])
        return_      = torch.tensor(sol_this['return_'])
        fixed_init   = sol_this['fixed_init']
        vpg_loss_ref = sol_this['vpg_loss']

        policy = utils.SimpleReLuNetwork(obs_space_sz, act_space_sz,
                                        out_logsoftmax=True, fixed_init=fixed_init)
        vpg_loss_submitted = submitted.get_vanilla_policy_gradient_loss(
            policy              = policy,
            observation         = obs,
            action              = action,
            return_or_advantage = return_,
        )
        self.assertEqual(type(vpg_loss_submitted), torch.Tensor, msg=f"Output of get_returns should be a torch tensor, got {type(vpg_loss_submitted)}")
        self.assertEqual(vpg_loss_submitted.shape, torch.Size([]), msg=f"Output of get_returns should be singleton tensor, got shape {vpg_loss_submitted.shape}")
        self.assertAlmostEqual(vpg_loss_submitted.item(),
                               vpg_loss_ref, places=4, msg=f"Output of get_vanilla_policy_gradient_loss differend more than allowed from reference")
    
    @weight(8)
    def test_rollout_collection(self):
        sol_this = self.solution['ts_rollout_collection']
        rollout_seed = np.array(sol_this['rollout_seed'])
        actor_state_dict = {k:torch.tensor(v) for k, v in sol_this['actor'].items()}
        rbref = utils.RolloutBuffer().load(sol_this['rollout_buffer'])
        final_reward_mean_ref = sol_this['final_reward_mean']

        env = utils.GridWorldPointTargetEnv(grid_size=100, dimensions=2,
                                            set_target=np.array([1, 1]),
                                            set_state=np.array([50, 50]),
                                            episode_length=100)
        actor = utils.SimpleReLuNetwork(4, 5, hidden_dims=[16], out_logsoftmax=True)
        actor.load_state_dict(actor_state_dict)
        rollout_buffer, final_reward_mean = submitted.collect_rollouts(
            env=env,
            policy=actor,
            num_rollouts=10,
            seed=rollout_seed
        )
        self.assertEqual(rollout_buffer.final_size, rbref.final_size, msg=f"Rollout buffer should have size {rbref.final_size}, has size {rollout_buffer.final_size} instead")
        self.assertEqual(type(rollout_buffer), utils.RolloutBuffer, msg=f"Output of collect_rollouts should be a utils.RolloutBuffer, got {type(rollout_buffer)}")
        for rb_submitted, ref, name in zip(
            [rollout_buffer.actions, rollout_buffer.observations, rollout_buffer.old_logits, rollout_buffer.rewards, rollout_buffer.terminateds],
            [rbref.actions, rbref.observations, rbref.old_logits, rbref.rewards, rbref.terminateds],
            ["actions", "observations", "old_logits", "rewards", "terminateds"]
        ):
            self.assertEqual(rb_submitted.shape, ref.shape, msg=f"Rollout buffer field {name} should have shape {ref.shape}, has shape {rb_submitted.shape} instead")
            if name != "terminateds":
                self.assertLessEqual(torch.max(torch.abs(rb_submitted - ref)).item(), 0.01, 
                                     msg=f"Rollout buffer field {name} max difference from reference more than allowed 0.01")
            else:
                self.assertTrue(torch.all(torch.logical_not(torch.logical_xor(rb_submitted, ref))).item(), 
                                     msg=f"Rollout buffer field {name} different from reference")
            if name == "old_logits":
                self.assertFalse(rb_submitted[0].requires_grad, f"Logits placed in the rollout buffer should not have attached gradients!")

    @weight(9)
    def test_training_loop(self):
        sol_this = self.solution['ts_training_loop']
        env_target = sol_this['env_target']
        env_start =  sol_this['env_start']
        rollout_seed = np.array(sol_this['rollout_seed'])
        actor_state_dict = {k:torch.tensor(v) for k, v in sol_this['actor_init'].items()}
        actor_state_dict_after = {k:torch.tensor(v) for k, v in sol_this['actor_after'].items()}
        losses_actor_ref = torch.tensor(sol_this['losses_actor'])
        final_rewards_ref = sol_this['final_rewards']

        env = utils.GridWorldPointTargetEnv(grid_size=10, dimensions=1,
                                            set_target=env_target, set_state=env_start,
                                            episode_length=4)
        actor = utils.SimpleReLuNetwork(2, 3, hidden_dims=[16], out_logsoftmax=True)
        actor.load_state_dict(actor_state_dict)
        optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)

        losses_actor, _, final_rewards, lr = submitted.train_policy_gradient(
            env=env,
            policy=actor, optimizer=optimizer,
            get_policy_gradient_loss=submitted.get_vanilla_policy_gradient_loss,
            get_returns=submitted.get_returns,
            critic_loss_multiplier = 1.0,
            rollouts=1,
            rollouts_before_training=1,
            training_epochs_per_rollout=1,
            minibatch_size=4,
            rollout_seed=rollout_seed
        )

        self.assertAlmostEqual(losses_actor[0].item(), losses_actor_ref.item(), places=4,
                                msg=f"Loss returned from training loop differed more than allowed from reference")

        for k, v in actor.state_dict().items():
            ref = actor_state_dict_after[k]
            self.assertLessEqual(torch.max(torch.abs(v - ref)).item(), 0.01, 
                                 msg=f"After executing training step, found difference in weights more than allowed 0.01")

    @weight(2)
    def test_value_network_loss(self):
        sol_this = self.solution['ts_value_net_loss']
        obs_space_sz = sol_this['obs_sz']
        batch_sz = sol_this['batch_sz']
        obs = torch.tensor(sol_this['obs'])
        return_ = torch.tensor(sol_this['return_'])
        fixed_init = sol_this['fixed_init']
        value_net_loss_ref = sol_this['value_net_loss']

        critic = utils.SimpleReLuNetwork(obs_space_sz, 1, hidden_dims=[16], fixed_init=fixed_init, out_logsoftmax=False)
        value_net_loss = submitted.get_value_net_loss(
            critic, obs, return_
        )
        self.assertEqual(type(value_net_loss), torch.Tensor, msg=f"Output of get_value_net_loss should be a torch tensor, got {type(value_net_loss)}")
        self.assertEqual(value_net_loss.shape, torch.Size([]), msg=f"Output of get_value_net_loss should be singleton tensor, got shape {value_net_loss.shape}")
        self.assertAlmostEqual(value_net_loss.item(),
                               value_net_loss_ref, places=4, msg=f"Output of get_value_net_loss not within 4 places of reference")
    
    @weight(8)
    def test_advantages_calculation(self):
        sol_this = self.solution['ts_generate_advantages']
        obs_space_sz = sol_this['obs_sz']
        batch_sz = sol_this['batch_sz']
        fixed_init = sol_this['fixed_init']
        critic = utils.SimpleReLuNetwork(obs_space_sz, 1, hidden_dims=[16], fixed_init=fixed_init, out_logsoftmax=False)
        obs = torch.tensor(sol_this['obs'])
        return_ = torch.tensor(sol_this['return_'])
        advantages = submitted.get_advantages(
            critic, obs, return_
        )
        advantages_ref = torch.tensor(sol_this['advantages'])

        self.assertEqual(type(advantages), torch.Tensor, msg=f"Output of get_advantages should be a torch tensor, got {type(advantages)}")
        self.assertEqual(advantages.shape, advantages_ref.shape, msg=f"Output of get_advantages should be batch x 1, got shape {advantages.shape}")
        self.assertLessEqual(torch.max(torch.abs(advantages - advantages_ref)).item(), 0.01, 
                                msg=f"Output from get_advantages max difference from reference more than allowed 0.01")
    
    @weight(5)
    def test_training_loop_advantage_estimation(self):
        sol_this = self.solution['ts_training_loop_ae']
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
            get_policy_gradient_loss=submitted.get_vanilla_policy_gradient_loss,
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