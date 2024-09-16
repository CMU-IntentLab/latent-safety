import copy
import torch
import torch.optim
from torch import nn
from termcolor import cprint
from functools import partial
import numpy as np
import torch.nn.functional as F

import dreamer.networks as networks
import dreamer.tools as tools
from tqdm import trange
from torch.nn.utils import spectral_norm

def to_np(x):
    return x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        ''' TEMPORARY '''
        shapes.pop('obs_state')
        shapes.pop('state')
        self._init_model(shapes, config)
        self._init_heads(shapes, config)
        self._init_optims(config)
        self.obs_step = config.obs_step

        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        self.always_frozen_layers = []
        self.freeze_encoder = config.freeze_encoder
        if self.freeze_encoder:
            cprint(
                "Freezing embeddings from encoder during training",
                color="red",
                attrs=["bold"],
            )
            self.always_frozen_layers = [
                name
                for name, _ in self.named_parameters()
                if "encoder." in name or "_obs" in name
            ]

    def _init_model(self, shapes, config):
        self.encoder = networks.MultiEncoder(
            shapes, augment_images=config.augment_images, **config.encoder
        )
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )

    def _init_heads(self, shapes, config):
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        reward_base_kwargs = dict(
            inp_dim=feat_size,
            shape=(255,) if config.reward_head["dist"] == "symlog_disc" else (),
            layers=config.reward_head["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        cont_base_kwargs = dict(
            inp_dim=feat_size,
            shape=(),
            layers=config.cont_head["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        self.heads["reward"] = tools.create_single_or_ensemble(
            config.reward_ensemble_size,
            config.reward_ensemble_subsample,
            base_kwargs=reward_base_kwargs,
            name="reward",
        )
        self.heads["cont"] = tools.create_single_or_ensemble(
            config.cont_ensemble_size,
            config.cont_ensemble_subsample,
            base_kwargs=cont_base_kwargs,
            name="cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name

    def _init_obs_mlp(self, config, obs_shape):
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        obs_mlp = nn.Sequential(
            nn.Linear(feat_size, config.units),
            nn.ReLU(),
            nn.Linear(config.units, obs_shape),
        )
        """networks.MLP(
            feat_size,
            (obs_shape),
            config.actor["layers"], # for now
            config.units,
            config.act,
            config.norm,
            dist=config.actor["dist"],
            outscale=config.actor["outscale"],
            device=config.device,
            name="observation",
        )"""
        obs_mlp.to(config.device)
        standard_kwargs = {
            "lr": config.obs_lr,
            "eps": config.opt_eps,
            "clip": config.grad_clip,
            "wd": config.weight_decay,
            "opt": config.opt,
            "use_amp": self._use_amp,
        }
        obs_recon_opt = tools.Optimizer(
            "obs_mlp", obs_mlp.parameters(), **standard_kwargs
        )
        return obs_mlp, obs_recon_opt
    
    def _init_lx_mlp(self, config, obs_shape):
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        lx_mlp = nn.Sequential(
            spectral_norm(nn.Linear(feat_size, 16)),
            nn.ReLU(),
            spectral_norm(nn.Linear(16, obs_shape)),
        )
        
        lx_mlp.to(config.device)
        standard_kwargs = {
            "lr": config.obs_lr,
            "eps": config.opt_eps,
            "clip": config.grad_clip,
            "wd": config.weight_decay,
            "opt": config.opt,
            "use_amp": self._use_amp,
        }
        lx_recon_opt = tools.Optimizer(
            "lx_mlp", lx_mlp.parameters(), **standard_kwargs
        )
        return lx_mlp, lx_recon_opt

    def _init_optims(self, config):
        standard_kwargs = {
            "lr": config.model_lr,
            "eps": config.opt_eps,
            "clip": config.grad_clip,
            "wd": config.weight_decay,
            "opt": config.opt,
            "use_amp": self._use_amp,
            "lr_decay": config.decay_model_lr,
        }
        self.decay_model_lr = config.decay_model_lr
        self._model_opt = tools.Optimizer("model", self.parameters(), **standard_kwargs)
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

    def _train(self, data, frozen_heads=[]):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)
        with tools.RequiresGrad(self, always_frozen_layers=self.always_frozen_layers):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                losses_sum, losses_dict = self._compute_head_losses(
                    data, embed, post, frozen_heads
                )
                model_loss = losses_sum + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        if self.decay_model_lr:
            self._model_opt.step()

        metrics["model_lr"] = self._model_opt.get_lr()
        metrics.update(
            {f"{name}_loss": to_np(loss) for name, loss in losses_dict.items()}
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def _train_reward_only(self, data):
        data = self.preprocess(data)
        relevant_params = list(self.heads["reward"].parameters()) + list(
            self.heads["cont"].parameters()
        )
        with tools.RequiresGrad(self.heads["cont"]), tools.RequiresGrad(
            self.heads["reward"]
        ):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, _ = self.dynamics.observe(embed, data["action"], data["is_first"])
                losses_sum, losses_dict = self._compute_head_losses(
                    data, embed, post, frozen_heads=["decoder"]
                )
            metrics = self._model_opt(torch.mean(losses_sum), relevant_params)
        metrics.update(
            {f"{name}_loss": to_np(loss) for name, loss in losses_dict.items()}
        )
        return metrics

    def _compute_head_losses(self, data, embed, post, frozen_heads=[]):
        preds = {}
        for name, head in self.heads.items():
            if name in frozen_heads:
                # skip ["cont", "reward"]
                continue
            grad_head = name in self._config.grad_heads
            feat = self.dynamics.get_feat(post)
            feat = feat if grad_head else feat.detach()
            pred = head(feat)
            if type(pred) is dict:
                preds.update(pred)
            else:
                preds[name] = pred
        losses = {}
        for name, pred in preds.items():
            loss = -pred.log_prob(data[name])
            assert loss.shape == embed.shape[:2], (name, loss.shape)
            losses[name] = loss
        scaled = {
            key: value * self._scales.get(key, 1.0) for key, value in losses.items()
        }
        return sum(scaled.values()), losses

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        # For all keys in obs which contain "image", normalize the values by 255.0
        for key, value in obs.items():
            if "image" in key:
                obs[key] = torch.Tensor(np.array(value)) / 255.0

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(np.array(v)).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        obs_steps = self.obs_step
        

        states, _ = self.dynamics.observe(
            embed[:6, :obs_steps], data["action"][:6, :obs_steps], data["is_first"][:6, :obs_steps]
        )
        '''recon = self.heads["decoder"](self.dynamics.get_feat(states))[
            "agentview_image"
        ].mode()[:6]
        recon_hand = self.heads["decoder"](self.dynamics.get_feat(states))[
            "robot0_eye_in_hand_image"
        ].mode()[:6]'''
        recon = self.heads["decoder"](self.dynamics.get_feat(states))[
            "image"
        ].mode()[:6]

        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, obs_steps:], init)
        
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))[
            "image"
        ].mode()
        '''openl_hand = self.heads["decoder"](self.dynamics.get_feat(prior))[
            "robot0_eye_in_hand_image"
        ].mode()'''
        
        
        
        
        #model_hand = torch.cat([recon_hand[:, :5], openl_hand], 1)
        truth = data["image"][:6]

        row, col = torch.where(data['is_first'][:6, obs_steps:] == 1.)
        for i in range(row.size(0)):
            data['is_first'][row[i], obs_steps+col[i]:] = 1.
            openl[row[i], col[i]:] = openl[row[i], col[i]-1]
            truth[row[i], obs_steps+col[i]:] = truth[row[i], obs_steps+col[i]-1]
        
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :obs_steps], openl], 1)
        #truth_hand = data["robot0_eye_in_hand_image"][:6]
        error = (model - truth + 1.0) / 2.0
        #error_hand = (model_hand - truth_hand + 1.0) / 2.0

        return torch.cat([truth, model, error], 2) #torch.cat(
            #[truth_hand, model_hand, error_hand], 2
        #)

    @torch.no_grad()
    def video_pred_imagine(self, envs, exp_data, policy, time_limit):
        # TODO: work for multiple image, currently hardcoded to key "agentview_image"
        assert len(envs) > 1
        resets = [env.reset()() for env in envs]
        obs = {k: np.stack([o[k] for o in resets]) for k in resets[0]}
        obs.update({"action": np.zeros((len(envs), exp_data["action"].shape[-1]))})
        data = self.preprocess(obs)
        embed = self.encoder(data)
        embed = embed.unsqueeze(1)
        action = torch.from_numpy(obs["action"]).unsqueeze(1)
        original_actions = torch.clone(action)
        is_first = torch.from_numpy(obs["is_first"]).unsqueeze(1)
        states, _ = self.dynamics.observe(embed, action, is_first)
        gt_obs = torch.from_numpy(obs["agentview_image"]).unsqueeze(1)

        recon = None
        print(states["stoch"].size())
        num_successes = 0
        total_episodes = len(envs)
        for i in trange(time_limit, desc="Imagined rollouts in model"):
            feat = self.dynamics.get_feat(states)
            if recon is None:
                recon = self.heads["decoder"](feat)["agentview_image"].mode()[:6]
                print(recon.size())
            else:
                openl = self.heads["decoder"](self.dynamics.get_feat(states))[
                    "agentview_image"
                ].mode()
                recon = torch.cat([recon, openl], 1)
                gt_obs = torch.cat(
                    [
                        gt_obs,
                        torch.from_numpy(obs["agentview_image"]).unsqueeze(1),
                    ],
                    1,
                )
            inp = feat
            action = policy(inp).mode()
            states = self.dynamics.img_step(states, action)

            action = action.squeeze(1).cpu().numpy()
            policy_output = [{"action": a} for a in action]
            results = [e.step(a) for e, a in zip(envs, policy_output)]
            results = [r() for r in results]
            obs, reward, done = zip(*[p[:3] for p in results])
            # list of obs dicts to dict of obs list
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
            assert gt_obs.shape == recon.shape, f"{gt_obs.shape} != {recon.shape}"

            done = np.stack(done)
            reward = list(reward)
            success = np.array([r == 1 for r in reward])
            num_successes += np.stack([r == 1 for r in reward]).sum()
            done = np.logical_or(done, success)
            if done.any():
                indices = [index for index, d in enumerate(done) if d]
                resets = [envs[i].reset() for i in indices]
                resets = [r() for r in resets]
                total_episodes += len(indices)

                # also reset the world_model
                reset_obs = {k: np.stack([r[k] for r in resets]) for k in resets[0]}
                data = self.preprocess(reset_obs)
                embed = self.encoder(data)
                embed = embed.unsqueeze(1)
                is_first = torch.from_numpy(reset_obs["is_first"]).unsqueeze(1)
                reset_states, _ = self.dynamics.observe(
                    embed, original_actions[: len(done)], is_first=is_first
                )

                for key in states:
                    for j, index in enumerate(indices):
                        # reset_states is only of len(indices) <= len(envs)
                        states[key][index] = reset_states[key][j]

        truth = gt_obs / 255.0
        model = recon.cpu()
        error = (model - truth + 1.0) / 2.0

        print(f"{num_successes=}, {total_episodes=}")
        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model

        # recurrent states are deterministic, latent are stochastic
        if config.dyn_discrete:
            stoch_dims = config.dyn_stoch * config.dyn_discrete
        else:
            stoch_dims = config.dyn_stoch
        self.feat_size = stoch_dims + config.dyn_deter
        self.stoch_dim, self.deter_dim = stoch_dims, config.dyn_deter

        self.train_residuals = config.train_residuals
        self.ensemble_residuals = config.ensemble_residuals and self.train_residuals
        cprint(
            f"{config.train_residuals=}, {config.ensemble_residuals=}",
            color="red",
            attrs=["bold"],
        )

        self.bc_reg = config.bc_reg
        self.bc_reg_weight = config.bc_reg_weight
        self.bc_reg_wd = config.bc_reg_wd
        if self.bc_reg:
            assert self.bc_reg_weight > 0.0 and self.bc_reg_wd > 0.0
            cprint(
                f"Using BC regularization with weight {self.bc_reg_weight} and weight decay {self.bc_reg_wd}",
                color="cyan",
                attrs=["bold"],
            )

        self._init_actor()
        self._init_critic()
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _init_actor(self):
        config = self._config
        self.actor_policy_kwargs = dict(
            inp_dim=self.feat_size,
            shape=(config.num_actions,),
            layers=config.actor["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist=config.actor["dist"],
            std=config.actor["std"],
            min_std=config.actor["min_std"],
            max_std=config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
            masked_dims=-self.deter_dim if config.mask_recur else None,
        )
        self.actor_opt_kwargs = dict(
            lr=config.actor["lr"],
            eps=config.actor["eps"],
            clip=config.actor["grad_clip"],
            wd=config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self.actor = networks.MLP(**self.actor_policy_kwargs)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            **self.actor_opt_kwargs,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        if config.dropout_recurrent_prob > 0.0 and (
            config.pretrain_actor_steps > 0 or config.pretrain_joint_steps > 0
        ):
            cprint(
                f"Doing dropout on recurrent layer with probability {config.dropout_recurrent_prob} during pretraining",
                color="red",
                attrs=["bold"],
            )
            self.recurrent_dropout_layer = nn.Dropout(p=config.dropout_recurrent_prob)

    def _init_critic(self):
        config = self._config
        critic_kwargs = dict(
            inp_dim=self.feat_size,
            shape=(255,) if config.critic["dist"] == "symlog_disc" else (),
            layers=config.critic["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist=config.critic["dist"],
            outscale=config.critic["outscale"],
            name="Value",
        )
        self.value = tools.create_single_or_ensemble(
            config.critic_ensemble_size,
            config.critic_ensemble_subsample,
            name="critic",
            base_kwargs=critic_kwargs,
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )

    def apply_recurrent_dropout(self, feat):
        stoch, deter = torch.split(feat, [self.stoch_dim, self.deter_dim], dim=-1)
        deter = self.recurrent_dropout_layer(deter)
        return torch.cat([stoch, deter], dim=-1)

    def decay_bc_weight(self):
        self.bc_reg_weight *= self._config.bc_reg_wd
        return self.bc_reg_weight

    def reset_critics(self):
        self.value.reset_last_three_layers()

    def replace_actor_with_ensemble(self, device="cuda", copy_base_actor_weights=True):
        max_ensemble_size = (
            1
            if not self.ensemble_residuals
            else int(self._config.steps / self._config.steps_per_batch)
        )
        new_actor = networks.ResidualMLP(
            self.actor_policy_kwargs,
            action_dim=self._config.num_actions,
            max_ensemble_size=max_ensemble_size,
            discount_factor=self._config.residual_discount,
            ignore_base_policy=self._config.ignore_base_policy,
            init_zeros=self._config.residual_init_zeros,
            dropout_dims=-self.deter_dim
            if self._config.dropout_recur_in_residuals
            else None,
        )
        self._actor_opt = tools.Optimizer(
            "actor", new_actor.parameters(), **self.actor_opt_kwargs
        )
        if copy_base_actor_weights:
            new_actor.load_base_state_dict(self.actor.state_dict())
        self.actor = new_actor
        self.actor.to(device)

    def add_new_residual(self):
        assert self.train_residuals and isinstance(self.actor, networks.ResidualMLP)
        new_residual_params = self.actor.create_new_residual()
        if new_residual_params:
            self._actor_opt.add_new_params(new_residual_params)

    def _train(self, start, objective, expert_data=None):
        self._update_slow_target()
        metrics = {}

        if expert_data:
            expert_data = self._world_model.preprocess(expert_data)
            expert_embed = self._world_model.encoder(expert_data)
            expert_start, _ = self._world_model.dynamics.observe(
                expert_embed, expert_data["action"], expert_data["is_first"]
            )
        else:
            expert_start = None
            expert_data = dict(action=None)

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # start['stoch'] = (batch_size, batch_length, stoch_dim)
                imag_feat, imag_state, imag_action = self._imagine(
                    start,
                    self.actor,
                    self._config.imag_horizon,
                )
                # imag_state['stoch'] = (imag_horizon, batch_size*batch_length, stoch_dim)
                reward = objective(imag_feat, imag_state, imag_action)
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                    exp_feat=expert_start,
                    exp_actions=expert_data["action"],
                )
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))

        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _train_critic_only(self, start, objective):
        self._update_slow_target()
        metrics = {}
        with torch.no_grad():
            imag_feat, imag_state, imag_action = self._imagine(
                start, self.actor, self._config.imag_horizon
            )
            reward = objective(imag_feat, imag_state, imag_action)
            target, weights, base = self._compute_target(imag_feat, imag_state, reward)
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(imag_feat[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(imag_feat[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
        exp_feat=None,
        exp_actions=None,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)

        if self._config.bc_reg:
            # inp: (imag_horizon, batch_size*batch_length, ...)
            # state: (batch_size, batch_length, ...)
            state = self._world_model.dynamics.get_feat(exp_feat)
            state = state.detach()
            bc_policy = self.actor(state)
            exp_actions = torch.Tensor(exp_actions).to(self._config.device)
            bc_loss = self.bc_reg_weight * F.mse_loss(bc_policy.mode(), exp_actions)
            metrics["bc_loss"] = to_np(bc_loss)
            # calculate norm on last layer:
            if isinstance(bc_policy, tools.ResidualActionWrapper):
                metrics["residual_discount"] = bc_policy._discount
                residual_dist = bc_policy._residual_dist
                metrics["residual_mean"] = to_np(
                    torch.mean(torch.norm(residual_dist.mode(), dim=-1))
                )
            if self._config.bc_loss_only:
                return bc_loss, metrics
        else:
            bc_loss = 0

        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        rl_loss = -weights[:-1] * actor_target
        actor_ent = policy.entropy()
        rl_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
        actor_loss = torch.mean(rl_loss) + bc_loss
        metrics["rl_loss"] = to_np(torch.mean(rl_loss))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
