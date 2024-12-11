import os
import time
import numpy as np
import pathlib
import torch

from torch import nn
import dreamer.exploration as expl
import dreamer.models as models
import dreamer.tools as tools
from common.utils import to_np, combine_dictionaries
#from diffusers.training_utils import EMAModel
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib.patches as patches
import io

# os.environ["MUJOCO_GL"] = "osmesa"


class Dreamer(nn.Module):
    def __init__(
        self, obs_space, act_space, config, logger, dataset, expert_dataset=None
    ):
        super(Dreamer, self).__init__()
        self.dpi = config.size[0]
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_log_video = tools.Every(config.log_every_video)
        batch_steps = config.batch_size * config.batch_length
        self._batch_train_steps = int(
            config.steps_per_batch * config.train_ratio / batch_steps
        )
        print(
            f"Updating the agent for {self._batch_train_steps} every {config.steps_per_batch} env steps"
        )
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))

        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat if logger is not None else 0
        self._update_count = 0
        self._dataset = dataset
        self._expert_dataset = expert_dataset
        self._hybrid_training = expert_dataset is not None

        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()  # noqa: E731
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        if (
            config.pretrain_actor_steps > 0
            or config.pretrain_joint_steps > 0
            or config.from_ckpt is not None
        ):
            self._make_pretrain_opt()

    def __call__(self, obs, reset, state=None, training=True):
        # step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._batch_train_steps
            )
            for _ in range(steps):
                # if self._hybrid_training and np.random.uniform() < 0.5:
                if self._hybrid_training:
                    learner_data, exp_data = (
                        next(self._dataset),
                        next(self._expert_dataset),
                    )
                    mixed_data = combine_dictionaries(
                        learner_data, exp_data, take_half=True
                    )
                    self._train(mixed_data)
                else:
                    self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            self._maybe_log_metrics(video_pred_log=self._config.video_pred_log)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _train(self, data, expert_data=None):
        metrics = {}

        # train world model
        if self._hybrid_training and expert_data:
            data = combine_dictionaries(data, expert_data, take_half=True)
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post

        # train actor
        metrics.update(
            self._task_behavior._train(start, self._reward_fn, expert_data)[-1]
        )
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        self._update_running_metrics(metrics)

    def _train_model_only(self, data, expert_data=None, frozen_heads=[]):
        if expert_data:
            data = combine_dictionaries(data, expert_data)
        _, _, mets = self._wm._train(data, frozen_heads)
        self._update_running_metrics(mets)

    def _train_reward_only(self, data, expert_data=None):
        if expert_data:
            data = combine_dictionaries(data, expert_data)
        mets = self._wm._train_reward_only(data)
        self._update_running_metrics(mets)

    def _train_actor_only(self, data, expert_data=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast(self._wm._use_amp):
                data = self._wm.preprocess(data)
                embed = self._wm.encoder(data)
                post, _ = self._wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
        post = {k: v.detach() for k, v in post.items()}
        metrics = self._task_behavior._train(post, self._reward_fn, expert_data)[-1]
        self._update_running_metrics(metrics)

    def _train_critic_only(self, train_dataset, expert_dataset, utd_ratio):
        for i in range(utd_ratio):
            if self._config.hybrid_critic_fitting:
                data = combine_dictionaries(next(train_dataset), next(expert_dataset))
            else:
                data = next(train_dataset)
            with torch.cuda.amp.autocast(self._wm._use_amp):
                with torch.no_grad():
                    data = self._wm.preprocess(data)
                    embed = self._wm.encoder(data)
                    post, _ = self._wm.dynamics.observe(
                        embed, data["action"], data["is_first"]
                    )
                post = {k: v.detach() for k, v in post.items()}
                metrics = self._task_behavior._train_critic_only(post, self._reward_fn)[
                    -1
                ]
                self._update_running_metrics(metrics)

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def pretrain_actor_model(self, data, step=None):
        """
        Data: "agentview_image", "state", "robot0_eye_in_hand_image",
              "reward", "is_first", "is_last", "is_terminal", "discount", "action", "cont"
        Shape: (batch_size x batch_length x ...)
        """
        metrics = {}
        wm = self._wm
        actor = self._task_behavior.actor

        data = wm.preprocess(data)
        if self._config.pretrain_annealing is None:
            recon_weight = 1.0
        elif self._config.pretrain_annealing == "linear":
            recon_weight = (
                self._config.pretrain_joint_steps - (step - 1)
            ) / self._config.pretrain_joint_steps
            recon_weight = max(0.0, recon_weight)
        else:
            print(self._config.pretrain_annealing)
            raise Exception("Annealing strategy must be None or Linear")

        with tools.RequiresGrad(wm), tools.RequiresGrad(actor):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                if (
                    self._config.recon_pretrain
                    and step <= self._config.pretrain_joint_steps
                ):
                    # preds is dictionary of all all MLP+CNN keys
                    preds = wm.heads["decoder"](feat)
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    recon_loss = sum(losses.values())
                else:
                    recon_loss = torch.zeros_like(kl_loss)

                # Zero out kl_loss and recon_loss if pretrain_bc_loss_only is True
                if self._config.pretrain_bc_loss_only == True:
                    kl_loss = torch.zeros_like(kl_loss)
                    recon_loss = torch.zeros_like(recon_loss)

                model_loss = kl_loss + recon_loss

                target = torch.Tensor(data["action"]).to(self._config.device)
                if self._config.dropout_recurrent_prob > 0:
                    feat = self._task_behavior.apply_recurrent_dropout(feat)
                action = actor(feat)
                if self._config.pretrain_loss == "mse":
                    action = action.mode()
                    actor_loss = torch.mean((action - target) ** 2)
                elif self._config.pretrain_loss == "ce":
                    actor_loss = torch.mean(-action.log_prob(target))
                else:
                    raise NotImplementedError(self._config.pretrain_loss)
                model_loss += actor_loss

            metrics = self.pretrain_opt(torch.mean(model_loss), self.pretrain_params)
            if self._config.pretrain_ema:
                self.ema.step(self._task_behavior.actor.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["actor_loss"] = to_np(actor_loss)
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["recon_weight"] = recon_weight

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        metrics = {
            f"model_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics

        self._update_running_metrics(metrics)
        self._step += 1

        if self._logger is not None:
            self._maybe_log_metrics(fps_namespace="model_pretrain/")
            self._logger.step = self._step

        return metrics

    def pretrain_model_only(self, data, step=None):
        metrics = {}
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        if self._config.pretrain_annealing is None:
            recon_weight = 1.0
        elif self._config.pretrain_annealing == "linear":
            recon_weight = (
                self._config.pretrain_joint_steps - (step - 1)
            ) / self._config.pretrain_joint_steps
            recon_weight = max(0.0, recon_weight)
        else:
            print(self._config.pretrain_annealing)
            raise Exception("Annealing strategy must be None or Linear")

        with tools.RequiresGrad(wm), tools.RequiresGrad(actor):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                if (
                    self._config.recon_pretrain
                    and step <= self._config.pretrain_joint_steps
                ):
                    # preds is dictionary of all all MLP+CNN keys
                    preds = wm.heads["decoder"](feat)
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    recon_loss = sum(losses.values())
                else:
                    recon_loss = 0

                model_loss = kl_loss + recon_weight * recon_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["recon_weight"] = recon_weight

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics(fps_namespace="model_only_pretrain/")
        self._step += 1
        self._logger.step = self._step

    def pretrain_actor_only(self, data, step=None):
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        with tools.RequiresGrad(wm), tools.RequiresGrad(actor):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = self._wm.encoder(data)
                post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
                feat = self._wm.dynamics.get_feat(post)
                target = torch.Tensor(data["action"]).to(self._config.device)
                action = actor(feat).mean
                actor_loss = torch.mean((action - target) ** 2)
            metrics = self.pretrain_opt(torch.mean(actor_loss), self.actor_params)
            if self._config.pretrain_ema:
                self.ema.step(self.actor_params)
        metrics["actor_loss"] = to_np(actor_loss)
        metrics = {f"model_pretrain/{k}": v for k, v in metrics.items()}
        self._update_running_metrics(metrics)
        self._maybe_log_metrics(fps_namespace="model_pretrain/")
        self._step += 1
        self._logger.step = self._step

    def pretrain_regress_obs(self, data, obs_mlp, obs_opt, eval=False):
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        if eval:
            obs_mlp.eval()
        with tools.RequiresGrad(obs_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = self._wm.encoder(data)
                post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

                feat = self._wm.dynamics.get_feat(prior).detach() # want the imagined prior to be strong
                target = torch.Tensor(data["privileged_state"]).to(self._config.device)
                pred_state = obs_mlp(feat)
                obs_loss = torch.mean((pred_state - target) ** 2)
            if not eval:
                obs_opt(torch.mean(obs_loss), obs_mlp.parameters())
            else:
                obs_mlp.train()
        return obs_loss.item()

    def get_latent(self, xs, ys, thetas, imgs, lx_mlp):
        states = np.expand_dims(np.expand_dims(thetas,1),1)
        imgs = np.expand_dims(imgs, 1)
        dummy_acs = np.zeros((np.shape(xs)[0], 1, 3))
        rand_idx = 1 #np.random.randint(0, 3, np.shape(xs)[0])
        dummy_acs[np.arange(np.shape(xs)[0]), :, rand_idx] = 1
        firsts = np.ones((np.shape(xs)[0], 1))
        lasts = np.zeros((np.shape(xs)[0], 1))
        
        cos = np.cos(states)
        sin = np.sin(states)
        states = np.concatenate([cos, sin], axis=-1)
        data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}

        data = self._wm.preprocess(data)
        embed = self._wm.encoder(data)

        post, prior = self._wm.dynamics.observe(
            embed, data["action"], data["is_first"]
            )
        feat = self._wm.dynamics.get_feat(post).detach()
        with torch.no_grad():  # Disable gradient calculation
            g_x = lx_mlp(feat).detach().cpu().numpy().squeeze()
        feat = self._wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()
        return g_x, feat, post
    def capture_image(self, state=None):
        """Captures an image of the current state of the environment."""
        # For simplicity, we create a blank image. In practice, this should render the environment.
        fig,ax = plt.subplots()
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis('off')
        fig.set_size_inches( 1, 1 )
        # Create the circle patch
        circle = patches.Circle([0,0], 0.5, edgecolor=(1,0,0), facecolor='none')
        # Add the circle patch to the axis
        dt = 0.05
        v = 1
        dpi=self.dpi
        ax.add_patch(circle)
        
        plt.quiver(state[0], state[1], dt*v*np.cos(state[2]), dt*v*np.sin(state[2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        #plt.savefig('logs/tests/test_rarl.png', dpi=dpi)
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)

        # Load the buffer content as an RGB image
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)
        plt.close()
        return img_array

    def evaluate_classifier(self, obs_mlp, theta=None):
        nx, ny, nz = 41, 41, 5
        if theta is None:
            thetas = np.linspace(0, 2*np.pi, nz, endpoint=True)
        else:
            nz = 1
            thetas = np.array([theta])

        v = np.zeros((nx, ny, nz))
        xs = np.linspace(-1, 1, nx)
        ys = np.linspace(-1, 1, ny)
        tn, tp, fn, fp = 0, 0, 0, 0
        it = np.nditer(v, flags=['multi_index'])
        ###
        idxs = []  
        imgs = []
        labels = []
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]
            theta = thetas[idx[2]]
            if (x**2 + y**2) < (0.5**2):
                labels.append(1) # unsafe
            else:
                labels.append(0) # safe
            x = x - np.cos(theta)*1*0.05
            y = y - np.sin(theta)*1*0.05
            imgs.append(self.capture_image(np.array([x, y, theta])))
            idxs.append(idx)        
            it.iternext()
        idxs = np.array(idxs)
        x_lin = xs[idxs[:,0]]
        y_lin = ys[idxs[:,1]]
        theta_lin = thetas[idxs[:,2]]
        
        g_x = []
        ## all of this is because I can't do a forward pass with 128x128 images in one go
        num_c = nz
        chunk = int(np.shape(x_lin)[0]/num_c)
        for k in range(num_c):
            g_xlist, _, _ = self.get_latent(x_lin[k*chunk:(k+1)*chunk], y_lin[k*chunk:(k+1)*chunk], theta_lin[k*chunk:(k+1)*chunk], imgs[k*chunk:(k+1)*chunk], obs_mlp)
            g_x = g_x + g_xlist.tolist()
        g_x = np.array(g_x)
        v[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = g_x

        return {
            "v": v,
            "g_x": g_x,
            "labels": labels,
        }

    def plot_classifier_evaluation(self, v, g_x, labels):
        _, _, nz = v.shape
        fig, axes = plt.subplots(nz, 2, figsize=(12, nz*6))
        vmax = round(max(np.max(v), 0),1)
        vmin = round(min(np.min(v), -vmax),1)
        safe_idxs = np.where(np.array(labels) == 0)
        unsafe_idxs = np.where(np.array(labels) == 1)
        tp  = np.where(g_x[safe_idxs] > 0)
        fn  = np.where(g_x[safe_idxs] <= 0)
        fp  = np.where(g_x[unsafe_idxs] > 0)
        tn  = np.where(g_x[unsafe_idxs] <= 0)
        
        for i in range(nz):
            if nz > 1:
                ax0 = axes[i, 0]
            else:
                ax0 = axes[0]

            im = ax0.imshow(
                v[:, :, i].T, interpolation='none', extent=np.array([
                -1.1, 1.1, -1.1,1.1, ]), origin="lower",
                cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax0, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax0.set_title(r'$g(x)$', fontsize=18)

            if nz > 1:
                ax1 = axes[i, 1]
            else:
                ax1 = axes[1]

            im = ax1.imshow(
                v[:, :, i].T > 0, interpolation='none', extent=np.array([
                -1.1, 1.1, -1.1,1.1, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax1, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax1.set_title(r'$v(x)$', fontsize=18)
            fig.tight_layout()
            circle = plt.Circle((0, 0), 0.5, fill=False, color='blue', label = 'GT boundary')

            # Add the circle to the plot
            ax0.add_patch(circle)
            ax0.set_aspect('equal')
            circle2 = plt.Circle((0, 0), 0.5, fill=False, color='blue', label = 'GT boundary')

            ax1.add_patch(circle2)
            ax1.set_aspect('equal')

        fp_g = np.shape(fp)[1]
        fn_g = np.shape(fn)[1]
        tp_g = np.shape(tp)[1]
        tn_g = np.shape(tn)[1]
        tot = fp_g + fn_g + tp_g + tn_g
        fig.suptitle(r"$TP={:.0f}\%$ ".format(tp_g/tot * 100) + r"$TN={:.0f}\%$ ".format(tn_g/tot * 100) + r"$FP={:.0f}\%$ ".format(fp_g/tot * 100) +r"$FN={:.0f}\%$".format(fn_g/tot * 100),
            fontsize=10,)
        buf = BytesIO()

        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        return np.array(plot), tp, fn, fp, tn
    
    def get_eval_plot(self, obs_mlp, theta=None):
        eval_result = self.evaluate_classifier(obs_mlp, theta)
        return self.plot_classifier_evaluation(eval_result["v"], eval_result["g_x"], eval_result["labels"])

    def train_lx(self, data, lx_mlp, lx_opt, eval=False):
        wm = self._wm
        wm.dynamics.sample = False
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        R = 0.5
        with tools.RequiresGrad(lx_mlp):
            if not eval:
                with torch.cuda.amp.autocast(wm._use_amp):
                    embed = self._wm.encoder(data)
                    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                    feat = self._wm.dynamics.get_feat(post).detach() 
                    
                    x, y, theta = data["privileged_state"][:,:,0], data["privileged_state"][:,:,1], data["privileged_state"][:,:, 2]

                    safety_data = (x**2 + y**2) - R**2
                    safe_data = torch.where(safety_data > 0)
                    unsafe_data = torch.where(safety_data <= 0)

                    safe_dataset = feat[safe_data]
                    unsafe_dataset = feat[unsafe_data]

                    pos = lx_mlp(safe_dataset)
                    neg = lx_mlp(unsafe_dataset)
                    
                    
                    gamma = 0.75
                    lx_loss = (1/pos.size(0))*torch.sum(torch.relu(gamma - pos)) #penalizes safe for being negative
                    lx_loss +=  (1/neg.size(0))*torch.sum(torch.relu(gamma + neg)) # penalizes unsafe for being positive
                    
                    lx_loss = lx_loss
            
                    lx_opt(torch.mean(lx_loss), lx_mlp.parameters())
                    plot_arr = None
                    score = 0
            else:
                lx_mlp.eval()
                plot_arr, tp, fn, fp, tn = self.get_eval_plot(lx_mlp)
                '''safe_pts = data['privileged_state'][safe_data]
                unsafe_pts = data['privileged_state'][unsafe_data]

                s_scores = lx_mlp(feat)[safe_data]
                tp_idx = torch.where(s_scores > 0)
                fn_idx = torch.where(s_scores <= 0)
                u_scores = lx_mlp(feat)[unsafe_data]
                fp_idx = torch.where(u_scores > 0)
                tn_idx = torch.where(u_scores <= 0)


                fig, ax = plt.subplots()
                tp = safe_pts[tp_idx[0]].detach().cpu().numpy()
                fn = safe_pts[fn_idx[0]].detach().cpu().numpy()
                tn = unsafe_pts[tn_idx[0]].detach().cpu().numpy()
                fp = unsafe_pts[fp_idx[0]].detach().cpu().numpy()
                plt.xlim([-1.1, 1.1])
                plt.ylim([-1.1, 1.1])
                plt.axis('off')
                fig.set_size_inches( 6, 6 ) 
                plt.title('loss: ' + str(lx_loss.item()))
                plt.quiver(tp[:,0], tp[:,1], 0.05*1*np.cos(tp[:,2]), 0.05*1*np.sin(tp[:,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.4,zorder=2, color='green', label='True Positive')
                plt.quiver(fn[:,0], fn[:,1], 0.05*1*np.cos(fn[:,2]), 0.05*1*np.sin(fn[:,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.4,zorder=3, color='purple', label='False Negative')
                plt.quiver(fp[:,0], fp[:,1], 0.05*1*np.cos(fp[:,2]), 0.05*1*np.sin(fp[:,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.4,zorder=3, color='blue', label='False Positive')
                plt.quiver(tn[:,0], tn[:,1], 0.05*1*np.cos(tn[:,2]), 0.05*1*np.sin(tn[:,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.4,zorder=2, color='red', label='True Negative')
                plt.legend()
                circle = plt.Circle((0, 0), R, fill=False, color='blue', label = 'GT boundary')

                # Add the circle to the plot
                ax.add_patch(circle)
                ax.set_aspect('equal')
                buf = BytesIO()

                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                plot = Image.open(buf).convert("RGB")
                plot_arr = np.array(plot)'''

                lx_mlp.train()
                fp_num = np.shape(fp)[1]
                fn_num = np.shape(fn)[1]
                tp_num = np.shape(tp)[1]
                tn_num = np.shape(tn)[1]
                print('TP: ', tp_num)
                print('FN: ', fn_num)

                print('TN: ', tn_num)
                print('FP: ', fp_num)
            
                score = (fp_num + fn_num) / (fp_num + fn_num + tp_num + tn_num)

        return score, plot_arr
    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _maybe_log_metrics(self, video_pred_log=False, fps_namespace=""):
        if self._logger is not None:
            logged = False
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    if not np.isnan(np.mean(values)):
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                logged = True

            if video_pred_log and self._should_log_video(self._step):
                video_pred, video_pred2 = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl_agent", to_np(video_pred))
                self._logger.video("train_openl_hand", to_np(video_pred2))
                logged = True

            if logged:
                self._logger.write(fps=True, fps_namespace=fps_namespace)

    def _make_pretrain_opt(self):
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (
            config.pretrain_actor_steps + config.pretrain_joint_steps > 0
            or config.from_ckpt is not None
        ):
            # have separate lrs/eps/clips for actor and model
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
            }
            if config.recon_pretrain:
                model_params["params"] += list(self._wm.heads["decoder"].parameters())
            actor_params = {
                "params": list(self._task_behavior.actor.parameters()),
                "lr": config.actor["lr"],
                "eps": config.actor["eps"],
                "clip": config.actor["grad_clip"],
            }
            self.pretrain_params = list(model_params["params"]) + list(
                actor_params["params"]
            )
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt", [model_params, actor_params], **standard_kwargs
            )
            self.actor_params = list(self._task_behavior.actor.parameters())
            if config.pretrain_ema:
                self.ema = EMAModel(
                    parameters=self.actor_params,
                    decay=config.ema_decay,
                    power=config.ema_power,
                )
                print("EMA")
            print(
                f"Optimizer pretrain has {sum(param.numel() for param in self.pretrain_params)} variables."
            )

    def _reward_fn(self, feat, state, action):
        return self._wm.heads["reward"](self._wm.dynamics.get_feat(state)).mode()
