import os
from packaging.version import Version
from metamotivo.fb_cpr.huggingface import FBcprModel
from huggingface_hub import hf_hub_download
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from pathlib import Path

# Set up the environment
device = "cpu"

if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        )
else:
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space
        )

env, _ = make_humenv(
    num_envs=1,
    wrappers=[
        FlattenObservation,
        transform_obs_wrapper,
    ],
    state_init="Default",
)

# Load the model
model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")
model.to(device)

# Sample a context embedding
z = model.sample_z(1)
print(f"embedding size {z.shape}")
print(f"z norm: {torch.norm(z)}")
print(f"z norm / sqrt(d): {torch.norm(z) / np.sqrt(z.shape[-1])}")

# Run the policy
observation, _ = env.reset()
frames = []
for i in range(30):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    frames.append(env.render())

# Save the video
video_path = "rendered_video.mp4"
fps = 30
height, width, layers = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
video.release()
print(f"Video saved to {video_path}")

# Compute Q-function
def Qfunction(state, action, z_reward, z_policy):
    F = model.forward_map(obs=state, z=z_policy.repeat(state.shape[0],1), action=action)
    Q = F @ z_reward.ravel()
    return Q.mean(axis=0)

z_reward = model.sample_z(1)
z_policy = model.sample_z(1)
state = torch.rand((10, env.observation_space.shape[0]), device=model.cfg.device, dtype=torch.float32)
action = torch.rand((10, env.action_space.shape[0]), device=model.cfg.device, dtype=torch.float32)*2 - 1
Q = Qfunction(state, action, z_reward, z_policy)
print(Q)

# Prompting the model
local_dir = "metamotivo-S-1-datasets"
dataset = "buffer_inference_500000.hdf5"
buffer_path = hf_hub_download(
    repo_id="facebook/metamotivo-S-1",
    filename=f"data/{dataset}",
    repo_type="model",
    local_dir=local_dir,
)

hf = h5py.File(buffer_path, "r")
data = {}
for k, v in hf.items():
    print(f"{k:20s}: {v.shape}")
    data[k] = v[:]
buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
buffer.extend(data)
del data

reward_fn = humenv_rewards.LocomotionReward(move_speed=2.0)
N = 100_000
batch = buffer.sample(N)
rewards = []
for i in range(N):
    rewards.append(
        reward_fn(
            env.unwrapped.model,
            qpos=batch["next_qpos"][i],
            qvel=batch["next_qvel"][i],
            ctrl=batch["action"][i])
    )
rewards = np.stack(rewards).reshape(-1,1)
print(rewards.ravel())

z = model.reward_wr_inference(
    next_obs=batch["next_observation"],
    reward=torch.tensor(rewards, device=model.cfg.device, dtype=torch.float32)
)
print(z.shape)

# Run the inferred policy
observation, _ = env.reset()
frames = []
for i in range(30):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    frames.append(env.render())

# Save the inferred policy video
inferred_video_path = "inferred_policy_video.mp4"
video = cv2.VideoWriter(inferred_video_path, fourcc, fps, (width, height))
for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
video.release()
print(f"Inferred policy video saved to {inferred_video_path}")

# Compute Q-function for inferred policy
z_reward = torch.sum(
    model.backward_map(obs=batch["next_observation"]) * torch.tensor(rewards, dtype=torch.float32, device=model.cfg.device),
    dim=0
)
z_reward = model.project_z(z_reward)
Q = Qfunction(batch["observation"], batch["action"], z_reward, z)
print(Q)

# Goal inference example
goal_qpos = np.array([0.13769039,-0.20029453,0.42305034,0.21707786,0.94573617,0.23868944
,0.03856998,-1.05566834,-0.12680767,0.11718296,1.89464102,-0.01371153
,-0.07981451,-0.70497424,-0.0478,-0.05700732,-0.05363342,-0.0657329
,0.08163511,-1.06263979,0.09788937,-0.22008936,1.85898192,0.08773695
,0.06200327,-0.3802791,0.07829525,0.06707749,0.14137152,0.08834448
,-0.07649805,0.78328658,0.12580912,-0.01076061,-0.35937259,-0.13176489
,0.07497022,-0.2331914,-0.11682692,0.04782308,-0.13571422,0.22827948
,-0.23456622,-0.12406075,-0.04466465,0.2311667,-0.12232673,-0.25614032
,-0.36237662,0.11197906,-0.08259534,-0.634934,-0.30822742,-0.93798716
,0.08848668,0.4083417,-0.30910404,0.40950143,0.30815359,0.03266103
,1.03959336,-0.19865537,0.25149713,0.3277561,0.16943092,0.69125975
,0.21721349,-0.30871948,0.88890484,-0.08884043,0.38474549,0.30884107
,-0.40933304,0.30889523,-0.29562966,-0.6271498])

env.unwrapped.set_physics(qpos=goal_qpos, qvel=np.zeros(75))
goal_obs = torch.tensor(env.unwrapped.get_obs()["proprio"].reshape(1,-1), device=model.cfg.device, dtype=torch.float32)
print("goal pose")

# Save the goal pose image
goal_image_path = "goal_pose.png"
plt.imshow(env.render())
plt.axis('off')
plt.savefig(goal_image_path, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Goal pose image saved to {goal_image_path}")

z = model.goal_inference(next_obs=goal_obs)

observation, _ = env.reset()
frames = []
for i in range(30):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    frames.append(env.render())

# Save the goal inference video
goal_video_path = "goal_inference_video.mp4"
video = cv2.VideoWriter(goal_video_path, fourcc, fps, (width, height))
for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
video.release()
print(f"Goal inference video saved to {goal_video_path}")
