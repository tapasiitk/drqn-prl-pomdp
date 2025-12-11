# drqn-prl-pomdp
A PyTorch implementation of a Deep Recurrent Q-Network (DRQN) agent solving a Probabilistic Reversal Learning (PRL) task. This project benchmarks the agent's learned strategy and cognitive flexibility against human behavioral data (Weiss et al., 2021) by treating the task as a POMDP.
# Deep Recurrent Q-Networks in Probabilistic Reversal Learning

This repository examines how a Deep Recurrent Q-Network (DRQN) solves a probabilistic reversal learning (PRL) task and compares its behavior to that of **human participants** from Weiss et al. (2021).

The core idea is to treat PRL as a POMDP, train a DRQN agent in a simulated version of the task, and compare its behavioral signatures with those of humans.

***

## 1. Task: Probabilistic Reversal Learning

In probabilistic reversal learning, a subject repeatedly chooses between two options and must:

- learn which option is currently **more rewarding**,
- and detect when this hidden rule **reverses** without warning.


### 1.1 Basic structure

- Two stimuli (left / right image) are shown on each trial.
- The subject chooses one option per trial.
- After each choice, they receive feedback:
    - “correct” (reward; e.g., smiley face), or
    - “incorrect” (no reward/punishment; e.g., frown).
- At any given time, **one stimulus is the “better” option**:
    - Better option: reward on 80% of trials.
    - Worse option: reward on 20% of trials.
- Importantly, this mapping **reverses** during the task.


### 1.2 Instructions to human participants (Weiss et al.-style)

Humans in Weiss et al. (2021) performed a PRL task with instructions of the following form (paraphrased):

> On each trial, you will see two images on the screen, one on the left and one on the right.
> Use the corresponding keys to choose one of the images.
> After each choice, you will receive feedback indicating whether your choice was correct or incorrect (for example, a smiley face for correct, a frown for incorrect).
>
> Your goal is to get as many correct responses as possible.
> The same image will not always lead to the same feedback; sometimes you may get incorrect feedback even for a generally good choice, and correct feedback for a generally bad choice. Therefore, you should pay attention to the overall pattern of feedback over several trials.
>
> The relationship between each image and the feedback may change during the task without warning. A choice that used to be usually correct may later become usually incorrect, and vice versa. If you think this has happened, you should adapt your choices accordingly.

Participants are **not** told the exact reward probabilities (e.g., 80% vs 20%), only that feedback is noisy and can change.

***

## 2. Task Implementation in This Repo

This repository implements a PRL environment closely aligned with Weiss et al. (2021):

- **Actions**: choose `LEFT` or `RIGHT`.
- **Hidden state**:
    - Which option is currently “correct” (higher reward probability).
    - Whether a reversal has recently occurred.
- **Feedback probabilities**:
    - Correct option: reward with probability 0.8.
    - Incorrect option: reward with probability 0.2.
- **Performance-dependent reversals**:
    - After the agent achieves a criterion of **5 consecutive correct choices**, the task **reverses**:
        - The previously correct option becomes incorrect,
        - The previously incorrect option becomes correct.

Thus, the agent faces a partially observable environment; it never directly sees the underlying “correct option” and must infer it from its **history of choices and feedback**.

We treat this as a **POMDP** where:

- true state = current correct option + environment configuration,
- observation = current available options and scalar feedback,
- The agent must maintain an internal belief / memory over the hidden state.

***

## 3. Human Data: Weiss et al. (2021)

We use the human PRL dataset from:

- Weiss, A. et al. (2021). *Developmental Differences in Probabilistic Reversal Learning: A Computational Modeling Approach*. Frontiers in Neuroscience.

They provide trial-by-trial behavioral data (choices and feedback) collected from participants performing a PRL task with:

- two stimuli,
- probabilistic feedback (80/20),
- performance-dependent reversals (learning to criterion).

In this repository, the human dataset is used as a **behavioral benchmark**:

- We do **not** train DRQN directly on these human trials.
- Instead, we compare the **behavioral patterns** of a trained DRQN agent with the aggregated human data.

***

## 4. DRQN Agent: POMDP Solution

We model the agent with a **Deep Recurrent Q-Network (DRQN)** to handle partial observability.

### 4.1 Observation and action

On each trial $t$, the agent observes:

- which options are currently available (left/right stimulus identity),
- the previous feedback signal (reward / no reward).

The agent chooses one of two actions:

- `0 = choose left`,
- `1 = choose right`.


### 4.2 Network architecture

The DRQN consists of:

- an **input encoder** for the observation (e.g., one-hot encoding of choice alternatives + previous reward),
- a **recurrent layer** (LSTM) to maintain a hidden state $h_t$ summarizing history,
- a **linear output layer** producing Q-values for each action:

$$
Q(h_t, a) \quad \text{for } a \in \{\text{left}, \text{right}\}.
$$

At each time step:

1. Observation $o_t$ and previous hidden state $h_{t-1}$ are passed through the LSTM:

$$
h_t = f_{\text{LSTM}}(o_t, h_{t-1}).
$$
2. Q-values are computed from $h_t$.
3. An $\epsilon$-greedy action is selected.

### 4.3 Training

- **Environment**: the simulated PRL task described above.
- **Objective**: maximize cumulative discounted reward.
- **Algorithm**: Q-learning with DRQN:
    - sample trajectories from the environment,
    - compute TD targets using next-step Q-values,
    - backpropagate through the LSTM over sequences.

Crucially:

- The LSTM hidden state $h_t$ is a **learned memory** of past choices and feedback.
- It serves as a **neural approximation** to a belief state over which option is currently better and whether a reversal likely occurred.

***

## 5. Comparison: Humans vs DRQN

After training the DRQN agent in the PRL environment, we compare its behavior to the human data along several metrics, such as:

1. **Choice accuracy over trials**
    - Proportion of choices selecting the currently better option.
    - Learning curves before and after reversals.
2. **Switch latency after reversal**
    - Number of trials the agent/human continues choosing the old correct option after a hidden reversal.
3. **Win–stay / lose–shift patterns**
    - Probability of repeating the same choice after rewarded vs unrewarded outcomes.
4. **Perseveration and exploration**
    - Tendency to stick with the same option despite multiple losses.
    - Frequency of exploration of the alternative option.

The human dataset thus serves as a **behavioral reference**:

- Does the DRQN agent show similar adaptation patterns to humans?
- Does it over- or under-switch after reversals compared to humans?
- Are there qualitative differences in how uncertainty is handled?

***

## 6. Repository Structure

- `env_reversal_learning.py`
POMDP-style PRL environment (2 choices, probabilistic feedback, performance-dependent reversals).
- `drqn_agent.py`
DRQN implementation (observation encoder, LSTM, Q-head).
- `train_drqn.py`
Training loop for the DRQN agent in the simulated PRL environment.
- `human_data/`
Folder containing the Weiss et al. (2021) dataset and a loader.
- `analyze_humans.py`
Scripts to compute behavioral metrics from human data.
- `analyze_agent.py`
Scripts to compute the same metrics for the trained DRQN agent.
- `compare_human_drqn.py`
Functions to generate comparison plots (learning curves, switch behavior, win–stay/lose–shift, etc.).

***

## 7. How to Run

Example workflow (to adapt once your code is in place):

```bash
# 1. Train DRQN in the simulated reversal environment
python train_drqn.py --episodes 50000 --hidden_dim 64

# 2. Analyze trained DRQN behavior
python analyze_agent.py --model_path models/drqn_final.pt

# 3. Analyze human behavior from Weiss et al. dataset
python analyze_humans.py --data_path human_data/weiss_2021.csv

# 4. Generate comparison plots
python compare_human_drqn.py --agent_stats results/agent_stats.pkl \
                             --human_stats results/human_stats.pkl
```


***

## 8. Conceptual Justification

- We treat probabilistic reversal learning as a **task family** with fixed rules (two options, 80/20 feedback, performance-dependent reversals).
- A DRQN agent is trained in a **simulated version** of this task to learn a generic strategy for solving such POMDPs from reward signals alone.
- The Weiss et al. (2021) dataset provides **human behavior** in the same class of tasks.
- We use the human data as a **behavioral benchmark**, comparing qualitative signatures of learning and reversal adaptation between humans and DRQN.

***
