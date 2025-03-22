
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the Every-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and Every-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) Every-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / Every-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Gamma affects MC more significantly because it explicitly sums discounted rewards across entire episodes, which can be very long. In sparse reward environments (like FrozenLake) with γ=0.8, the single reward from transitioning to the terminal state becomes negligible for long episodes (e.g., 0.8^200), drastically reducing early state-action value updates. However, Q-learning updates incrementally via bootstrapping, propagating reward information backward efficiently without explicitly summing long-term returns. Thus, Q-learning is less sensitive to gamma, as distant rewards influence estimates through shorter incremental steps rather than through entire episode returns over many discounted timesteps. However, they are both still tremendously affected by gamma." # TYPE YOUR ANSWER HERE (100 words max)


    return answer

def question2_5() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) on the differences between the non-slippery and the slippery varian of the problem.
    by [Q-learning / Every-Visit Monte Carlo].
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "In the non-slippery variant, actions produce deterministic outcomes, allowing stable convergence—Monte Carlo (~15,000 episodes) and Q-learning (~2,000 episodes) quickly reach optimal returns with zero variance. The slippery variant introduces stochastic transitions (2/3 unintended), significantly complicating value estimation. The resulting uncertainty causes returns to fluctuate, deflating state-action value estimates, increasing variance and slowing convergence. Lower gamma values (e.g., 0.8) further amplify this effect, heavily discounting the single delayed reward signal, thus deflating early state-action values further. Consequently, mean returns fluctuate more, final returns are lower, and convergence is slower compared to the deterministic scenario."   # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In the DiscreteRL algorithm, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "e"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "A decay strategy based on exploration fraction is more generally applicable because it directly links exploration to the proportional progress of training rather than absolute timesteps. When you specify 'explore fully for the first 10 percent of training', this scales automatically to any environment regardless of episode length or training duration. In contrast, exponential decay requires careful tuning of the decay parameter for each environment's specific timestep scale. This means exploration fraction parameters transfer more easily between tasks with different time horizons, making them more intuitive and requiring less hyperparameter adjustment when changing environments."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "In DQN, the loss does not steadily decrease because the targets depend on a periodically updated target network, creating a 'moving target' problem. Unlike supervised learning with fixed labels, DQN's Q-targets (R + γ·max Q_target) shift as the target network parameters change. This dynamic loss surface prevents stable convergence. And, because we are not simply converging on a fixed dataset, as exploration proceeds, the agent may visit new states where the Q-values are poorly estimated, causing spikes or increases in the loss. Additionally, policy improvement alters the data distribution in the replay buffer (non-stationarity), disrupting learning. Loss may increase when target updates introduce abrupt changes to Q-value estimates, forcing the network to adapt to new targets. Temporal difference errors spike until the critic network adjusts, preventing monotonic loss reduction. This moving target scenario, coupled with changing experience distributions, prevents a stable, consistent decrease in loss throughout training."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The spikes occurring at regular intervals throughout DQN training correspond precisely to the moments when the target network parameters are updated (every 2000 steps in this case). When this hard update occurs, the target values for the Bellman equation suddenly shift, creating an immediate discrepancy between the Q-network's predictions and the new targets. This abrupt change in the loss landscape causes the characteristic spike pattern. After each spike, the loss gradually decreases as the Q-network adapts to the new target values, only to spike again at the next target network update, creating the sawtooth pattern visible in the graph."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = ""  # TYPE YOUR ANSWER HERE (200 words max)
    return answer

print(question3_7())