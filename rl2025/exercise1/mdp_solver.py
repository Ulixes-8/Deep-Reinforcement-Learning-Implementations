from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2025.constants import EX1_CONSTANTS as CONSTANTS
from rl2025.exercise1.mdp import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

        Useful Variables:
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        V = np.zeros(self.state_dim)
        # Main loop of value iteration
        while True:
            # Initialize delta for convergence check
            delta = 0
            
            # Loop through all states
            for s in range(self.state_dim):
                # Skip terminal states (their value remains 0)
                if self.mdp.terminal_mask[s]:
                    continue
                    
                # Store old value for convergence check
                v_old = V[s]
                
                # Initialize array to store action values
                action_values = np.zeros(self.action_dim)
                
                # Calculate value for each action
                for a in range(self.action_dim):
                    # Only consider next states with non-zero transition probability
                    next_states = np.where(self.mdp.P[s, a] > 0)[0]
                    
                    # Sum over all possible next states
                    for s_next in next_states:
                        # Get transition probability and reward
                        p = self.mdp.P[s, a, s_next]
                        r = self.mdp.R[s, a, s_next]
                        
                        # Add weighted value to action value
                        action_values[a] += p * (r + self.gamma * V[s_next])
                
                # Update state value with maximum action value
                V[s] = np.max(action_values)
                
                # Update delta for convergence check
                delta = max(delta, abs(v_old - V[s]))
            
            # Check for convergence
            if delta < theta:
                break
        
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        
        # For each state...
        for s in range(self.state_dim):
            # Calculate action values for this state
            action_values = np.zeros(self.action_dim)
            for a in range(self.action_dim):
                # Calculate expected value for each action
                for s_next in range(self.state_dim):
                    p = self.mdp.P[s, a, s_next]
                    if p > 0:  # Only consider non-zero transitions
                        r = self.mdp.R[s, a, s_next]
                        action_values[a] += p * (r + self.gamma * V[s_next])
            
            # Find best action (argmax)
            best_action = np.argmax(action_values)
            
            # Set deterministic policy (probability 1.0 for best action)
            policy[s, best_action] = 1.0
        
        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)
        
        # Policy evaluation loop
        while True:
            # Initialize delta for tracking the maximum change
            delta = 0
            
            # Loop through each state
            for s in range(self.state_dim):
                # Skip terminal states - their value remains 0
                if self.mdp.terminal_mask[s]:
                    continue
                    
                # Store old value for convergence check
                v_old = V[s]
                
                # Initialize new V(s) to zero before summation
                new_value = 0
                
                # For each action with non-zero probability under the policy
                for a in range(self.action_dim):
                    if policy[s, a] > 0:
                        # Calculate expected value for this action
                        action_value = 0
                        
                        # Sum over all possible next states
                        for s_prime in range(self.state_dim):
                            # Get transition probability and reward
                            prob = self.mdp.P[s, a, s_prime]
                            reward = self.mdp.R[s, a, s_prime]
                            
                            # Add to expected value: p(s',r|s,a) * [r + γV(s')]
                            action_value += prob * (reward + self.gamma * V[s_prime])
                        
                        # Weight by the probability of taking this action under the policy
                        new_value += policy[s, a] * action_value
                
                # Update the value function
                V[s] = new_value
                
                # Update delta for convergence check
                delta = max(delta, abs(v_old - V[s]))
            
            # Check for convergence
            if delta < self.theta:
                break
        
        return V


    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes policy iteration until a stable policy is reached

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
    # Initialize policy arbitrarily
        # policy = np.ones([self.state_dim, self.action_dim]) / self.action_dim
        policy = np.zeros([self.state_dim, self.action_dim])

            # Set first action to 1.0 for each state to ensure a valid initial policy
        for s in range(self.state_dim):
            policy[s, 0] = 1.0

        # Main policy iteration loop
        while True:
            # Step 1: Policy evaluation
            V = self._policy_eval(policy)
            
            # Step 2: Policy improvement
            policy_stable = True
            
            # For each state
            for s in range(self.state_dim):
                # Skip terminal states in improvement
                if self.mdp.terminal_mask[s]:
                    continue
                    
                # Store the old action
                old_action = np.argmax(policy[s])
                
                # Calculate action values
                action_values = np.zeros(self.action_dim)
                for a in range(self.action_dim):
                    for s_prime in range(self.state_dim):
                        action_values[a] += self.mdp.P[s, a, s_prime] * (
                            self.mdp.R[s, a, s_prime] + self.gamma * V[s_prime]
                        )
                
                # Find the new best action
                new_action = np.argmax(action_values)
                
                # Create deterministic policy
                new_policy_for_state = np.zeros(self.action_dim)
                new_policy_for_state[new_action] = 1.0
                
                # Check if policy has changed
                if old_action != new_action:
                    policy_stable = False
                
                # Update policy
                policy[s] = new_policy_for_state
            
            # If policy is stable, we're done
            if policy_stable:
                # V = self._policy_eval(policy)
                break
        
        return policy, V
    
    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
        Transition("rock0", "jump0", "rock0", 1, 0),
        Transition("rock0", "stay", "rock0", 1, 0),
        Transition("rock0", "jump1", "rock0", 0.1, 0),
        Transition("rock0", "jump1", "rock1", 0.9, 0),
        Transition("rock1", "jump0", "rock1", 0.1, 0),
        Transition("rock1", "jump0", "rock0", 0.9, 0),
        Transition("rock1", "jump1", "rock1", 0.1, 0),
        Transition("rock1", "jump1", "land", 0.9, 10),
        Transition("rock1", "stay", "rock1", 1, 0),
        Transition("land", "stay", "land", 1, 0),
        Transition("land", "jump0", "land", 1, 0),
        Transition("land", "jump1", "land", 1, 0),
    )

    solver = ValueIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)
