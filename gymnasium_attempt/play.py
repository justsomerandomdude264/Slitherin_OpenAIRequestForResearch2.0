import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Make sure to import registration first to register environments
import slitherin

def main():
    """
    Example showing how to use Slitherin with 3 agents using the Gymnasium API
    """
    print("Creating Slitherin environment with 3 agents...")
    
    # Create the environment using the registered ID
    env = gym.make("Slitherin-7x7-v0")
    
    # Run a few episodes
    num_episodes = 3
    rewards_history = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        observations, _ = env.reset()
        
        episode_rewards = [0] * env.unwrapped.num_agents
        terminated = [False] * env.unwrapped.num_agents
        truncated = [False] * env.unwrapped.num_agents
        
        # Track step count
        step_count = 0
        
        while not all(terminated) and step_count < 100:
            step_count += 1
            
            # Generate random actions for each agent (skip actions for dead snakes)
            actions = []
            for i, term in enumerate(terminated):
                if term:
                    actions.append(0)  # Dummy action for dead snakes
                else:
                    actions.append(random.randint(0, 3))
            
            # Take step in environment
            observations, rewards, terminated, truncated, info = env.step(actions)
            
            # Update episode rewards
            for i in range(len(episode_rewards)):
                episode_rewards[i] += rewards[i]
            
            # Print step information
            print(f"Step {step_count}: Living snakes: {info['living_snakes']}")
            print(f"Rewards: {rewards}")
            print(f"Terminated: {terminated}")
            
            # Optional: add a small delay to see what's happening
            time.sleep(0.1)
        
        print(f"Episode {episode+1} completed in {step_count} steps")
        print(f"Final rewards: {episode_rewards}")
        rewards_history.append(episode_rewards)
    
    # Print overall results
    print("\nResults across all episodes:")
    rewards_array = np.array(rewards_history)
    for agent_idx in range(env.unwrapped.num_agents):
        agent_rewards = rewards_array[:, agent_idx]
        print(f"Agent {agent_idx+1}: Avg reward = {np.mean(agent_rewards):.2f}, Min = {np.min(agent_rewards):.2f}, Max = {np.max(agent_rewards):.2f}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()