from collections import deque
import time
from slitherin import Slitherin
from .model import DQNAgent

# Main training function
def train_dqn_self_play(num_episodes, model_path, random_seed, target_update, lr, mem_size, eps_start, eps_end, eps_decay, batch_size, gamma, rewards, grid_size, num_agents):
    # Create environment
    env = Slitherin(
        grid_size=grid_size,
        rewards=rewards,
        num_agents=num_agents,
        render_mode="human", 
    )
    
    # Initialize agents (one per snake)
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = DQNAgent(grid_size, 4, lr, mem_size, eps_start, eps_end, eps_decay, batch_size, gamma)  # 4 actions: up, right, down, left
    
    # Training loop
    best_avg_score = float('-inf')
    scores_window = deque(maxlen=100)
    
    try:
        for episode in range(1, num_episodes + 1):
            episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
            episode_losses = {agent_id: [] for agent_id in env.possible_agents}
            
            # Reset the environment
            observations, infos = env.reset(seed=random_seed + episode)
            
            # Run episode
            done = False
            step = 0
            
            while env.agents and not done and step < 5000:  # Max steps per episode
                step += 1
                
                # Select actions for each agent
                actions = {}
                for agent_id in env.agents:
                    actions[agent_id] = agents[agent_id].select_action(observations[agent_id])
                
                # Execute actions
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Store experiences and optimize models
                for agent_id in env.agents:
                    # Store the transition in memory
                    next_state = next_observations.get(agent_id, None)
                    if next_state is None:  # Terminal state
                        next_state = None
                    
                    agents[agent_id].memory.push(
                        observations[agent_id],
                        actions[agent_id],
                        next_state,
                        rewards[agent_id],
                        terminations.get(agent_id, False)
                    )
                    
                    # Accumulate rewards
                    episode_rewards[agent_id] += rewards[agent_id]
                    
                    # Optimize the model
                    loss = agents[agent_id].optimize_model()
                    if loss is not None:
                        episode_losses[agent_id].append(loss)
                
                # Update the observations
                observations = next_observations
                
                # Check if episode is done
                done = all(terminations.values()) or all(truncations.values())
            
            # Update target networks periodically
            if episode % target_update == 0:
                for agent_id in env.possible_agents:
                    agents[agent_id].update_target_network()
            
            # Calculate average score and loss
            avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
            avg_loss = sum([sum(losses) / max(1, len(losses)) for losses in episode_losses.values()]) / len(episode_losses) if any(episode_losses.values()) else 0
            
            scores_window.append(avg_reward)
            avg_score_window = sum(scores_window) / len(scores_window)
            
            # Save best model
            if avg_score_window > best_avg_score:
                best_avg_score = avg_score_window
                for agent_id in env.possible_agents:
                    agents[agent_id].save_model(f"{model_path}/best_agent_{agent_id}.pth")
            
            # Print progress
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {avg_score_window:.2f} | "
                  f"Length: {step} | "
                  f"Epsilon: {agents[env.possible_agents[0]].eps_threshold:.2f} | "
                  f"Loss: {avg_loss:.4f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Close the environment
        env.close()
        
        # Save final models
        for agent_id in env.possible_agents:
            agents[agent_id].save_model(f"{model_path}/final_agent_{agent_id}.pth")
        
        print("\n=== TRAINING COMPLETE ===")
        print(f"Best average score: {best_avg_score:.2f}")

# Evaluate trained agents
def evaluate_agents(model_path, random_seed, rewards, num_agents, grid_size, lr, mem_size, eps_start, eps_end, eps_decay, batch_size, gamma, num_episodes=10, render=True):
    # Create environment
    env = Slitherin(
        grid_size=grid_size,
        rewards=rewards,
        num_agents=num_agents,
        render_mode="human" if render else None
    )
    
    # Load trained agents
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = DQNAgent(grid_size, 4, lr, mem_size, eps_start, eps_end, eps_decay, batch_size, gamma)
        try:
            agents[agent_id].load_model(f"{model_path}/best_agent_{agent_id}.pth")
            print(f"Loaded model for agent {agent_id}")
        except:
            print(f"Could not load model for agent {agent_id}, using random policy")
    
    # Run evaluation episodes
    total_wins = {agent_id: 0 for agent_id in env.possible_agents}
    total_rewards = {agent_id: 0 for agent_id in env.possible_agents}
    
    for episode in range(num_episodes):
        observations, infos = env.reset(seed=random_seed + 10000 + episode)
        episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        done = False
        step = 0
        
        while env.agents and not done and step < 5000:
            step += 1
            
            # Select actions (no exploration during evaluation)
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = agents[agent_id].select_action(observations[agent_id], eval_mode=True)
            
            # Execute actions
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Accumulate rewards
            for agent_id in env.agents:
                episode_rewards[agent_id] += rewards[agent_id]
            
            # Render if enabled
            if render:
                env.render()
                time.sleep(0.1)
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
        
        # Determine winner
        surviving_agents = [agent_id for agent_id in env.possible_agents if agent_id in env.agents]
        if surviving_agents:
            best_agent = max(surviving_agents, key=lambda x: episode_rewards[x])
            total_wins[best_agent] += 1
        
        # Accumulate rewards
        for agent_id in env.possible_agents:
            total_rewards[agent_id] += episode_rewards[agent_id]
        
        print(f"Episode {episode+1}/{num_episodes} complete | "
              f"Steps: {step} | "
              f"Survivors: {len(env.agents)} | "
              f"Rewards: {', '.join([f'{agent_id}: {episode_rewards[agent_id]:.1f}' for agent_id in env.possible_agents])}")
    
    # Print overall statistics
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total episodes: {num_episodes}")
    print("Wins:")
    for agent_id in env.possible_agents:
        print(f"  Agent {agent_id}: {total_wins[agent_id]} ({total_wins[agent_id]/num_episodes*100:.1f}%)")
    print("Average rewards:")
    for agent_id in env.possible_agents:
        print(f"  Agent {agent_id}: {total_rewards[agent_id]/num_episodes:.2f}")
    
    env.close()