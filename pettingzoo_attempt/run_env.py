import numpy as np
import random
import time
from slitherin import Slitherin

def print_observations(observations):
    """Print observations in a more readable format."""
    for agent_id, obs in observations.items():
        print(f"\nObservation for {agent_id}:")
        # Convert observation values to symbols for better visualization
        symbol_map = {
            0: ' ',    # Empty
            1: '■',    # Own body
            2: '●',    # Own head
            -1: '□',   # Other body
            -2: '○',   # Other head
            3: 'A',    # Apple
        }
        
        for row in obs:
            print(''.join([symbol_map.get(int(cell), '?') for cell in row]))

def main():
    # Create environment with 3 agents
    env = Slitherin(
        grid_size=(8, 8),
        rewards={"win": 10, "idle": -0.1, "lose": -10},
        num_agents=3,
        render_mode="human"  # Set to "human" for visualization, None for faster execution
    )
    
    # Set seed for reproducibility
    seed = 42
    
    # Initialize environment
    observations, infos = env.reset(seed=seed)
    
    # Print initial state
    print("\n=== INITIAL STATE ===")
    print(f"Number of agents: {len(env.agents)}")
    print(f"Agent IDs: {env.agents}")
    
    print_observations(observations)
    
    # Run for a certain number of steps or until all agents are terminated
    max_steps = 100
    step = 0
    
    try:
        while env.agents and step < max_steps:
            step += 1
            print(f"\n=== STEP {step} ===")
            print(f"Active agents: {env.agents}")
            
            # Select random actions for each agent
            actions = {agent_id: random.randint(0, 3) for agent_id in env.agents}
            print(f"Actions: {actions}")
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Display results
            print(f"Rewards: {rewards}")
            
            # Print detailed info about living and dead snakes
            living_count = sum(1 for agent_id in env.possible_agents if 
                               agent_id in infos and not infos[agent_id].get('is_dead', False))
            print(f"Living snakes: {living_count}")
            
            for agent_id in env.possible_agents:
                if agent_id in infos:
                    snake_status = "ALIVE" if not infos[agent_id].get('is_dead', False) else "DEAD"
                    print(f"{agent_id}: {snake_status}")
            
            # Render the environment
            env.render()
            
            # Small delay to make visualization easier to follow
            if env.render_mode == "human":
                time.sleep(0.5)
            
            # Remove terminated agents
            terminated_agents = [agent_id for agent_id in env.agents if terminations[agent_id]]
            if terminated_agents:
                print(f"Agents terminated this step: {terminated_agents}")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        # Close the environment
        env.close()
        
        print("\n=== SIMULATION COMPLETE ===")
        print(f"Completed {step} steps")
        print(f"Remaining agents: {env.agents}")

if __name__ == "__main__":
    main()