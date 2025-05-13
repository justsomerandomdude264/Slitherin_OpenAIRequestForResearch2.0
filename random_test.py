import numpy as np
import random
import time
from slitherin import Slitherin

def main():
    # Create environment with 3 agents
    env = Slitherin(
        grid_size=(10, 10),
        rewards={"win": 10, "idle": -0.1, "lose": -10},
        num_agents=5,
        render_mode="human"  
    )
    
    # Set seed for reproducibility
    seed = 42
    
    # Initialize environment
    observations, infos = env.reset()
    
    # Print initial state
    print(f"Number of agents: {len(env.agents)}")
    print(f"Agent IDs: {env.agents}")
    
    # Run for a certain number of steps or until all agents are terminated
    max_steps = 3000
    step = 0
    
    try:
        while env.agents and step < max_steps:
            step += 1
            
            # Select random actions for each agent
            actions = {agent_id: random.randint(0, 3) for agent_id in env.agents}
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Get detailed info about living and dead snakes
            living_count = sum(1 for agent_id in env.possible_agents if 
                               agent_id in infos and not infos[agent_id].get('is_dead', False))                
            for agent_id in env.possible_agents:
                if agent_id in infos:
                    snake_status = "ALIVE" if not infos[agent_id].get('is_dead', False) else "DEAD"
                    snake_score = infos[agent_id].get('snake_score')
            
            # Render the environment
            env.render()
            
            # Small delay to make visualization easier to follow
            if env.render_mode == "human":
                time.sleep(0.1)
            
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
        print(f"Remaining agents: {env.agents}")

if __name__ == "__main__":
    main()