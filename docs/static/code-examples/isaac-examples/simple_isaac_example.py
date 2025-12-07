#!/usr/bin/env python3

"""
Simple Isaac Sim Python Example
This script demonstrates basic Isaac Sim functionality using Omniverse Kit APIs.
"""

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np


class SimpleIsaacExample:
    def __init__(self):
        """Initialize the Isaac Sim example"""
        print("Initializing Simple Isaac Example")

        # Create world instance
        self.world = World(stage_units_in_meters=1.0)

        # Add a simple robot to the stage
        self.setup_scene()

    def setup_scene(self):
        """Set up the simulation scene"""
        # Get the assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please enable Isaac Sim Nucleus.")
            return

        # Add a simple robot (using a basic cube as placeholder)
        # In a real example, you would load a proper robot asset
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Robots/Franka/franka_alt_fingers.usd",
            prim_path="/World/Robot"
        )

        # Add a ground plane
        self.world.scene.add_default_ground_plane()

        print("Scene setup complete")

    def run_simulation(self, num_steps=100):
        """Run the simulation for a specified number of steps"""
        print(f"Running simulation for {num_steps} steps")

        # Reset the world
        self.world.reset()

        # Run simulation steps
        for step in range(num_steps):
            # Perform any custom logic here
            if step % 10 == 0:
                print(f"Step {step}")

            # Step the world
            self.world.step(render=True)

        print("Simulation completed")

    def cleanup(self):
        """Clean up resources"""
        if self.world:
            self.world.clear()
        print("Resources cleaned up")


def main():
    """Main function"""
    print("Starting Isaac Sim Example")

    try:
        # Create the example
        example = SimpleIsaacExample()

        # Run the simulation
        example.run_simulation(num_steps=50)

        # Clean up
        example.cleanup()

        print("Isaac Sim Example completed successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()