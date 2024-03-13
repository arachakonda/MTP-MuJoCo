import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from gym_quad.envs.mujoco import UAVQuadBase
from gym_quad.controller import RpyController

class Trajectory:
    R = 0.5   # trajectory radius
    w = 0.1   # trajectory angular speed (rad/s)

def main():
    env = gym.make('UAVQuadBase-v0', render_mode='human')

    dt = env.dt
    mass = env.mass
    gravity = env.gravity[2]

    # controller
    ctrl = RpyController(dt, mass, gravity=gravity)

    # [x, y, z, q0, q1, q2, q3]
    observation, _ = env.reset()

    # To store the trajectory data
    actual_positions = []
    desired_positions = []

    for t in range(1000):
        env.render()

        # current position and quaternion
        s = np.array([observation[0], observation[1], observation[2]])
        q = observation[3:7]

        # Store actual position for plotting
        actual_positions.append(s)

        # Calculate and store desired position for plotting
        pos_d = np.array([
            Trajectory.R * np.cos(Trajectory.w * dt * t),
            Trajectory.R * np.sin(Trajectory.w * dt * t),
            1.0
        ])
        desired_positions.append(pos_d)

        yaw_d = (Trajectory.w * dt * t + np.pi) % (2 * np.pi) - np.pi

        # control
        action = ctrl.control(s, q, pos_d, yaw_d)
        observation, reward, done, _, info = env.step(action)

        if done:
            break

    # Convert lists to numpy arrays for easier plotting
    actual_positions = np.array(actual_positions)
    desired_positions = np.array(desired_positions)

    # Plotting the trajectory
    plt.figure()
    plt.title('3D Trajectory Comparison')
    ax = plt.axes(projection='3d')
    ax.plot3D(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 'blue', label='Actual Path')
    ax.plot3D(desired_positions[:, 0], desired_positions[:, 1], desired_positions[:, 2], 'green', linestyle='dashed', label='Desired Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
