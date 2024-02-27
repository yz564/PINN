import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json
import os
from scipy.interpolate import interp1d
class Pendulum:
    def __init__(self, length=1.0, mass=1.0, gravity=9.81):
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.torque_profile = lambda t: 0.0  # Default torque profile (constant torque)

    def set_torque_profile(self, profile):
        self.torque_profile = profile

    def pendulum_equations(self, t, y):
        theta, omega = y
        torque = self.torque_profile(t)
        dydt = [omega, -(self.gravity / self.length) * np.sin(theta) + torque/(self.mass*self.length*self.length)]
        return dydt

    def simulate(self, initial_state, simulation_time):
        solution = solve_ivp(
            fun=self.pendulum_equations,
            t_span=simulation_time,
            y0=initial_state,
            max_step=0.1,
        )
        return solution

    def mytest(self, initial_state, t_span, step=0.001):
        T=np.arange(t_span[0],t_span[1],step)
        theta=initial_state[0]
        omega=initial_state[1]
        self.theta_array=np.empty(T.shape, dtype=float)
        self.omega_array=np.empty(T.shape, dtype=float)
        self.time_array=T
        for i, t in enumerate(T):
            self.theta_array[i]=theta
            self.omega_array[i]=omega
            omega=omega+step*(self.torque_profile(t)/(self.mass*self.length)-self.gravity*np.sin(theta))/self.length
            theta=theta+omega*step

    def plot_simulation(self, data):
        time=data['time']
        theta=data['theta']
        torque=data['torque']
        plt.plot(time, theta, label='Theta')
        plt.plot(self.time_array, self.theta_array, linestyle="--", label='Theta_validation')
        plt.plot(time, torque, label='Torque')
        #plt.plot(time, [self.torque_profile(t) for t in time], label='Torque')
        plt.title('Pendulum Motion with Time-Dependent Torque')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_torque_data(self, time_points, torque_function):
        torque_data = np.array([[t, torque_function(t)] for t in time_points])
        return torque_data
    def generate_random_torque_data(self, num_points=20):
        time_points = np.linspace(0, 10, num_points)
        torque_values = np.random.uniform(-1.5, 1.5, num_points)  # Adjust the range as needed
        torque_data = np.column_stack((time_points, torque_values))
        return torque_data
    def run_simulation(self, torque_data, initial_state, simulation_time, save_file=None):
        # Create interpolation function from provided torque data
        time_values, torque_values = torque_data[:, 0], torque_data[:, 1]
        torque_interpolation = interp1d(time_values, torque_values, kind='quadratic', fill_value=0.0, bounds_error=False)
        # Setting the interpolated torque profile
        self.set_torque_profile(torque_interpolation)
        # Running simulation
        solution = self.simulate(initial_state, simulation_time)
        torque = torque_interpolation(solution.t)
        def round_array_values(arr, decimals=4):
            return np.round(arr, decimals).tolist()
        data_to_save = {
            'time': round_array_values(solution.t),
            'torque': round_array_values(torque),
            'theta': round_array_values(solution.y[0]),
            'omega': round_array_values(solution.y[1]),
        }
        # Save input and output data to a file if save_file is provided
        if save_file:
            if os.path.exists(save_file):
                # Append new data to existing file
                #with open(save_file, 'r') as file:
                #    existing_data = json.load(file)
                #existing_data.append(data_to_save)

                with open(save_file, 'a+') as file:
                    file.write('\n')
                    json.dump(data_to_save, file)
            else:
                # Create a new file if it doesn't exist
                with open(save_file, 'w') as file:
                    json.dump(data_to_save, file)
        return data_to_save

# Example usage
if __name__ == "__main__":
    pendulum = Pendulum(length=1.0, mass=1, gravity=9.81)

    # Simulation parameters
    initial_state = [0, 0]  # Initial state [theta, omega]
    simulation_time = (0, 10)  # Simulation time (start, end)
    
    # Define a time-dependent torque profile (e.g., torque increases linearly with time)
    def linear_torque(t):
        return 1.2*t
    def square_wave_torque(t, frequency=0.2, amplitude=1):
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    # Read data from the data file
    loaded_data = np.load('get_data_from_paper_figure/data.npy', allow_pickle=True).item()
    X = loaded_data['X']
    Y = loaded_data['Y']
    # Perform interpolation to smooth the curve
    f = interp1d(X, Y, kind='cubic', fill_value='extrapolate')
    f_linear = interp1d(X[-2:], Y[-2:], kind='linear')
    # Define your function my_torque(t)
    def my_torque(t):
        if t> X[-2]:
            return f_linear(t)-0.66
        else:
            return f(t)
    
    pendulum.set_torque_profile(my_torque)
    
    time_points = np.linspace(simulation_time[0], simulation_time[1], num=100)
    torque_data = pendulum.generate_torque_data(time_points, my_torque)
    #torque_data = pendulum.generate_random_torque_data()
    data = pendulum.run_simulation(torque_data, initial_state, simulation_time, save_file='mydata.json')
    pendulum.mytest(initial_state, simulation_time, step=0.001)
    pendulum.plot_simulation(data)
