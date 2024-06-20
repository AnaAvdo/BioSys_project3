import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Initial values for the model
N = 100000  # Total population
H = 0.1*N # Hospital capacity
initial_conditions = [N - 1, 1, 0, 0, 0]  # [S0, E0, I0, R0, D0]
prob_of_infecting = 1/200
avg_no_contacts_per_individual = 50
beta_true = prob_of_infecting * avg_no_contacts_per_individual / N
gamma_true = 1/14  # Recovery rate
sigma_true = 1/7  # Incubation rate
rho_true = 1/60  # Immunity waning rate
delta_true = 0.01  # Death rate
seasonal_amplitude = 0.2  # Amplitude for seasonal variation
seasonal_period = 365.25  # Period for seasonal variation

def sir_model(t, y, beta, gamma, sigma, rho, delta):
    """
    SIR model equations
    """
    S, E, I, R, D = y
    dSdt = -beta * S * I + rho * R  # Susceptible individuals decreasing due to infection and increasing due to recovery
    dEdt = beta * S * I - sigma * E # Exposed individuals becoming infected
    dIdt = sigma * E - gamma * I - delta * I  # Infected individuals recovering or dying
    dRdt = gamma * I - rho * R  # Recovered individuals increasing and returning to susceptible state
    dDdt = delta * I  # Deceased individuals increasing due to death rate

    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def generate_data(beta, gamma, sigma, rho, delta, initial_conditions, t_points, t_start, t_end, add_noise=True):
    """
    Generate simulated data based on SIR model with immunity waning
    """
    solution = solve_ivp(sir_model, [t_start, t_end], initial_conditions, 
                         args=(beta, gamma, sigma, rho, delta), t_eval=t_points)
    S_data, E_data, I_data, R_data, D_data = solution.y
    
    if add_noise:
        # Increase variance of noise and ensure data doesn't go below 0
        noise_S = np.random.normal(0, 10, len(S_data))  
        S_data += noise_S
        S_data = np.maximum(S_data, 0) 

        noise_E = np.random.normal(0, 60, len(E_data))  
        E_data += noise_E
        E_data = np.maximum(E_data, 0)  

        noise_I = np.random.normal(0, 600, len(I_data))  
        I_data += noise_I
        I_data = np.maximum(I_data, 0)  

        noise_R = np.random.normal(0, 300, len(R_data))
        R_data += noise_R
        R_data = np.maximum(R_data, 0)  

        noise_D = np.random.normal(0, 10, len(D_data))  
        D_data += noise_D
        D_data = np.maximum(D_data, 0)  

    # Round infected values to integers
    S_data = np.round(S_data).astype(int)
    E_data = np.round(E_data).astype(int)
    I_data = np.round(I_data).astype(int)
    R_data = np.round(R_data).astype(int)
    D_data = np.round(D_data).astype(int)

    # Store data in a DataFrame
    data = pd.DataFrame({
        'Time': t_points,
        'Susceptible': S_data,
        'Exposed': E_data,
        'Infected': I_data,
        'Recovered': R_data,
        'Deceased': D_data
    })

    # Save DataFrame to CSV
    data.to_csv('simulated_data_seasonal.csv', index=False)

    return S_data, E_data, I_data, R_data, D_data

def loss_function(params):
    """
    Loss function to minimize during parameter estimation
    """
    beta, gamma, sigma, rho, delta = params
    t_start = 0
    t_end = 360
    t_step = 1
    t_points = np.arange(t_start, t_end, t_step)
    _, _, I_data, _, _ = generate_data(beta, gamma, sigma, rho, delta, initial_conditions, t_points, t_start, t_end, add_noise=False)
    peak_infected = np.max(I_data)
    error = (peak_infected - H) ** 2  # Ensure the peak of infections does not exceed hospital capacity
    return error

def seasonal_beta(t, base_beta, amplitude, period):
    """
    Function to model seasonal variation in transmission rate (beta)
    """
    return base_beta * (1 + amplitude * np.sin(2 * np.pi * t / period))

# Scenario 1: Hospital Capacity
def scenario_hospital_capacity():
    # Time points
    t_start = 0
    t_end = 360
    t_step = 1
    t_points = np.arange(t_start, t_end, t_step)

    _, _, I_observed, _, _ = generate_data(beta_true, gamma_true, sigma_true, rho_true, delta_true, initial_conditions, t_points, t_start, t_end, t_step)
    beta_guess = beta_true
    gamma_guess = 1/20
    sigma_guess = 1/10
    rho_guess = 1/120
    initial_guess = [beta_guess, gamma_guess, sigma_guess, rho_guess, delta_true]

    result = minimize(loss_function, initial_guess, method='Nelder-Mead')
    estimated_params = result.x
    beta_estimated, gamma_estimated, sigma_estimated, rho_estimated, delta_estimated = estimated_params
    _, _, I_estimated, _, _ = generate_data(beta_estimated, gamma_estimated, sigma_estimated, rho_estimated, delta_estimated, initial_conditions, t_points, t_start, t_end,add_noise=False)
    plt.plot(t_points, I_observed, label='Infected (Observed)')
    plt.plot(t_points, I_estimated, '--', label='Infected (Estimated)')
    #plt.plot(t_points, S_estimated, '--', label='Susceptible (Estimated)')
    #plt.plot(t_points, E_estimated, '--', label='Exposed (Estimated)')
    #plt.plot(t_points, R_observed, label='Recovered (Observed)')
    #plt.plot(t_points, R_estimated, '--', label='Recovered (Estimated)')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.axhline(y=H, color='r', linestyle='-', label='Hospital Capacity')
    plt.title('SIR Model Parameter Estimation - Hospital Capacity Scenario')
    plt.legend()

    plt.show()
    print(f"True Parameters: beta = {beta_true}, gamma = {gamma_true}, sigma = {sigma_true}, rho = {rho_true}, delta = {delta_true}")
    print(f"Estimated Parameters: beta = {beta_estimated}, gamma = {gamma_estimated}, sigma = {sigma_estimated}, rho = {rho_estimated}, delta = {delta_estimated}")

# Scenario 2: Seasonal Variation
def scenario_seasonal_variation():
    initial_conditions = [N - 1, 1, 0, 0, 0]
    t_points_interval = np.arange(0, 90, 1)
    t_start = 0
    t_end = 360
    t_step = 90
    S_results = []
    E_results = []
    I_results = []
    R_results = []
    D_results = []
    betas = []

    for i in range(t_start, t_end, t_step):
        beta_cur = seasonal_beta(i+t_step, beta_true, seasonal_amplitude, seasonal_period)
        betas.append(beta_cur)
        S_data, E_data, I_data, R_data, D_data = generate_data(beta_cur, gamma_true, sigma_true, rho_true, delta_true, initial_conditions, t_points_interval, 0, 90, add_noise=True)
        
        S_results.extend(S_data)
        E_results.extend(E_data)
        I_results.extend(I_data)
        R_results.extend(R_data)
        D_results.extend(D_data)
        
        # Update initial conditions for the next time interval
        initial_conditions = [S_data[-1], E_data[-1], I_data[-1], R_data[-1], D_data[-1]]

    print('Betas for seasons: ', betas)
    plt.plot(np.arange(t_start, t_end, 1), I_results, label='Infected with Seasonal Variation')
    plt.plot(np.arange(t_start, t_end, 1), S_results, '--', label='Susceptible')
    plt.plot(np.arange(t_start, t_end, 1), E_results, '--', label='Exposed')
    plt.plot(np.arange(t_start, t_end, 1), R_results, label='Recovered')
    plt.plot(np.arange(t_start, t_end, 1), D_results, '--', label='Deceased')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model with Seasonal Variation in Transmission Rate')
    plt.legend()
    plt.show()


def sir_variable_gamma_model(t, y, beta, sigma, rho, delta, gamma_mean, gamma_std):
    S, E, I, R, D = y
    
    # Gamma for each infected individual follows a normal distribution
    gamma_t = np.random.normal(gamma_mean, gamma_std)
    
    # Ensure gamma_t is positive
    if gamma_t < 0:
        gamma_t = gamma_mean
    
    dSdt = -beta * S * I + rho * R
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma_t * I - delta * I
    dRdt = gamma_t * I - rho * R
    dDdt = delta * I
    
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def generate_variable_gamma_data(beta, sigma, rho, delta, gamma_mean, gamma_std, initial_conditions, t_points):
    solution = solve_ivp(sir_variable_gamma_model, [t_points[0], t_points[-1]], initial_conditions, 
                         args=(beta, sigma, rho, delta, gamma_mean, gamma_std), t_eval=t_points)
    S_data, E_data, I_data, R_data, D_data = solution.y

    # Round values to integers
    S_data = np.round(S_data).astype(int)
    E_data = np.round(E_data).astype(int)
    I_data = np.round(I_data).astype(int)
    R_data = np.round(R_data).astype(int)
    D_data = np.round(D_data).astype(int)

    return S_data, E_data, I_data, R_data, D_data

# Scenario 3: Severity and Mortality
def scenario_severity(gamma_mean, gamma_std):
    initial_conditions = [N - 1, 1, 0, 0, 0]
    t_start = 0
    t_end = 360
    t_step = 1
    t_points = np.arange(t_start, t_end, t_step)

    S_data, E_data, I_data, R_data, D_data = generate_variable_gamma_data(
        beta_true, sigma_true, rho_true, delta_true, gamma_mean, gamma_std, initial_conditions, t_points)
    
    plt.plot(t_points, I_data, label='Infected with Variable Gamma')
    plt.plot(t_points, S_data, '--', label='Susceptible')
    plt.plot(t_points, E_data, '--', label='Exposed')
    plt.plot(t_points, R_data, label='Recovered')
    plt.plot(t_points, D_data, '--', label='Deceased')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model with Variable Recovery Rate (Gamma)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    scenario_hospital_capacity()
    scenario_seasonal_variation()
    # recovery ~ 14 days
    scenario_severity(0.071, 0.01)
    scenario_severity(0.071, 0.05)
    # recovery ~ 10 days
    scenario_severity(0.1, 0.01)
    scenario_severity(0.1, 0.05)
    # recovery ~ 7 days
    scenario_severity(0.143, 0.01)
    scenario_severity(0.143, 0.05)