#Steps:
#User inputs the ticker of the stock, N bins, and the number of iterations
#Download the data from Yahoo Finance
#Calculate a series of log returns
#Discretize the log returns into the N bins
#Calculate the transition matrix on the first four years 
#Run through the iterations of the Markov Chain for the fifth year
#Create a histogram of the results
#Add the actual return at the end of the fifth year on the histogram

import yfinance as yf
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def fetch_log_returns(ticker):
    period = "5y"
    interval = "1d" 
    data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True, actions=False)

    data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    split_index = int(len(data) * 0.8)
    
    data_first_part = data.iloc[1:split_index]
    data_second_part = data.iloc[split_index:]
    sum_log_returns_second_part = data_second_part['Log Returns'].sum()

    close_start_of_second_part = data_second_part['Close'].iloc[0]  
    close_end_of_second_part = data_second_part['Close'].iloc[-1]
    final_log_return = np.log(close_end_of_second_part / close_start_of_second_part)
    
    return data_first_part['Log Returns'], final_log_return, close_start_of_second_part, close_end_of_second_part

def calculate_evenly_filled_bin_edges(data, n_bins):
    data_sorted = np.sort(data) 
    quantiles = np.linspace(0, 1, n_bins + 1)
    indices = np.ceil(quantiles * (len(data) - 1)).astype(int)
    bin_edges = data_sorted[indices]
    return bin_edges

def calculate_evenly_spaced_bin_edges(data, n_bins):
    data_min = np.min(data)
    data_max = np.max(data)
    bin_edges = np.linspace(data_min, data_max, n_bins + 1)
    return bin_edges

def calculate_transition_matrix(data, bin_edges):
    n_bins = len(bin_edges) - 1 
    bin_indices = np.digitize(data, bin_edges, right=True)
    transition_matrix = np.zeros((n_bins, n_bins))
    for (i, j) in zip(bin_indices[:-1], bin_indices[1:]):
        transition_matrix[i-1][j-1] += 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    return transition_matrix

def run_monte_carlo_simulations_from_last_return(log_returns, num_iterations, num_steps, transition_matrix, bin_edges):
    n_bins = len(bin_edges) - 1
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate midpoints of bins for log returns
    
    # Determine the initial state from the last entry in log_returns
    last_return = log_returns[-1]
    initial_state = np.digitize([last_return], bin_edges, right=True)[0] - 1
    
    results = []
    
    for _ in tqdm(range(num_iterations)):
        state = initial_state
        total_log_return = 0.0
        
        for _ in range(int(num_steps)):
            # Draw the next state based on the transition matrix probabilities of the current state
            state = np.random.choice(n_bins, p=transition_matrix[state])
            # Accumulate the log return using the midpoint of the bin corresponding to the new state
            total_log_return += bin_midpoints[state]
        
        results.append(total_log_return)
    
    return results

if __name__ == "__main__":
    print("Enter the ticker of the stock")
    ticker = input()
    print("Enter the number of bins")
    N = int(input())
    print("Enter the number of iterations")
    iterations = int(input())

    log_returns, last_year_log_return, close1, close2 = fetch_log_returns(ticker)
    #bin_edges = calculate_evenly_filled_bin_edges(log_returns, N)
    bin_edges = calculate_evenly_spaced_bin_edges(log_returns, N)
    transition_matrix = calculate_transition_matrix(log_returns, bin_edges)
    results = run_monte_carlo_simulations_from_last_return(log_returns, iterations, len(log_returns)*0.25, transition_matrix, bin_edges)
    plt.hist(results, bins=30, alpha=0.75, label='Simulated Returns')

    # Add a vertical line for the true last year return
    plt.axvline(last_year_log_return, color='red', linewidth=2, label='True Last Year Return')
    mean_result = np.mean(results)
    median_result = np.median(results)
    plt.axvline(mean_result, color='green', linestyle='dotted', linewidth=2, label='Mean of Simulated Returns')
    plt.axvline(median_result, color='green', linestyle='dashed', linewidth=2, label='Median of Simulated Returns')
    txt = f'{ticker} was {close1:.3f} a year ago and is {close2:.3f} now. The true return is {last_year_log_return:.3f}. The mean of the simulated returns is {mean_result:.3f} and the median is {median_result:.3f}'
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8)
    # Adding labels and title
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated Log Returns')
    plt.legend()

    # Display the plot
    plt.show()
