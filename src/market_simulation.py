from datetime import date
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MarketSimulation:
    def __init__(
        self,
        simulation_start: date,
        simulation_end: date,
        power_volatility: float,
        coal_volatility: float,
        correlation: float,
        fwd_curve_power: pd.Series,
        fwd_curve_coal: pd.Series,
        sigma_ou_power: float,
        sigma_ou_coal: float,
        beta_power: float,
        beta_coal: float,
    ) -> None:
        """Initializes the MarketSimulation class.

        Args:
            simulation_start (date): Start date of the simulation.
            simulation_end (date): End date of the simulation.
            power_volatility (float): Volatility of the Power prices.
            coal_volatility (float): Volatility of the Coal prices.
            correlation (float): Correlation between the Power and Coal price shocks.
            fwd_curve_power (pd.Series): Forward curve for Power with monthly granularity.
            fwd_curve_coal (pd.Series): Forward curve for Coal with monthly granularity.
            sigma_ou_power (float): Volatility of the OU process for Power.
            sigma_ou_coal (float): Volatility of the OU process for Coal.
            beta_power (float): Mean reversion speed for the Power process.
            beta_coal (float): Mean reversion speed for the Coal process.
        """
        self.simulation_start = simulation_start
        self.simulation_end = simulation_end
        self.power_volatility = power_volatility
        self.coal_volatility = coal_volatility
        self.correlation = correlation
        self.sigma_ou_power = sigma_ou_power
        self.sigma_ou_coal = sigma_ou_coal
        self.beta_power = beta_power
        self.beta_coal = beta_coal
        self.simulation_days = pd.date_range(start=simulation_start, end=simulation_end, freq='D')
        self.fwd_curve_power = fwd_curve_power
        self.fwd_curve_coal = fwd_curve_coal

        self.num_days = len(self.simulation_days)

    def simulate(
        self, n_sims: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates simulated month-ahead prices for Power and Coal, and derived Spot Prices based on an OU process.

        Args:
            n_sims (int): Number of simulations to run.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Simulated Power, Coal, Spot Power, and Spot Coal prices.
        """
        dt = 1 / 365  # Daily time step assuming 365 days in a year

        # Generate correlated random shocks for all simulations
        cov_matrix = [[1.0, self.correlation], [self.correlation, 1.0]]
        dW_t = np.random.multivariate_normal(
            [0, 0], cov_matrix, size=(self.num_days - 1, n_sims)
        ) * np.sqrt(dt)

        # Initialize arrays for simulated prices
        month_ahead_power = np.zeros((self.num_days, n_sims))
        month_ahead_coal = np.zeros((self.num_days, n_sims))
        day_ahead_power = np.zeros((self.num_days, n_sims))
        day_ahead_coal = np.zeros((self.num_days, n_sims))
        spot_process_power = np.zeros((self.num_days, n_sims))
        spot_process_coal = np.zeros((self.num_days, n_sims))

        # Set initial values
        month_ahead_power[0, :] = 1.0
        month_ahead_coal[0, :] = 1.0

        # Simulate the month-ahead prices and spot prices
        for i_day in range(1, self.num_days):
            # Month-ahead price evolution (using Geometric Brownian Motion)
            month_ahead_power[i_day, :] = month_ahead_power[i_day - 1, :] + (
                month_ahead_power[i_day - 1, :] * self.power_volatility * dW_t[i_day - 1, :, 0]
            )
            month_ahead_coal[i_day, :] = month_ahead_coal[i_day - 1, :] + (
                month_ahead_coal[i_day - 1, :] * self.coal_volatility * dW_t[i_day - 1, :, 1]
            )

            # Apply mean-reverting OU process for Spot Prices
            spot_process_power[i_day, :] = (
                spot_process_power[i_day - 1, :]
                - self.beta_power * spot_process_power[i_day - 1, :] * dt
                + self.sigma_ou_power * np.sqrt(dt) * np.random.randn(n_sims)
            )

            spot_process_coal[i_day, :] = (
                spot_process_coal[i_day - 1, :]
                - self.beta_coal * spot_process_coal[i_day - 1, :] * dt
                + self.sigma_ou_coal * np.sqrt(dt) * np.random.randn(n_sims)
            )

        # Convert to DataFrame
        month_ahead_power = pd.DataFrame(data=month_ahead_power, index=self.simulation_days)
        month_ahead_coal = pd.DataFrame(data=month_ahead_coal, index=self.simulation_days)
        spot_process_power = pd.DataFrame(data=spot_process_power, index=self.simulation_days)
        spot_process_coal = pd.DataFrame(data=spot_process_coal, index=self.simulation_days)

        # Apply forward curves
        power_curve_mapping = self.fwd_curve_power.reindex(
            self.simulation_days.to_period('M')
        ).values
        coal_curve_mapping = self.fwd_curve_coal.reindex(
            self.simulation_days.to_period('M')
        ).values
        month_ahead_power = month_ahead_power.multiply(power_curve_mapping, axis=0)
        month_ahead_coal = month_ahead_coal.multiply(coal_curve_mapping, axis=0)

        day_ahead_power = np.exp(spot_process_power).mul(month_ahead_power)
        day_ahead_coal = np.exp(spot_process_coal).mul(month_ahead_coal)

        t = np.arange(self.num_days) * dt

        drift_correction_power = np.exp(
            -self.sigma_ou_power**2
            / (4.0 * self.beta_power)
            * (1.0 - np.exp(-2.0 * self.beta_power * t))
        )
        drift_correction_coal = np.exp(
            -self.sigma_ou_coal**2
            / (4.0 * self.beta_coal)
            * (1.0 - np.exp(-2.0 * self.beta_coal * t))
        )

        day_ahead_power = day_ahead_power.mul(drift_correction_power, axis=0)
        day_ahead_coal = day_ahead_coal.mul(drift_correction_coal, axis=0)

        return month_ahead_power, month_ahead_coal, day_ahead_power, day_ahead_coal

    def plot_simulations(
        self,
        month_ahead_power: pd.DataFrame,
        month_ahead_coal: pd.DataFrame,
        day_ahead_power: pd.DataFrame,
        day_ahead_coal: pd.DataFrame,
    ) -> None:
        """Plots the row-wise mean of the simulations along with the forward curve for Power and Coal.

        Args:
            month_ahead_power (pd.DataFrame): Simulated Power prices.
            month_ahead_coal (pd.DataFrame): Simulated Coal prices.
            spot_prices_power (pd.DataFrame): Simulated Spot Power prices.
            spot_prices_coal (pd.DataFrame): Simulated Spot Coal prices.
        """
        # Calculate the mean of the simulations
        power_mean = month_ahead_power.mean(axis=1)
        coal_mean = month_ahead_coal.mean(axis=1)
        day_ahead_power_mean = day_ahead_power.mean(axis=1)
        day_ahead_coal_mean = day_ahead_coal.mean(axis=1)

        # Plot for Power (Month Ahead and Spot)
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.simulation_days, power_mean, label='Mean Simulated Power Price', color='blue'
        )
        plt.plot(
            self.fwd_curve_power.index.to_timestamp(),
            self.fwd_curve_power.values,
            label='Forward Curve Power',
            color='orange',
            linestyle='--',
        )
        plt.plot(
            self.simulation_days,
            day_ahead_power_mean,
            label='Mean Simulated Spot Power Price',
            color='green',
            linestyle='--',
        )
        plt.title('Power Prices Simulation')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

        # Plot for Coal (Month Ahead and Spot)
        plt.figure(figsize=(10, 5))
        plt.plot(self.simulation_days, coal_mean, label='Mean Simulated Coal Price', color='green')
        plt.plot(
            self.fwd_curve_coal.index.to_timestamp(),
            self.fwd_curve_coal.values,
            label='Forward Curve Coal',
            color='red',
            linestyle='--',
        )
        plt.plot(
            self.simulation_days,
            day_ahead_coal_mean,
            label='Mean Simulated Spot Coal Price',
            color='purple',
            linestyle='--',
        )
        plt.title('Coal Prices Simulation')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_empirical_correlation(
        self, month_ahead_power: pd.DataFrame, month_ahead_coal: pd.DataFrame
    ) -> None:
        """Plots the empirical correlation between the log returns of Power and Coal simulations over time,
        and adds a horizontal line for the input correlation.

        Args:
            month_ahead_power (pd.DataFrame): Simulated Power prices.
            month_ahead_coal (pd.DataFrame): Simulated Coal prices.
        """
        # Calculate log returns for Power and Coal
        log_returns_power = np.log(month_ahead_power / month_ahead_power.shift(1))
        log_returns_coal = np.log(month_ahead_coal / month_ahead_coal.shift(1))

        # Calculate correlation between the log returns for each day
        correlations = log_returns_power.corrwith(log_returns_coal, axis=1)

        # Plot the correlation over time
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.simulation_days,
            correlations.values,
            label='Empirical Log Return Correlation',
            color='purple',
        )
        plt.axhline(
            y=self.correlation,
            color='red',
            linestyle='--',
            label=f'Input Correlation ({self.correlation})',
        )
        plt.title('Empirical Correlation of Log Returns between Power and Coal')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_first_5_paths(
        self,
        month_ahead_power: pd.DataFrame,
        month_ahead_coal: pd.DataFrame,
        spot_prices_power: pd.DataFrame,
        spot_prices_coal: pd.DataFrame,
    ) -> None:
        """Plots the first 5 simulation paths for Power and Coal, both for Month-Ahead and Spot prices.

        Args:
            month_ahead_power (pd.DataFrame): Simulated Power prices.
            month_ahead_coal (pd.DataFrame): Simulated Coal prices.
            spot_prices_power (pd.DataFrame): Simulated Spot Power prices.
            spot_prices_coal (pd.DataFrame): Simulated Spot Coal prices.
        """
        # Ensure we're selecting the first 5 columns
        power_paths = month_ahead_power.iloc[:, :5]
        coal_paths = month_ahead_coal.iloc[:, :5]
        spot_power_paths = spot_prices_power.iloc[:, :5]
        spot_coal_paths = spot_prices_coal.iloc[:, :5]

        # Create the plot for Power
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot Month-Ahead Power
        axes[0].plot(power_paths.index, power_paths.values, color='blue', alpha=0.6)
        axes[0].set_title('Power - Month-Ahead Price Simulations')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price')
        axes[0].grid()

        # Plot Spot Power
        axes[1].plot(spot_power_paths.index, spot_power_paths.values, color='green', alpha=0.6)
        axes[1].set_title('Power - Spot Price Simulations')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Price')
        axes[1].grid()

        plt.tight_layout()
        plt.show()

        # Create the plot for Coal
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot Month-Ahead Coal
        axes[0].plot(coal_paths.index, coal_paths.values, color='red', alpha=0.6)
        axes[0].set_title('Coal - Month-Ahead Price Simulations')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price')
        axes[0].grid()

        # Plot Spot Coal
        axes[1].plot(spot_coal_paths.index, spot_coal_paths.values, color='purple', alpha=0.6)
        axes[1].set_title('Coal - Spot Price Simulations')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Price')
        axes[1].grid()

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Simulation parameters
    simulation_start = date(2024, 1, 1)
    simulation_end = date(2024, 12, 31)
    power_volatility = 0.2  # 20% annualized volatility
    coal_volatility = 0.15  # 15% annualized volatility
    correlation = 0.7  # Positive correlation

    # Mean-reverting process parameters
    sigma_ou_power = 0.5
    sigma_ou_coal = 0.8
    beta_power = 2.1
    beta_coal = 3.0

    # Forward curves
    fwd_curve_power = pd.Series(
        data=[50, 52, 53, 55, 57, 58, 60, 61, 63, 64, 66, 67],
        index=pd.period_range(start="2024-01", end="2024-12", freq="M"),
    )
    fwd_curve_coal = pd.Series(
        data=[80, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68],
        index=pd.period_range(start="2024-01", end="2024-12", freq="M"),
    )

    # Number of simulations
    n_sims = 5000

    # Create and run the simulation
    simulator = MarketSimulation(
        simulation_start,
        simulation_end,
        power_volatility,
        coal_volatility,
        correlation,
        fwd_curve_power,
        fwd_curve_coal,
        sigma_ou_power,
        sigma_ou_coal,
        beta_power,
        beta_coal,
    )
    (
        month_ahead_power,
        month_ahead_coal,
        spot_prices_power,
        spot_prices_coal,
    ) = simulator.simulate(n_sims)

    # Plot the results
    simulator.plot_simulations(
        month_ahead_power, month_ahead_coal, spot_prices_power, spot_prices_coal
    )

    # Plot the empirical correlation of log returns
    simulator.plot_empirical_correlation(month_ahead_power, month_ahead_coal)

    # Plot the first 5 paths for Power and Coal
    simulator.plot_first_5_paths(
        month_ahead_power, month_ahead_coal, spot_prices_power, spot_prices_coal
    )
