from datetime import date
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from market_simulation import MarketSimulation


class OperationalState(Enum):
    """Enum for the operational states of the power plant."""

    IDLE = 0  # The plant is idle (not generating power).
    RAMPING_UP = 1  # The plant is ramping up to generate power.
    RAMPING_DOWN = 2  # The plant is ramping down from power generation.
    RUNNING = 3  # The plant is actively generating power.


class OptimalControl(Enum):
    """Enum for the optimal control actions that the plant can take."""

    RAMPING_UP = 1  # Represents the ramping up action
    RAMPING_DOWN = 2  # Represents the ramping down action
    DO_NOTHING = 3  # Represents the action where no change occurs (idle state)


def get_next_state(
    current_optimal_state: OperationalState, optimal_control: OptimalControl
) -> OperationalState:
    """
    Get the next operational state based on the current state and optimal control decision.

    Args:
        current_optimal_state (OperationalState): The current state of the power plant.
        optimal_control (OptimalControl): The optimal control decision for the next step.

    Returns:
        OperationalState: The next operational state after applying the optimal control decision.
    """
    if current_optimal_state == OperationalState.IDLE:
        if optimal_control == OptimalControl.RAMPING_UP:
            return OperationalState.RAMPING_UP
        else:
            return OperationalState.IDLE  # Do Nothing remains in IDLE

    elif current_optimal_state == OperationalState.RAMPING_UP:
        return OperationalState.RUNNING

    elif current_optimal_state == OperationalState.RUNNING:
        if optimal_control == OptimalControl.RAMPING_DOWN:
            return OperationalState.RAMPING_DOWN
        else:  # DO_NOTHING
            return OperationalState.RUNNING  # Stay running

    elif current_optimal_state == OperationalState.RAMPING_DOWN:
        return OperationalState.IDLE


class PowerPlant:
    """
    Class representing a power plant with various operational states, control decisions, and optimization.

    Attributes:
        operation_costs (float): Fixed operational costs of the power plant.
        alpha (float): Coefficient for operational cost factor.
        ramping_up_costs (float): Costs associated with ramping up operations.
        ramping_down_costs (float): Costs associated with ramping down operations.
        idle_costs (float): Costs associated with being idle.
        market_simulation (MarketSimulation): Instance of MarketSimulation class for market simulation.
        cashflows (np.ndarray): Cash flows for different operational states (idle, ramping up, running, ramping down).
        value (np.ndarray): Values associated with the cashflows for each simulation.
        optimal_control_decision (np.ndarray): Array storing the optimal control decisions (RAMPING_UP, RAMPING_DOWN, DO_NOTHING).
        optimal_value (np.ndarray): Optimal value for each simulation day.
    """

    def __init__(
        self,
        operation_costs: float,
        alpha: float,
        ramping_up_costs: float,
        ramping_down_costs: float,
        n_days_rampgin_up: int,
        n_days_rampgin_down: int,
        idle_costs: float,
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
        k: int = 3,  # Default polynomial degree
    ) -> None:
        """Initializes the PowerPlant class, which internally creates an instance of the MarketSimulation class with the
        provided operational and market simulation parameters.

        Args:
            operation_costs (float): Fixed operational costs of the power plant.
            alpha (float): Coefficient for operational cost factor.
            ramping_up_costs (float): Costs associated with ramping up operations.
            ramping_down_costs (float): Costs associated with ramping down operations.
            idle_costs (float): Costs associated with being idle.
            simulation_start (date): Start date for the market simulation.
            simulation_end (date): End date for the market simulation.
            power_volatility (float): Volatility of the Power prices.
            coal_volatility (float): Volatility of the Coal prices.
            correlation (float): Correlation between the Power and Coal price shocks.
            fwd_curve_power (pd.Series): Forward curve for Power with monthly granularity.
            fwd_curve_coal (pd.Series): Forward curve for Coal with monthly granularity.
            sigma_ou_power (float): Volatility for the mean-reverting Ornstein-Uhlenbeck process for power.
            sigma_ou_coal (float): Volatility for the mean-reverting Ornstein-Uhlenbeck process for coal.
            beta_power (float): Mean-reversion speed for the power price process.
            beta_coal (float): Mean-reversion speed for the coal price process.
            k (int): Polynomial degree for regression (default is 3).
        """
        self.operation_costs = operation_costs
        self.alpha = alpha
        self.ramping_up_costs = ramping_up_costs
        self.ramping_down_costs = ramping_down_costs
        self.n_days_rampgin_up = n_days_rampgin_up
        self.n_days_rampgin_down = n_days_rampgin_down
        self.idle_costs = idle_costs

        self.k = k

        # Initialize the MarketSimulation object with the provided parameters
        self.market_simulation = MarketSimulation(
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

    def _initialize_terminal_state(self) -> None:
        """Initializes the terminal state (end of simulation), setting the cashflows and values for the last day of simulation.
        This function assumes that the last day corresponds to an idle, ramping up, ramping down, or running state.

        Modifies:
            self.cashflows: Updates the last simulation day cashflows for different states.
            self.value: Updates the last simulation day value for each state.
        """
        # 0: idle state
        self.cashflows[-1, :, 0] = -self.idle_costs
        # 1: ramping up
        self.cashflows[-1, :, 1] = -self.ramping_up_costs
        # 2: ramping down
        self.cashflows[-1, :, 2] = -self.ramping_down_costs
        # 3: running
        self.cashflows[-1, :, 3] = -self.operation_costs + self.alpha * self.spread.iloc[-1, :]

        self.value[-1, :, :] = self.cashflows[-1, :, :]

    def _regress(self, i_day: int, j_state: int, plot: bool = False) -> None:
        """Regresses the optimal value for the given day and state against the polynomial features of power and coal prices.

        Args:
            i_day (int): The index of the day (i.e., row in the simulation data).
            j_state (int): The state index (0: idle, 1: ramping_up, 2: ramping_down, 3: running).
            plot (bool): If True, plots the actual vs predicted values for visual inspection.
        """
        # Extract the day-ahead power and coal prices for the given day
        day_ahead_power = self.day_ahead_power.iloc[i_day, :]
        day_ahead_coal = self.day_ahead_coal.iloc[i_day, :]

        # Stack the power and coal prices together as feature matrix X
        X = np.vstack([day_ahead_power.values, day_ahead_coal.values]).T

        # Generate polynomial features up to degree k (default is 3)
        poly = PolynomialFeatures(degree=self.k)
        X_poly = poly.fit_transform(X)

        # Extract the optimal value for the given day and state
        y = self.value[i_day + 1, :, j_state]

        # Perform the regression
        model = LinearRegression()
        model.fit(X_poly, y)

        # Get the R^2 score (goodness of fit)
        r2_score = model.score(X_poly, y)
        print(f"R² Score for day {i_day} and state {j_state}: {r2_score:.4f}")

        # Update the optimal value using the regression coefficients
        regressed_value = model.predict(X_poly)

        # Plotting if requested
        if plot:
            self._plot_3d_regression(X, y, model, X_poly, poly, i_day, j_state)

        # Optionally, return the model or coefficients if you need them
        return regressed_value

    def _plot_3d_regression(self, X, y, model, X_poly, poly, i_day, j_state):
        """Plots the actual vs predicted values and the regression surface in 3D."""
        # Predict the optimal values using the model
        y_pred = model.predict(X_poly)

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of actual values
        ax.scatter(X[:, 0], X[:, 1], y, label="Actual Values", color='blue', alpha=0.6)

        # Scatter plot of predicted values
        ax.scatter(X[:, 0], X[:, 1], y_pred, label="Predicted Values", color='red', alpha=0.6)

        # Create a grid for plotting the regression surface
        x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = model.predict(
            poly.transform(np.vstack([X_grid.ravel(), Y_grid.ravel()]).T)
        ).reshape(X_grid.shape)

        # Plot the regression surface
        ax.plot_surface(X_grid, Y_grid, Z_grid, color='gray', alpha=0.3)

        # Add labels and title
        ax.set_xlabel('Day Ahead Power')
        ax.set_ylabel('Day Ahead Coal')
        ax.set_zlabel('Optimal Value')
        ax.set_title(f"3D Regression: Day {i_day}, State {j_state}")

        # Show the legend
        ax.legend()

        # Show the plot
        plt.show()

    def _optimize(self, i_day: int):
        regressed_value = np.zeros((self.market_simulation.num_days, n_sims, 4))

        for j_state in range(4):
            regressed_value[i_day, :, j_state] = self._regress(i_day, j_state)

        # 0: idle state
        continuation_value = regressed_value[i_day, :, 0]
        exercise_value = regressed_value[i_day, :, 1]

        self.cashflows[i_day, :, 0] = -self.idle_costs

        self.value[i_day, :, 0] = -self.idle_costs + np.maximum(
            continuation_value, exercise_value
        )
        # Set the optimal control decision based on the comparison of continuation_value and exercise_value
        self.optimal_control_decision[i_day, :, 0] = np.where(
            continuation_value > exercise_value,
            OptimalControl.DO_NOTHING,  # If continuation is greater, do nothing
            OptimalControl.RAMPING_UP,  # Otherwise, ramp up
        )

        # 1: ramping up
        # no decision to be made
        self.cashflows[i_day,: ,1] = - self.ramping_up_costs
        self.value[i_day, :, 1] = - self.ramping_up_costs + regressed_value[i_day, :, 3]

        # 2: ramping down
        # no decision to be made
        self.cashflows[i_day,: ,2] = - self.ramping_down_costs
        self.value[i_day, :, 2] =  - self.ramping_down_costs + regressed_value[i_day, :, 0]

        # 3: running
        continuation_value = regressed_value[i_day, :, 3]
        exercise_value =  regressed_value[i_day, :, 2]

        self.cashflows[i_day, :, 3] = -self.operation_costs + self.alpha*self.spread.iloc[i_day]

        self.value[i_day, :, 3] = -self.operation_costs + self.alpha*self.spread.iloc[i_day]+ np.maximum(
            continuation_value, exercise_value
        )

        # Set the optimal control decision based on the comparison of continuation_value and exercise_value
        self.optimal_control_decision[i_day, :, 3] = np.where(
            continuation_value > exercise_value,
            OptimalControl.DO_NOTHING,  # If continuation is greater, do nothing
            OptimalControl.RAMPING_DOWN,  # Otherwise, ramp down
        )

    def optimize(self, n_sims: int) -> None:
        """Optimizes the plant operations over a specified number of simulation days and simulations.
        It creates the necessary data arrays for storing cashflows, values, control decisions, and optimizes the power plant's operation.

        Args:
            n_sims (int): The number of simulations to run.

        Modifies:
            self.cashflows (np.ndarray): Cashflows for each state and simulation.
            self.value (np.ndarray): Values for each state and simulation.
            self.optimal_control_decision (np.ndarray): Optimal control decisions for each simulation and day.
            self.optimal_value (np.ndarray): Optimal values for each simulation day.
            self.spread (pd.Series): The spread between day-ahead power and coal prices.
        """
        # Create xarray DataArrays
        self.cashflows = np.zeros((self.market_simulation.num_days, n_sims, 4))
        self.optimal_control_decision = np.full(
            (self.market_simulation.num_days - 1, n_sims, 4),
            OptimalControl.DO_NOTHING,
            dtype=OptimalControl,
        )
        self.value = np.zeros((self.market_simulation.num_days, n_sims, 4))

        # Run the market simulation for day-ahead power and coal prices
        _, _, self.day_ahead_power, self.day_ahead_coal = self.market_simulation.simulate(n_sims)

        # Compute the spread between power and coal prices
        self.spread = self.day_ahead_power.sub(self.day_ahead_coal)

        # Initialize the terminal state (end of simulation)
        self._initialize_terminal_state()

        for i_day in range(self.market_simulation.num_days - 2, -1, -1):
            self._optimize(i_day)

    def value_along_path(self, initial_state: OperationalState, simulation_path: int):
        value_along_path = np.zeros(self.market_simulation.num_days)
        cashflow_along_path = np.zeros(self.market_simulation.num_days)
        value_along_path[0] = self.value[0, simulation_path, initial_state.value]
        cashflow_along_path[0] =self.cashflows[0, simulation_path, initial_state.value]

        optimal_states_along_path = np.full(
            self.market_simulation.num_days,
            initial_state,
            dtype=OperationalState,
        )

        optimal_control_along_path = np.full(
            self.market_simulation.num_days - 1,
            OptimalControl.DO_NOTHING,
            dtype=OptimalControl,
        )

        for i_day in range(1, self.market_simulation.num_days):
            current_optimal_state = optimal_states_along_path[i_day - 1]


            optimal_control = self.optimal_control_decision[
                i_day - 1, simulation_path, current_optimal_state.value 
            ]
            optimal_control_along_path[i_day - 1] = optimal_control

            

            optimal_state = get_next_state(current_optimal_state, optimal_control)

            optimal_states_along_path[i_day] = optimal_state

            cashflow_along_path[i_day]=self.cashflows[
                i_day, simulation_path, optimal_state.value
            ]
            value_along_path[i_day] = self.value[
                i_day, simulation_path, optimal_state.value
            ]

        return value_along_path, optimal_states_along_path, optimal_control_along_path, cashflow_along_path

    def plot_optimization_results(
        self, initial_state: OperationalState, simulation_path: int
    ) -> None:
        """Plots the spot prices, optimal values, control decisions, and states along a given simulation path.

        Args:
            initial_state (OperationalState): The initial state to start from in the simulation.
            simulation_path (int): The index of the simulation path to plot.
        """
        # Retrieve the spot prices (power and coal) along the simulation path
        power_prices = self.day_ahead_power.iloc[:, simulation_path]
        coal_prices = self.day_ahead_coal.iloc[:, simulation_path]

        # Get the optimal value and control decisions along the path
        (
            value_along_path,
            optimal_states_along_path,
            optimal_control_along_path,
            cashflow_along_path
        ) = self.value_along_path(initial_state, simulation_path)

        # Create the plot figure with four subplots
        fig, axes = plt.subplots(5, 1, figsize=(12, 20))

        # Plot the spot prices (Power and Coal)
        axes[0].plot(power_prices.index, power_prices.values, label="Power Price", color='blue')
        axes[0].plot(coal_prices.index, coal_prices.values, label="Coal Price", color='orange')
        axes[0].set_title("Spot Prices along Simulation Path")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Price")
        axes[0].legend()

        # Plot the optimal value along the simulation path
        axes[1].plot(
            power_prices.index, value_along_path, label="Optimal Value", color='green'
        )
        axes[1].set_title("Optimal Value along Simulation Path")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Optimal Value")

        # Plot the optimal value along the simulation path
        axes[2].plot(
            power_prices.index, cashflow_along_path, label="Cashflow along path", color='red'
        )

        # Plot the control decisions along the simulation path as markers
        has_ramping_up_label_written = False
        has_ramping_down_label_written =False
        has_idle_label_written = False
        has_do_nothing_label_written = False
        
        for i, control in enumerate(optimal_control_along_path):
            if control == OptimalControl.RAMPING_UP:

                if not has_ramping_up_label_written:
                    has_ramping_up_label_written = True
                    axes[3].scatter(
                            power_prices.index[i],
                            value_along_path[i],
                            color='green',
                            marker='x',
                            label="Ramping Up",
                            s=30.0
                        )
                else:
                    axes[3].scatter(
                            power_prices.index[i],
                            value_along_path[i],
                            color='green',
                            marker='x',
                            s=15.0
                        )



            elif control == OptimalControl.RAMPING_DOWN:
                if not has_ramping_down_label_written:
                    has_ramping_down_label_written = True
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color='red',
                        marker='o',
                        label="Ramping Down",
                        s=15.0
                    )
                else:
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color='red',
                        marker='o',
                        s=15.0
                    )
            else:  # DO_NOTHING
                if not has_do_nothing_label_written:
                    has_do_nothing_label_written = True
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color='yellow',
                        marker='o',
                        label="Ramping Down",
                        s=5.0
                    )
                else:
                    axes[3].scatter(
                        power_prices.index[i],
                        value_along_path[i],
                        color='yellow',
                        marker='o',
                        s=5.0
                    )

        axes[3].set_title("Control Decisions along Simulation Path")
        axes[3].set_xlabel("Date")
        axes[3].set_ylabel("Control Decision")
        axes[3].legend()

        # Plot the states along the simulation path as markers
        for i, state in enumerate(optimal_states_along_path):
            if state == OperationalState.IDLE:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color='blue',
                    marker='o',
                    label="Idle" if i == 0 else "",
                )
            elif state == OperationalState.RAMPING_UP:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color='green',
                    marker='x',
                    label="Ramping Up" if i == 0 else "",
                )
            elif state == OperationalState.RAMPING_DOWN:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color='red',
                    marker='s',
                    label="Ramping Down" if i == 0 else "",
                )
            elif state == OperationalState.RUNNING:
                axes[4].scatter(
                    power_prices.index[i],
                    value_along_path[i],
                    color='yellow',
                    marker='o',
                    label="Running" if i == 0 else "",
                )

        axes[3].set_title("States along Simulation Path")
        axes[3].set_xlabel("Date")
        axes[3].set_ylabel("State")
        axes[3].legend()

        plt.tight_layout()
        plt.show()


# Example usage of the PowerPlant class:
if __name__ == "__main__":
    # Simulation parameters
    simulation_start = date(2024, 1, 1)
    simulation_end = date(2024, 12, 31)
    power_volatility = 0.2  # 20% annualized volatility
    coal_volatility = 0.25  # 15% annualized volatility
    correlation = 0.7  # Positive correlation

    # Power curve rises faster than coal but starts lower
    fwd_curve_power = pd.Series(
        data=[80, 83, 86, 89, 92, 95, 100, 105, 110, 115, 120, 125],  # Power rises over time
        index=pd.period_range(start="2024-01", end="2024-12", freq="M"),
    )

    # Coal curve falls over time, but starts higher
    fwd_curve_coal = pd.Series(
        data=[120, 128, 116, 104, 95, 85, 70, 95, 114, 120, 130, 110],  # Coal falls over time
        index=pd.period_range(start="2024-01", end="2024-12", freq="M"),
    )

    # Parameters for the mean-reverting OU process
    sigma_ou_power = 0.05
    sigma_ou_coal = 0.07
    beta_power = 10.0
    beta_coal = 15.0

    # Operational costs
    operation_costs = 2  # Example value
    alpha = 1.0
    beta = 1.0
    ramping_up_costs = 4
    ramping_down_costs = 3
    idle_costs = 0.02

    n_days_ramping_up = 1
    n_days_ramping_down = 1

    # Create the PowerPlant object
    power_plant = PowerPlant(
        operation_costs,
        alpha,
        ramping_up_costs,
        ramping_down_costs,
        n_days_ramping_up,
        n_days_ramping_down,
        idle_costs,
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

    # Run the simulation
    n_sims = 100
    power_plant.optimize(n_sims)

    simulation_path = 55  # Example path index to plot
    power_plant.plot_optimization_results(
        initial_state=OperationalState.IDLE, simulation_path=simulation_path
    )
