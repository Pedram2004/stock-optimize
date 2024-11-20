import numpy as np


def fitness_func(portfolio_choices_: list, return_list_: np.array, covariance_matrix_: np.array):

    portfolio_choices_transpose = np.transpose(portfolio_choices_)
    portfolio_risk = np.matmul(
        np.matmul(portfolio_choices_transpose, covariance_matrix_),
        portfolio_choices_)

    portfolio_reward = np.matmul(portfolio_choices_transpose, return_list_)

    return portfolio_reward / portfolio_risk