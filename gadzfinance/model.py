import numpy as np


class GBM:
    def fit(self, data: np.ndarray) -> None:
        # copies
        data = data.copy()

        # processing
        returns = data[1:] / data[:-1] - 1

        # fitting
        self.mean = np.mean(returns, axis=0)
        self.cov = np.cov(returns.T)
        self.last = data[-1]

    def sample(self, n_days: int, n_sim: int) -> np.ndarray:
        sim = np.random.multivariate_normal(
            self.mean, self.cov, size=[n_sim, n_days]
        )  # return simulation
        sim = self.last * (1 + sim).cumprod(axis=1)  # integrating the returns
        return sim
