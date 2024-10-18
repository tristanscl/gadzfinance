import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def optimize(self) -> np.ndarray:
        pass


class Tangency(Optimizer):
    def __init__(self, mean: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> None:
        self.mean = mean
        self.cov = cov
        self.rf = rf

    def optimize(self) -> np.ndarray:
        inv_cov = np.linalg.inv(self.cov)
        adj_ret = self.mean - self.rf
        one = np.ones_like(adj_ret)

        return inv_cov @ adj_ret / (one.T @ inv_cov @ adj_ret)


class TargetStd(Optimizer):
    def __init__(
        self, std: float, mean: np.ndarray, cov: np.ndarray, rf: Optional[float] = None
    ) -> None:
        self.std = std
        self.mean = mean
        self.cov = cov
        self.rf = rf

    def _optimize_risky(self) -> np.ndarray:
        # constants
        one = np.ones_like(self.mean)
        inv_cov = np.linalg.inv(self.cov)

        s11 = one.T @ inv_cov @ one
        s1m = one.T @ inv_cov @ self.mean
        smm = self.mean.T @ inv_cov @ self.mean

        a = s11 / self.std**2 - s11**2
        b = 2 * (s1m / self.std**2 - s11 * s1m)
        c = smm / self.std**2 - s1m**2

        # see all possibilities
        possible = []
        for S0 in [-1, 1]:
            for S1 in [-1, 1]:
                l1 = (-b + S1 * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
                l0 = (
                    S0
                    * np.sqrt(
                        (self.mean + l1 * one).T @ inv_cov @ (self.mean + l1 * one)
                    )
                    / (2 * self.std)
                )
                possible.append(-inv_cov @ (self.mean + l1 * one) / (2 * l0))

        # select the best case
        w_max = possible[0]
        max_ret = w_max @ self.mean
        for w in possible:
            ret = w @ self.mean
            if ret > max_ret:
                w_max = w
                max_ret = w_max @ self.mean
        return w_max

    def _optimize_rf(self) -> np.ndarray:
        adj_mean = self.mean - self.rf
        inv_cov = np.linalg.inv(self.cov)

        return self.std * inv_cov @ adj_mean / np.sqrt(adj_mean.T @ inv_cov @ adj_mean)

    def optimize(self) -> np.ndarray:
        if self.rf is None:
            return self._optimize_risky()
        else:
            return self._optimize_rf()


class TargetRet(Optimizer):
    def __init__(
        self, ret: float, mean: np.ndarray, cov: np.ndarray, rf: Optional[float] = None
    ) -> None:
        self.ret = ret
        self.mean = mean
        self.cov = cov
        self.rf = rf

    def _optimize_risky(self) -> np.ndarray:
        u = np.array([self.ret, 1])
        U = np.column_stack([self.mean, np.ones_like(self.mean)])
        inv_cov = np.linalg.inv(self.cov)
        M = U.T @ inv_cov @ U
        inv_M = np.linalg.inv(M)

        return inv_cov @ U @ inv_M @ u

    def _optimize_rf(self) -> np.ndarray:
        adj_mean = self.mean - self.rf
        adj_ret = self.ret - self.rf  # type: ignore
        inv_cov = np.linalg.inv(self.cov)
        return adj_ret * inv_cov @ adj_mean / (adj_mean.T @ inv_cov @ adj_mean)

    def optimize(self) -> np.ndarray:
        if self.rf is None:
            return self._optimize_risky()
        else:
            return self._optimize_rf()
