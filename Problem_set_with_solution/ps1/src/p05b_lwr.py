import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Fit a LWR model
    model = LocallyWeightedLinearRegression(0.5)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    mse = ((y_pred - y_val) ** 2).mean()
    print(mse)

class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        from numpy.linalg import inv, norm
        m, n = x.shape
        self_m, self_n = self.x.shape
        y = np.zeros(m)

        # For computing weights
        g = lambda x: np.exp(- (x ** 2) / (2 * self.tau ** 2))

        # For each x
        for i in range(m):
            w = np.zeros(self_m)
            for j in range(self_m):
                w[j] = g(norm(self.x[j,:] - x[i,:]))
            W = 0.5 * np.diag(w)

            theta = inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)
            y[i] = np.dot(x[i,:], theta)

        return y
