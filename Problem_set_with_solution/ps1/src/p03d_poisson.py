import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Fit a Poisson Regression model
    model = PoissonRegression(max_iter=1000000, step_size=lr, eps=1e-5)
    model.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path

    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    np.savetxt(pred_path, y_pred)

class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        self.theta = np.zeros(n)
        g = lambda x: np.exp(x) # Response function

        for j in range(self.max_iter):
            theta = self.theta.copy()
            gradient = np.zeros(n)
            for i in range(m):
                gradient += (1/m) * ( y[i] - g(np.dot(theta, x[i,:])) ) * x[i,:]

            self.theta  += self.step_size * gradient
            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break
    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        m, n = x.shape

        g = lambda x: np.exp(x)
        y = np.zeros(m);
        for i in range(m):
            y[i] = g( np.dot(self.theta, x[i]) )
        return y
