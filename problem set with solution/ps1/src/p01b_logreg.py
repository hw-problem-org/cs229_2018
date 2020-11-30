import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, pred_path + ".png")

    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred, fmt="%.4f")


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """

        g = lambda x: 1 / (1 + np.exp(-x))
        m, n = x.shape

        # initialize theta
        self.theta = np.zeros(n)

        # optimize theta
        while True:
            theta = self.theta

            # compute gradient
            gradient = np.zeros(n)
            for i in range(m):
                h_theta_xi = g(np.dot(theta, x[i]))
                gradient += (y[i] - h_theta_xi) * x[i]
            gradient = gradient * (-1/m)

            # compute H
            hessian = np.zeros((n,n))
            for i in range(m):
                h_theta_xi = g(np.dot(theta, x[i]))
                hessian += h_theta_xi * (1 - h_theta_xi) * np.tensordot(x[i], x[i], 0)
            hessian = hessian * (1/m)
            hessian_inv = np.linalg.inv(hessian)

            # update
            self.theta = theta - (self.step_size * np.dot(hessian_inv, gradient))

            # if norm is small, terminate
            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # compute probability
        g = lambda x: 1 / (1 + np.exp(-x))
        m, n = x.shape
        preds = np.zeros(m)
        for i in range(m):
            preds[i] = g(np.dot(x[i], self.theta))

        return preds
