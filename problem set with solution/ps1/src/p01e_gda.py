import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    y_train = y_train.astype("int32")

    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_val = y_val.astype("int32")
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, '{}.png'.format(pred_path))

    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, y_pred)

class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape

        # Find phi, mu_0, mu_1, and sigma
        phi = (y == 1).sum() / m
        mu_0 = x[y == 0].sum(axis=0) / (y == 0).sum()
        mu_1 = x[y == 1].sum(axis=0) / (y == 1).sum()

        sigma = np.zeros((n,n))
        for i in range(m):
            if y[i] == 1:
                mu = mu_1
            else:
                mu = mu_0
            sigma += np.tensordot(x[i] - mu, x[i] - mu, 0)
        sigma = sigma/m

        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta0 = np.log(phi/(1 - phi)) + (0.5 * (np.dot( mu_0, np.matmul(sigma_inv, mu_0) ) - np.dot( mu_1, np.matmul(sigma_inv, mu_1) )))
        theta0 = np.array([theta0])
        theta = np.hstack([theta0, theta])
        self.theta = theta

        return theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # we do not assume that intercept is added.
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        x = util.add_intercept(x)
        probs = sigmoid(x.dot(self.theta))
        preds = (probs >= 0.5).astype(np.int)
        return preds
