import math

import numpy as np

"""Online predictor"""
class OnlinePredictor:
    def __init__(self, basis_functions: list, state_dim: int, act_dim: int, w=None, matrix_size=None, library=None, regularization=1e-3):
        self.__library = library
        if matrix_size is None:
            # Calculate matrix size
            matrix_size = (state_dim + act_dim) * len(basis_functions)

        # Set basis functions
        self.__basis_functions = basis_functions
        # Initiate design matrix to all zeros
        self.__design_matrix = np.zeros((matrix_size, matrix_size))
        # Initiate vector to multiply to design matrix inverse to all zeros
        self.__time_feature_predict_sum = np.zeros((state_dim, matrix_size))
        # Initialize online matrix to None
        self.__w = w
        self.__state_dim = state_dim
        self.__act_dim = act_dim
        self.__regularization = regularization

    def __get_features(self, state: np.ndarray, action: np.ndarray):

        if self.__library is None:
            state_features = np.array([func(state) for func in self.__basis_functions]).flatten()
            action_features = np.array([func(action) for func in self.__basis_functions]).flatten()
            return np.concat((state_features, action_features)).flatten()

        else:

            all_features = self.__library.fit_transform(np.hstack((state, action)))
            return all_features


    def __get_batch_features(self, state: np.ndarray, action: np.ndarray):
        if self.__library is None:
            state_features = np.hstack([func(state) for func in self.__basis_functions])
            action_features = np.hstack([func(action) for func in self.__basis_functions])

            return np.concatenate([state_features, action_features], axis=1)

        else:
            all_features = self.__library.fit_transform(np.hstack((state, action)))
            return all_features

    """Update the predictor with a new (state, action, next_state) tuple"""
    def partial_fit(self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, filter=None):
        # Map state and action to basis functions
        current_features = self.__get_features(current_state, action)
        # Update design matrix with outer product of basis functions
        self.__design_matrix = self.__design_matrix + np.outer(current_features, current_features)
        # Update vector to multiply with design matrix inverse
        self.__time_feature_predict_sum = self.__time_feature_predict_sum + next_state[:, np.newaxis] * current_features
        try:
            # Try to invert design matrix
            inverted_design_matrix = np.linalg.inv(self.__design_matrix + self.__regularization * np.eye(self.__design_matrix.shape[0]))

            old = self.__w
            # Calculate online matrix
            self.__w = np.array([np.dot(inverted_design_matrix, vector_row) for vector_row in self.__time_feature_predict_sum])

            if old is not None:
                return self.__w - old
            else:
                return None
        except np.linalg.LinAlgError:
            pass

    def get_inverse(self):
        try:
            return np.linalg.inv(self.__design_matrix + self.__regularization * np.eye(self.__design_matrix.shape[0]))
        except np.linalg.LinAlgError:
            return None

    """Predict the next state"""
    def predict(self, state: np.ndarray, action: np.ndarray):
        # Apply the online matrix to the map of state and action to the basis functions
        return self.__w @ self.__get_features(state, action)

    def predict_direct(self, state: np.ndarray, action: np.ndarray):
        return self.__w @ np.concat((state,action)).flatten()

    def predict_batch(self, state, action):
        state = np.array(state)
        action = np.array(action)

        return np.matmul(self.__get_batch_features(state, action), self.__w.T)

    def predict_batch_direct(self, concatenated):
        return concatenated @ self.__w.T

    """Print the online matrix"""
    def print_matrix(self):
        if self.__w is None:
            return
        for r in self.__w:
            for c in r:
                print("{: 010.4f}".format(c), end=" ")
            print("")

    """Calculate the distance between the Online matrix and a input matrix"""
    def distance(self, w_star) -> float:
        if self.__w is None:
            return np.inf
        # Extend the input matrix with zeros if needed
        # extended_w_star = np.zeros(self.__w.shape)
        # extended_w_star[:, :w_star.shape[1]] = w_star
        # Calculate distance in norm of energy
        return max(np.linalg.eigvals((w_star - self.__w) @ (self.__design_matrix + self.__regularization * np.eye(self.__design_matrix.shape[0])) @ (w_star - self.__w).T))

    @property
    def w(self):
        return self.__w

    @property
    def w_state(self):
        if self.__w is not None:
            return self.w[:, :self.__state_dim * len(self.__basis_functions)]
        else:
            return None

    @property
    def w_action(self):
        if self.__w is not None:
            return self.w[:, self.__state_dim * len(self.__basis_functions):]
        else:
            return None

    @property
    def design_matrix(self):
        return self.__design_matrix + self.__regularization * np.eye(self.__design_matrix.shape[0])