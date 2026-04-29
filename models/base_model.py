class BaselineModel:
    # Base class for all motion models
    def __init__(self, name):
        self.name = name

    # Takes context data and predicts future trajectory
    # Input: context_data (DataFrame or numpy array)
    # Output: predicted_trajectory (numpy array of [x, y] coordinates)
    def predict(self, context_data):
        raise NotImplementedError("predict() must be implemented by child classes")