
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetaCognition:
    def __init__(self, initial_state):
        self.state = initial_state
        self.history = []

    def reflect(self):
        reflection_score = np.mean(self.history) if self.history else 0
        logging.info(f"Reflection on past actions gives a score of: {reflection_score}")
        return reflection_score

    def adapt(self, feedback):
        self.history.append(feedback)
        adjustment = np.tanh(feedback)
        self.state += adjustment
        logging.info(f"Adapted state based on feedback. New state: {self.state}")

    def predict_outcome(self, action):
        predicted_outcome = np.dot(self.state, action)
        logging.info(f"Predicted outcome of action {action} is {predicted_outcome}")
        return predicted_outcome

    def evaluate_and_adapt(self, action, outcome):
        predicted = self.predict_outcome(action)
        error = outcome - predicted
        self.adapt(error)
        logging.info(f"Evaluation and adaptation complete. Error: {error}")

def main():
    mc = MetaCognition(initial_state=np.array([0.5, 0.2, 0.3]))
    actions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    outcomes = [1, 0.5, 0.3]
    
    for action, outcome in zip(actions, outcomes):
        mc.evaluate_and_adapt(action, outcome)
    
    reflection_score = mc.reflect()
    logging.info(f"Meta-cognition process complete with reflection score: {reflection_score}")

if __name__ == "__main__":
    main()
