class SyntheticCandidate:

    def __init__(self, data, model_name, epsilon, meta_features=None):
        self.data = data
        self.model_name = model_name
        self.epsilon = epsilon

        # 🔥 Evaluation metrics
        self.privacy_score = None
        self.utility_score = None
        self.diversity_score = None
        self.realism_score = None
        self.reward = None

        # 🔥 Context
        self.meta_features = meta_features

    # -------------------------
    # SETTERS
    # -------------------------

    def set_privacy(self, score):
        self.privacy_score = score

    def set_utility(self, score):
        self.utility_score = score

    def set_diversity(self, score):
        self.diversity_score = score

    def set_realism(self, score):
        self.realism_score = score

    def set_reward(self, reward):
        self.reward = reward

    # -------------------------
    # VALIDATION
    # -------------------------

    def is_valid(self):
        return self.reward is not None

    # -------------------------
    # SUMMARY (for logs/UI)
    # -------------------------

    def summary(self):
        return {
            "model": self.model_name,
            "epsilon": self.epsilon,
            "privacy": self.privacy_score,
            "utility": self.utility_score,
            "diversity": self.diversity_score,
            "realism": self.realism_score,
            "reward": self.reward
        }