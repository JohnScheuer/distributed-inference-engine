class AnalyticalCostModel:
    """
    Prediz o tempo de comunicação baseado em parâmetros do hardware.
    Na sua máquina (localhost/gloo), a bandwidth é limitada pela RAM/CPU.
    """
    def __init__(self, alpha_ms: float, beta_gb_s: float):
        self.alpha = alpha_ms
        self.beta = beta_gb_s

    def predict_ms(self, num_bytes: int) -> float:
        size_gb = num_bytes / 1e9
        return self.alpha + (size_gb / self.beta) * 1000
