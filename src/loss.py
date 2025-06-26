import torch
import torch.nn as nn

class DifferentiableCIndexLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, risk_scores, times, events):
        """
        Differentiable approximation of C-index loss.

        risk_scores: (B,) predicted risk (higher = more risk)
        times: (B,) observed survival/censoring times
        events: (B,) event indicators (1 if event occurred, 0 if censored)
        """
        n = len(times)
        loss = 0.0
        count = 0

        for i in range(n):
            for j in range(n):
                if times[i] < times[j] and events[i] == 1:
                    diff = risk_scores[j] - risk_scores[i]
                    loss += torch.sigmoid(diff / self.sigma)
                    count += 1

        return loss / (count + 1e-6)
