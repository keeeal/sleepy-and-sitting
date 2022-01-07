from dataclasses import dataclass


@dataclass
class ConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def __len__(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / len(self)

    def precision(self) -> float:
        return self.true_positive / (self.true_positive + self.false_positive)

    def recall(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    def f_score(self, beta: float = 1) -> float:
        precision, recall = self.precision(), self.recall()
        return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
