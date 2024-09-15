import torch
from torchmetrics import Metric
from sklearn.metrics import average_precision_score

class MeanAveragePrecision(Metric):
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.add_state("true_labels", default=[], dist_reduce_fx="cat")
        self.add_state("pred_probs", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        # if is logits, convert to probs using sigmoid
        if torch.max(preds) > 1.0:
            preds = torch.sigmoid(preds)

        if preds.shape[-1] != self.num_labels:
            raise ValueError("Number of labels in predictions does not match the expected number of labels.")

        self.true_labels.append(target.cpu())
        self.pred_probs.append(preds.cpu())

    def compute(self):
        true_labels = torch.cat(self.true_labels, dim=0).detach().numpy()
        pred_probs = torch.cat(self.pred_probs, dim=0).detach().numpy()

        average_precision = average_precision_score(
            true_labels,
            pred_probs,
            average=None
        )

        mean_ap = average_precision.mean()
        return torch.tensor(mean_ap, device=self.device)