from transformers import EvalPrediction

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy": (predictions == labels).sum().item() / len(predictions)}

# ... (evaluate the model using the compute_metrics function)
