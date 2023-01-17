def mask_predictions(predictions, targets, mask):
    for k in range(len(predictions)):
        predictions[k] = (~mask) * predictions[k] + mask * targets[k]
    return predictions
