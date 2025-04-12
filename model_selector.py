def select_best_model(results):
    return max(results, key=lambda x: x["score"])