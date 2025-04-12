
def select_best_model(models):
    return max(models, key=lambda m: m['performance'])
