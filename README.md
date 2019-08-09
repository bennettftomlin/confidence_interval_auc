# confidence_interval_auc

Place script in working directory and then you can use function by:

from conf_auc import conf_auc

Pass the function: test_predictions, ground truth, and optionally: number of bootstraps, seed, and confidence interval

test_predictions = predicted results from model 

ground_truth = true values to compare to predictions

bootstrap = number of bootstrapping operations to perform default=1000

seed = seed to be used for random number generation default=None  

confint=confidence interval to calculate default=0.95

returns a tuple containing (lower bound for confidence interval, auc, upper bound for confidence interval)
