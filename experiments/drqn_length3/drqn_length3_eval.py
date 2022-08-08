import numpy as np
import pickle
import os


results_file = open('drqn_length3/data/results.pkl', 'rb')
results = pickle.load(results_file)
results_file.close()

avg_train_score = 0
avg_test_score = 0
for train_patterns, tps_results in results.items():
    # Compute average train and test score for dqrn agent trained on given
    # train patterns
    avg_local_train_score = np.array([
        s for p, r in tps_results.items() if p in train_patterns for s in r
    ]).mean()
    avg_local_test_score = np.array([
        s for p, r in tps_results.items() if p not in train_patterns for s in r
    ]).mean()

    # Aggregate train and test score across train patterns
    avg_train_score += avg_local_train_score
    avg_test_score += avg_local_test_score

    # Print stats for given train patterns
    print('Training patterns: {train_patterns} | Train: {train:3.2f} | Test: {test:3.2f}'.format(
        train_patterns=train_patterns,
        train=avg_local_train_score,
        test=avg_local_test_score,
    ))

    # Separately print stats for each possible evaluation pattern
    for eval_pattern, ep_results in tps_results.items():
        print('-> {eval_pattern} : {mean:6.2f} {indicator}'.format(
            eval_pattern=eval_pattern,
            indicator=('*' if eval_pattern in train_patterns else '?'),
            mean=np.array(ep_results).mean()
        ))

print('-' * 100, '\n')
avg_train_score = avg_train_score / len(results)
avg_test_score = avg_test_score / len(results)

print('Overall results. | Train: {train:3.2f} | Test: {test:3.2f}'.format(
    train=avg_train_score,
    test=avg_test_score,
))