from attacks.attacks import Attack
import pandas as pd
import time
import numpy as np


def L2_norm(x, axis=0):
  return np.sqrt(np.square(x).sum(axis=axis))


def robustness(X, X_adv, norm=L2_norm):
  return np.sum(norm(X - X_adv)) / X.shape[0]

def get_insights(classifier, epsilons, X_test_c, Y_test_c, max_iterations=1000, ignore_not_adversarial=False, show_progress=True):
  attack = Attack(classifier)

  df = pd.DataFrame(
      columns=['target', 'successful attempts', 'epsilon'])  # 'initial',
  non_targeted = pd.DataFrame(
      columns=['original', 'prediction', 'iterations', 'epsilon', 'L2 norm'])

  Y_hat_c = classifier.predict(X_test_c)

  tick = time.time()
  # np.arange(50)/255:#, 3/255, 5/255, 10/255, 15/255, 30/255, 50/255, 80/255, 120/255]:#[1/255]:#0.007, 0.01, 0.05, 0.1, 0.2]:
  for epsilon in epsilons:
    print('epsilon =', epsilon)
    total_attempts = 0
    total_iter = 0
    example = []
    initial = []
    true_label = []
    adversarial = []
    y_hat_adv = []

    for (x, y, y_hat) in zip(X_test_c, Y_test_c, Y_hat_c):
      x = np.array([x])

      if(not ignore_not_adversarial or y == y_hat):
        x_adv = attack.attack(x,
                              np.array([[1 if i == y else 0 for i in range(10)]]),\
                              #'ADAM',\
                              'FGSM',\
                              #'grads',\
                              max_iterations,\
                              #pretrub_importance=0.01,\
                              adapting_rate=epsilon, print_cost=False, targeted=False).T[0]

        total_iter += attack.iter

        example.append(x_adv)
        true_label.append([y])

        total_attempts += 1

        prediction = classifier.predict(np.array([x_adv]))[0]
        if(prediction != y):
          initial.append(x[0])
          adversarial.append(x_adv)

        non_targeted = non_targeted.append({'original': y[0], 'prediction': prediction,
                                            'iterations': attack.iter, 'L2 norm': L2_norm(x[0]-x_adv),
                                            'epsilon': epsilon}, ignore_index=True)

      y_hat_adv.append(prediction)

      if(show_progress and total_attempts % 100 == 0):
        print(total_attempts, 'instances,', len(adversarial), 'adversaries')

    example = np.array(example)
    initial = np.array(initial)
    true_label = np.array(true_label)
    adversarial = np.array(adversarial)
    y_hat_adv = np.array(y_hat_adv)

    print('------------------------------')
    #print('------------------------------',example.shape)
    #Y_hat = classifier.predict(example).reshape(true_label.shape)

    #print((Y_hat == target).shape)
    #print((Y_hat != Y_test) * (Y_hat == target) * ((cls.predict(X_test)==Y_test).reshape(Y_hat.shape)))

    successful_attempts = len(adversarial)  # np.sum(
    #  (Y_hat != true_label)
    #  * (Y_hat == target))
    #* (true_label == cls.predict(X_test_c).reshape(true_label.shape)))

    df = df.append({'successful attempts': successful_attempts,
                    'epsilon': epsilon}, ignore_index=True)

  print('total time:', time.time() - tick)
  print('total iter:', total_iter)
  return df, non_targeted, initial, adversarial
