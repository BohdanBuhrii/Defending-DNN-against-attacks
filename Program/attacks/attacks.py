import numpy as np
import matplotlib.pyplot as plt

class Attack:

    def __init__(self, model):
        self.model = model
        self.parameters = model.parameters
        self.mask = None

    def __tanh(self, Z):
        return np.tanh(Z)

    def __tanh_derivative(self, Z):
        return 1 / np.power(np.cosh(Z), 2)

    def compute_cost(self, A, Y):
        J = -np.mean(Y.T * np.log(A.T + 1e-8))
        return J

    def e_vector(self, num, length=784):
      return np.array([[1] if i == num else [0] for i in range(length)])

    def __backward_linear_activation(self, dX, cache, activation):

        linear_cache, activation_cache = cache

        # activation backward
        Z = activation_cache.T

        if activation == 'sigmoid':
            dZ = dX * self.__sigmoid_derivative(Z)

        if activation == 'relu':
            dZ = dX * self.__relu_derivative(Z)

        if activation == 'tanh':
            dZ = dX * self.__tanh_derivative(Z)

        # linear backward
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        dX_prev = 1 / m * np.dot(dZ, W)

        return dX_prev

    def __multilayer_backward(self, Y_hat, Y, caches):
        Y = Y.reshape(Y_hat.shape)

        L = len(self.parameters)//2
        m = Y_hat.shape[1]

        dZ = Y_hat - Y

        dX = 1 / m * np.dot(dZ.T, self.parameters['W'+str(L)])

        for l in reversed(range(L-1)):
            dX = self.__backward_linear_activation(
                dX, caches[l], activation='tanh')

        return dX

    def update_instance_grads(self, instance, grads, adapting_rate, targeted):
        return instance + (-1 if targeted else 1)*adapting_rate*grads
        #return instance + (-1)*adapting_rate*grads

    def update_instance_FGSM(self, instance, grads, adapting_rate, targeted):
        return instance + (-1 if targeted else 1)*adapting_rate*np.sign(grads)

    def update_instance_ADAM(self, instance, grads, adapting_rate, targeted,
                             v, s, t, beta1=0.1, beta2=0.999,  epsilon=1e-8):

        v = beta1*v + (1 - beta1)*grads
        v_corrected = v/(1 - beta1**t)

        s = beta2*s + (1-beta2)*np.square(grads)
        s_corrected = s/(1 - beta2**t)

        instance = instance + (-1 if targeted else 1)*adapting_rate * \
            v_corrected/np.sqrt(s_corrected + epsilon)

        return instance, v, s

    def attack(self, instance, target, attack_type, num_iters,
               pretrub_importance=None, adapting_rate=0.01, targeted=False, print_cost=True):

        instance = instance.T
        initial = instance.copy()

        target = target.T
        m = instance.shape[1]
        costs = []

        v = np.zeros(instance.shape)
        s = np.zeros(instance.shape)

        for i in range(num_iters):
            Y_hat, caches = self.model.multilayer_forward(instance)

            self.iter = i

            # stop if adversarial
            if (instance.shape[1] == 1
                and (np.argmax(Y_hat) == np.argmax(target) and targeted)
                    or (np.argmax(Y_hat) != np.argmax(target) and not targeted)):

                break

            cost = self.model.compute_cost(Y_hat, target)

            grads = self.__multilayer_backward(Y_hat, target, caches).T

            if(pretrub_importance != None):
                # TODO pretrub_importance*
                grads = grads + 2*(instance - initial)

            if(attack_type == 'grads'):
                buff = instance - adapting_rate * \
                    (2*(instance - initial) - 0.01*grads)

            if(attack_type == 'FGSM'):
                buff = self.update_instance_FGSM(
                    instance, grads, adapting_rate, targeted)

            if(attack_type == 'ADAM'):
                buff, v, s = self.update_instance_ADAM(instance, grads,
                                                       adapting_rate, targeted, v, s, i+1)

            instance = np.clip(buff, 0, 1)

            costs.append(cost)
            if print_cost and i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        if print_cost:
            plt.title(str(num_iters) + " iterations")
            plt.plot(costs)
            plt.ylabel("Cost")
            plt.xlabel("Iteration")
            plt.show()

        return instance

    def zoo_attack_stohastic(self, instance, target, attack_type, num_iters,
                             pretrub_importance=None, adapting_rate=0.01, targeted=False, print_cost=True, h=1e-8):

        instance = instance.T.copy()
        initial = instance.copy()

        target = target.T
        m = instance.shape[1]
        costs = []

        v = np.zeros(instance.shape)
        s = np.zeros(instance.shape)

        counter = 1

        for i in range(num_iters):
            Y_hat, caches = self.model.multilayer_forward(instance)

            attack.iter = i
            print('attack iter', i)

            range_ = np.arange(instance.shape[0])
            np.random.shuffle(range_)

            for j in range_:
                # stop if adversarial
                if (instance.shape[1] == 1
                    and (np.argmax(Y_hat) == np.argmax(target) and targeted)
                        or (np.argmax(Y_hat) != np.argmax(target) and not targeted)):

                    break

                #cost = self.model.compute_cost(Y_hat, target)

                Y_left, caches = self.model.multilayer_forward(
                    instance + self.e_vector(j)*h)

                Y_right, caches = self.model.multilayer_forward(
                    instance - self.e_vector(j)*h)

                grad = (self.compute_cost(Y_left, target) -
                        self.compute_cost(Y_right, target))/(2*h)

                if(attack_type == 'grads'):
                    buff = self.update_instance_grads(
                        instance[j][0], grad, adapting_rate, targeted)

                if(attack_type == 'FGSM'):
                    buff = self.update_instance_FGSM(
                        instance[j][0], grad, adapting_rate, targeted)

                if(attack_type == 'ADAM'):
                    buff, v, s = self.update_instance_ADAM(instance[j][0], grad,
                                                           adapting_rate, targeted, v, s, counter)
                counter += 1
                #print(buff)

                instance[j][0] = np.clip(buff[0][0], 0, 1)

            if (instance.shape[1] == 1
                    and (np.argmax(Y_hat) == np.argmax(target) and targeted)
                    or (np.argmax(Y_hat) != np.argmax(target) and not targeted)):

                break

            #costs.append(cost)
            if print_cost and i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        if print_cost:
            plt.title(str(num_iters) + " iterations")
            plt.plot(costs)
            plt.ylabel("Cost")
            plt.xlabel("Iteration")
            plt.show()

        return instance

    def zoo_attack(self, instance, target, attack_type, num_iters,
                   pretrub_importance=None, adapting_rate=0.01, targeted=False, print_cost=True, h=1e-8):

        instance = instance.T
        initial = instance.copy()

        target = target.T
        m = instance.shape[1]
        costs = []

        v = np.zeros(instance.shape)
        s = np.zeros(instance.shape)

        for i in range(num_iters):
            Y_hat, caches = self.model.multilayer_forward(instance)

            attack.iter = i
            print('attack iter', i)

            if (instance.shape[1] == 1
                    and (np.argmax(Y_hat) == np.argmax(target) and targeted)
                    or (np.argmax(Y_hat) != np.argmax(target) and not targeted)):

                break

            grads = np.zeros(instance.shape)

            for j in range(instance.shape[0]):

                Y_left, caches = self.model.multilayer_forward(
                    instance + self.e_vector(j)*h)

                Y_right, caches = self.model.multilayer_forward(
                    instance - self.e_vector(j)*h)

                grad = (self.compute_cost(Y_left, target) -
                        self.compute_cost(Y_right, target))/(2*h)

                grads[j][0] = grad

            if(pretrub_importance != None):
                grads = grads + pretrub_importance*(instance - initial)  # TODO

            if(attack_type == 'grads'):
                buff = self.update_instance_grads(
                    instance, grads, adapting_rate, targeted)

            if(attack_type == 'FGSM'):
                buff = self.update_instance_FGSM(
                    instance, grads, adapting_rate, targeted)

            if(attack_type == 'ADAM'):
                buff, v, s = self.update_instance_ADAM(instance, grads,
                                                       adapting_rate, targeted, v, s, i+1)

            instance = np.clip(buff, 0, 1)

            #costs.append(cost)
            if print_cost and i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        if print_cost:
            plt.title(str(num_iters) + " iterations")
            plt.plot(costs)
            plt.ylabel("Cost")
            plt.xlabel("Iteration")
            plt.show()

        return instance

    def attack_on_dp(self, instance, target, attack_type, num_iters,
            pretrub_importance=None, adapting_rate=0.01, targeted=False, print_cost=True):

        instance = instance.T
        initial = instance.copy()

        target = target.T
        m = instance.shape[1]
        costs = []
        n_attempts = self.model.n_expected_scores

        v = np.zeros(instance.shape)
        s = np.zeros(instance.shape)

        for i in range(num_iters):
            self.iter = i
            
            grads = np.zeros(instance.shape)
            Y_hat = np.zeros(target.shape)
            for _ in range(n_attempts):
                Y_hat_s, caches = self.model.multilayer_forward(instance)

                cost = self.model.compute_cost(Y_hat_s, target)

                grads += self.__multilayer_backward(Y_hat_s, target, caches).T
                Y_hat += Y_hat_s

            grads /= n_attempts
            Y_hat /= n_attempts

            # stop if adversarial
            if (instance.shape[1] == 1 and not self.stop_if_adversarial
                and (np.argmax(Y_hat) == np.argmax(target) and targeted
                     or np.argmax(Y_hat) != np.argmax(target) and not targeted)):

                break

            if(pretrub_importance != None):
                # TODO pretrub_importance*
                grads = grads + 2*(instance - initial)

            if(attack_type == 'grads'):
                buff = instance - adapting_rate * \
                    (2*(instance - initial) - 0.01*grads)

            if(attack_type == 'FGSM'):
                buff = self.update_instance_FGSM(
                    instance, grads, adapting_rate, targeted)

            if(attack_type == 'ADAM'):
                buff, v, s = self.update_instance_ADAM(instance, grads,
                                                    adapting_rate, targeted, v, s, i+1)

            instance = np.clip(buff, 0, 1)

            costs.append(cost)
            if print_cost and i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        if print_cost:
            plt.title(str(num_iters) + " iterations")
            plt.plot(costs)
            plt.ylabel("Cost")
            plt.xlabel("Iteration")
            plt.show()

        return instance
