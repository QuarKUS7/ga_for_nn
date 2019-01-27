import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')[:1000] / 255
x_test = x_test.reshape(10000, 784).astype('float32')[:1000] / 255
y_train = to_categorical(y_train, 10)[:1000]
y_test = to_categorical(y_test, 10)[:1000]

classes = 10
batch_size = 64
population = 20
generations = 100

threshold = 0.99

def serve_model(epochs, units, acts, classes, loss, opt, xtrain, ytrain, summary=False):
    model.add(Dense(units1, input_shape=[784,]))
    model = Sequential()
    for unit, act in zip(units, acts[:-1]):
        model.add(Activation(act))
        model.add(Dense(unit))
    model.add(Dense(classes))
    model.add(Activation(acts[-1]))
    model.compile(loss=loss, optimizer=opt, metrics=['acc'])
    if summary:
        model.summary()

    model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=0)

    return model

class Network():
    def __init__(self):
        self._epochs = np.random.randint(1, 15)

        self._units = np.random.randint(1, 14, size =np.random.randint(1, 8)).tolist()*2

        self._acts = np.random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'], size=len(self._units)+1).tolist()

        self._loss = np.random.choice([
            'categorical_crossentropy',
            'binary_crossentropy',
            'mean_squared_error',
            'mean_absolute_error',
            'sparse_categorical_crossentropy'
        ])

        self._opt = np.random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])

        self._accuracy = 0

    def init_hyperparams(self):
        hyperparams = {
            'epochs': self._epochs,
            'units': self._units,
            'acts': self._acts,
            'loss': self._loss,
            'optimizer': self._opt
        }
        return hyperparams

def init_networks(population):
    return [Network() for _ in range(population)]

def fitness(networks):
    for network in networks:
        hyperparams = network.init_hyperparams()
        epochs = hyperparams['epochs']
        units = hyperparams['units']
        acts = hyperparams['acts']
        loss = hyperparams['loss']
        opt = hyperparams['optimizer']

        try:
            model = serve_model(epochs, units, acts, classes, loss, opt, x_train, y_train)
            accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
            network._accuracy = accuracy
            print ('Accuracy: {}'.format(network._accuracy))
        except:
            network._accuracy = 0
            print ('Failed build. Moving to next model.')

    return networks

def selection(networks):
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    networks = networks[:int(0.2 * len(networks))]

    return networks

def crossover(networks):
    offspring = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()

        # Crossing over parent hyper-params
        child1._epochs = int(parent1._epochs/4) + int(parent2._epochs/2)
        child2._epochs = int(parent1._epochs/2) + int(parent2._epochs/4)

        child1._units = random.choice(parent1._units, size=len(parent1._units)/2, replace=False) + random.choice(parent2._units, size=len(parent2._units)/2, replace=False)
        child2._units = random.choice(parent1._units, size=len(parent1._units/2, replace=False) + random.choice(parent2._units, size=parent2._units/2, replace=False)

        child1._acts = random.choice(parent1._acts[:-1], size=parent1._acts[:-1]/2, replace=False) + random.choice(parent2._acts[:-1], size=parent2._acts[:-1]/2, replace=False).tolist()
        child1._acts.extend(random.choice([parent1._acts[-1],parent2._acts[-1]]))

        child2._acts = random.choice(parent1._acts[:-1], size=parent1._acts[:-1]/2, replace=False) + random.choice(parent2._acts[:-1], size=parent2._acts[:-1]/2, replace=False).tolist()
        child2._acts.extend(random.choice([parent1._acts[-1],parent2._acts[-1]]))

        offspring.append(child1)
        offspring.append(child2)

    networks.extend(offspring)

    return networks

def mutate(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.1:
            network._epochs += np.random.randint(0,100)
            network._units = np.random.randint(1, 14, size =np.random.randint(1, 8)).tolist()*2

    return networks

def main():
    networks = init_networks(population)

    for gen in range(generations):
        print ('Generation {}'.format(gen+1))

        networks = fitness(networks)
        networks = selection(networks)
        networks = crossover(networks)
        networks = mutate(networks)

        for network in networks:
            if network._accuracy > threshold:
                print ('Threshold met')
                print (network.init_hyperparams())
                print ('Best accuracy: {}'.format(network._accuracy))
                #exit(0)
                return

if __name__ == '__main__':
    main()
