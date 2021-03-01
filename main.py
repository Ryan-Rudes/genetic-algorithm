from tqdm import tqdm
import numpy as np
import retro
import heapq
import time
import cv2
import os

if not os.path.exists("./models/"):
    os.mkdir("./models/")
    
start = time.time()
folder = f"./models/{start}/"
os.mkdir(folder)

env = retro.make("SuperMarioBros-Nes")

INPUT_SHAPE = (36, 36)
NUM_ACTIONS = env.action_space.n
POPULATION_SIZE = 16
SURVIVORS = 6

in_nodes = np.prod(INPUT_SHAPE)
out_nodes = NUM_ACTIONS

def preprocess(observation, render=False):
    observation = cv2.resize(observation, INPUT_SHAPE, interpolation = cv2.INTER_AREA)
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return observation.flatten() / 255.0

def evaluate(weights, decay=0.999, render=False):
    observation = env.reset()
    terminal = False
    score = 0
    discount = 1.0
    while not terminal:
        observation = preprocess(observation)
        q = np.dot(weights, observation)
        q = (q - q.min()) / (q.max() - q.min())
        p = q / q.sum()
        button = np.random.choice(np.arange(NUM_ACTIONS), p = p)
        action = np.zeros(NUM_ACTIONS)
        action[button] = 1
        observation, reward, terminal, info = env.step(action)
        score += reward * discount
        discount *= decay
        terminal = terminal or info['lives'] < 2
        if render:
            env.render()
    # fitness = info['xscrollLo'] + score
    fitness = score
    return fitness

def crossover(parentA, parentB):
    parentA = np.copy(parentA)
    parentB = np.copy(parentB)
    for rowA, rowB in zip(parentA, parentB):
        divider = np.random.randint(0, in_nodes)
        if np.random.random() < 0.5:
            temp = rowA[:divider]
            rowA[:divider] = rowB[:divider]
            rowB[:divider] = temp
        else:
            temp = rowA[divider:]
            rowA[divider:] = rowB[divider:]
            rowB[divider:] = temp
    return parentA, parentB

def mutate(weights, prob=0.99):
    weights = np.copy(weights)
    p = np.random.random(weights.shape)
    p[p > prob] = 1
    p[p < prob] = 0
    weights = weights * (1 - p)
    mutations = np.random.normal(size = weights.shape)
    weights = weights + mutations * p
    return weights

def generate_offspring(survivors):
    offspring = []
    for i in range(POPULATION_SIZE // 2):
        a, b = np.random.choice(np.arange(SURVIVORS), size = 2, replace = False)
        childA, childB = crossover(survivors[a], survivors[b])
        offspring.append(childA)
        offspring.append(childB)
    offspring = [mutate(child) for child in offspring]
    return offspring

def sample():
    return np.random.normal(size = (out_nodes, in_nodes))

population = [sample() for i in range(POPULATION_SIZE)]

for generation in range(100):
    fitnesses = []
    for individual in tqdm(population, "Generation %d" % generation):
        fitness = evaluate(individual)
        fitnesses.append(fitness)
        print ("Fitness:", fitness)
    survivors = list(map(population.__getitem__, heapq.nlargest(SURVIVORS, range(POPULATION_SIZE), fitnesses.__getitem__)))
    np.save(folder + f"{generation}.npy", survivors[0])
    population = generate_offspring(survivors)

env.close()
