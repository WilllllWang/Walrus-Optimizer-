import numpy as np
from scipy.stats import qmc
from scipy.special import gamma
from matplotlib import pyplot as plt


# Get best and second best individuals from the population
def getTwoBest(costs):
    bestIdx = np.argmin(costs)
    secondCost = float('inf') 
    secondIdx = -1
    for i in range(len(costs)):
        if i != bestIdx and costs[i] < secondCost:
            secondCost = costs[i]
            secondIdx = i
    return bestIdx, secondIdx


# Levy Flight step generator
def LF(D):
    alpha = 1.5
    sigmaX = ((gamma(1 + alpha) * np.sin((np.pi * alpha) / 2)) / (gamma((1 + alpha) / 2) * alpha * (2 ** ((alpha - 1) / 2)))) ** (1 / alpha)
    sigmaY = 1
    x = np.random.normal(0, sigmaX, size=D)
    y = np.random.normal(0, sigmaY, size=D)
    LF = 0.05 * (x / (np.abs(y) ** (1 / alpha)))
    return LF


# Main optimizer function
def WO(lu, iterMax, FOBJ, target):
    D = lu.shape[1]     # Dimension
    LB = lu[0]          # LowerBound
    UB = lu[1]          # Upper bound
    nPop = 100          # Population size should be divisible by 10

    # Initialization
    walruses = LB + np.random.rand(nPop, D) * (UB - LB)
    HS = qmc.Halton(d=D)
    costs = np.array([FOBJ(w) for w in walruses])

    bestIdx, secondIdx = getTwoBest(costs)
    bestW = walruses[bestIdx].copy()
    secondW = walruses[secondIdx].copy()
    globalBest = costs[bestIdx]
    globalBestPerIter = np.empty(iterMax)
    optimumIter = -1

    trueGlobalBest = globalBest
    trueGlobalBestParams = bestW.copy()

    for iter in range(iterMax):
        alpha = 1 - iter / iterMax
        A = 2 * alpha
        R = 2 * np.random.rand() - 1
        dangerSig = A * R
        safeSig = np.random.rand()

        if abs(dangerSig) >= 1:
            for i in range(nPop):
                while True:
                    vig1, vig2 = np.random.randint(0, nPop, size=2)
                    if vig1 != i and vig2 != i and vig1 != vig2:
                        break
                beta = 1 - 1 / (1 + np.exp((-(iter - (iterMax / 2)) / iterMax) * 10))
                migrationStep = (walruses[vig1] - walruses[vig2]) * beta * (np.random.rand()**2)
                walruses[i] = walruses[i] + migrationStep
                walruses[i] = np.clip(walruses[i], LB, UB)

        else:
            if safeSig >= 0.5:
                walruses[: int(nPop * 0.45)] = qmc.scale(HS.random(int(nPop * 0.45)), LB, UB)
                for i in range(int(nPop * 0.45), int(nPop * 0.9)):
                    walruses[i] = walruses[i] + alpha * (walruses[i - int(nPop * 0.45)] - walruses[i]) + (1 - alpha) * (bestW - walruses[i])
                    walruses[i] = np.clip(walruses[i], LB, UB)
                for i in range(int(nPop * 0.9), nPop):
                    O = bestW + walruses[i] * LF(D)
                    P = np.random.rand()
                    walruses[i] = (O - walruses[i]) * P
                    walruses[i] = np.clip(walruses[i], LB, UB)
            else:
                if abs(dangerSig) >= 0.5:
                    for i in range(nPop):
                        beta = 1 - 1 / (1 + np.exp((-(iter - (iterMax / 2)) / iterMax) * 10))
                        a1 = beta * np.random.rand() - beta
                        a2 = beta * np.random.rand() - beta   
                        b1 = np.tan(np.random.uniform(0 + 1e-12, np.pi - 1e-12))
                        b2 = np.tan(np.random.uniform(0 + 1e-12, np.pi - 1e-12))
                        x1 = bestW - a1 * b1 * abs(bestW - walruses[i])
                        x2 = secondW - a2 * b2 * abs(secondW - walruses[i])
                        walruses[i] = (x1 + x2) / 2
                        walruses[i] = np.clip(walruses[i], LB, UB)
                else:
                    for i in range(nPop):
                        walruses[i] = walruses[i] * R - abs(bestW - walruses[i]) * (np.random.rand()**2) 
                        walruses[i] = np.clip(walruses[i], LB, UB)

        costs = np.array([FOBJ(w) for w in walruses])
        bestIdx, secondIdx = getTwoBest(costs)
        bestW = walruses[bestIdx].copy()
        secondW = walruses[secondIdx].copy()
        globalBest = costs[bestIdx]

        if globalBest < trueGlobalBest:
            trueGlobalBest = globalBest
            trueGlobalBestParams = bestW.copy()

        globalBestPerIter[iter] = trueGlobalBest

        if globalBest <= target:
            optimumIter = iter
            globalBestPerIter[iter+1:] = trueGlobalBest
            break

    return trueGlobalBest, trueGlobalBestParams, globalBestPerIter, optimumIter


def sphere(X):
    return np.sum(X**2)


if __name__ == "__main__":
    D = 10
    FOBJ = sphere
    lu = np.zeros((2, D))
    lu[0, :] = -1
    lu[1, :] = 1
    iterMax = 2000
    target = 0.0
    for i in range(30):
        globalBest, globalBestParams, globalBestPerIter, optimumIter = WO(lu, iterMax, FOBJ, target)
        print(f"Global Best = {globalBest}\n")
        plt.plot(globalBestPerIter)
        plt.xlabel("Iteration")
        plt.ylabel("Global Best")
        plt.grid(True)
        plt.show(block=False)
        plt.pause(2)
        plt.close()