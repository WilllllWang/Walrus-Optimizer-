import numpy as np
from scipy.stats.qmc import Halton
from matplotlib import pyplot as plt


def getTwoBest(costs):
    bestIdx = np.argmin(costs)
    secondCost = float('inf') 
    secondIdx = -1
    for i in range(len(costs)):
        if i != bestIdx and costs[i] < secondCost:
            secondCost = costs[i]
            secondIdx = i
    return bestIdx, secondIdx



def wo(lu, iterMax, FOBJ, target):
    # Independent variables
    D = lu.shape[1]     # Dimension
    LB = lu[0]          # LowerBound
    UB = lu[1]          # Upper bound
    nPop = 100          # Population size should be divisible by 10

    # Initialization
    walruses = LB + np.random.rand(nPop, D) * (UB - LB)

    # Quasi-Monte Carlo method with Halton sequence
    HS = Halton( d=D)
    # Fitness
    costs = np.array([FOBJ(w) for w in walruses])

    # Global best and second best
    bestIdx, secondIdx = getTwoBest(costs)
    bestW = walruses[bestIdx].copy()
    secondW = walruses[secondIdx].copy()
    globalBest = costs[bestIdx]
    globalBestPerIter = np.full(iterMax, target)
    optimumIter = -1

    # Main loop
    for iter in range(iterMax):
        # Signals
        ## Danger signal
        alpha = 1 - iter / iterMax
        A = 2 * alpha               
        R = 2 * np.random.randint(0, 1) - 1
        dangerSig = A * R
        ## Safety signal
        safeSig = np.random.randint(0, 1)

        # Exploration
        ## Migration
        if abs(dangerSig) >= 1:             
            for w in walruses:
                # Choose two random vigilantes
                vig1, vig2 = -1, -1
                while vig1 == w or vig2 == w or vig1 == vig2:
                    vig1, vig2 = np.random.randint(0, nPop, size=2)
                # Get the weighted migration step
                beta = 1 - 1 / (1 + np.exp((-(iter - (iterMax / 2)) / iterMax) * 10))
                migrationStep = (vig1 - vig2) * beta * (np.random.randint(0, 1)**2)
                # Update position and keep in search range
                w = w + migrationStep
                w = np.clip(w, LB, UB)

        # Exploitation
        else:
            ## Roosting
            if safeSig >= 0.5:              
                # Male
                walruses[: nPop * 0.45] = HS.random(nPop * 0.45)

                # Female
                for i in range(nPop * 0.45, nPop * 0.9):
                    w[i] = w[i] + alpha * (w[i - nPop * 0.45] - w[i]) + (1 - alpha) * (bestW - w[i])
                    w[i] = np.clip(w[i], LB, UB)
                # Juvenile
                for w in walruses[nPop * 0.9: ]:
                    LF = 
                    O = bestW + w * LF
                    w = (O - w) * np.random.randint(0, 1)
                    w = np.clip(w, LB, UB)


            else:
                ## Foraging
                if abs(dangerSig) >= 0.5:
                    for w in walruses:
                        a1 = beta * np.random.randint(0, 1) - beta
                        a2 = beta * np.random.randint(0, 1) - beta   
                        b1 = np.tan(np.random.uniform(0 + 1e-12, np.pi - 1e-12))
                        b2 = np.tan(np.random.uniform(0 + 1e-12, np.pi - 1e-12))
                        x1 = bestW - a1 * b1 * abs(bestW - w)
                        x2 = bestW - a2 * b2 * abs(secondW - w)
                        w = (x1 + x2) / 2
                        w = np.clip(w, LB, UB)

                ## Fleeing
                else:      
                    for w in walruses:
                        w = w * R - abs(bestW - w) * (np.random.randint(0, 1)**2) 
                        w = np.clip(w, LB, UB)             

        costs = np.array([FOBJ(w) for w in walruses])
        bestIdx, secondIdx = getTwoBest(costs)
        bestW = walruses[bestIdx].copy()
        secondW = walruses[secondIdx].copy()
        globalBest = costs[bestIdx]
        globalBestPerIter[iter] = globalBest
        
        # If optimum found
        if globalBest == target:
            optimumIter = iter
            break
    
    return globalBest, bestW, globalBestPerIter, optimumIter







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
    globalBest, globalBestParams, globalBestPerIter, optimumIter = wo(lu, iterMax, FOBJ, target)
    print(f"Global Best = {globalBest}\n")
    # Optional
    plt.plot(globalBestPerIter)
    plt.xlabel("Iteration")
    plt.ylabel("Global Best")
    plt.grid(True)
    plt.show()

