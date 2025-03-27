# FuzzyCoco parameter description


## Fuzzy system parameters

The fuzzy system parameters are the following:
- **fixedVars**: a boolean value indicating whether the variables of the fuzzy system are fixed or not. If it is the case
then the number of variables are equal to nbMaxVarPerRule, and they are chosen in the order of the variables in the database.
- **nbRules**: the number of rules in the fuzzy system. 
- **nbMaxVarPerRule**: the maximum number of variables per rule.
- **nbOutVars**: the number of output variables. (In case of multiple output variables, the fitness parameters are computed by averaging the fitness parameters of each output variable)
- **nbInSets**: the number of input sets per variable. (Usually 2, stay the same for all variables)
- **nbOutSets**: the number of output sets per variable.
- **inVarsCodeSize**: the number of bits used to code the input variables.
- **outVarsCodeSize**: the number of bits used to code the output variables.
- **inSetsCodeSize**: the number of bits used to code one input set's parameters
- **outSetsCodeSize**: the number of bits used to code one output set's parameters
- **inSetsPosCodeSize**: the number of bits used to code the position of the input sets.
- **outSetPosCodeSize**: the number of bits used to code the position of the output sets.

## Co-evolution parameters

- **maxGenPop1**: the maximum number of generations for the population of the membership functions.
- **maxFitPop1**: the maximum fitness for the population of the membership functions.
- **elitePop1**: the number of elite individuals in the population of the membership functions.
- **popSizePop1**: the size of the population of the membership functions.
- **cxProbPop1**: the crossover probability for the population of the membership functions.
- **mutFlipIndPop1**: the probability that an individual is a target for a mutation in the population of the membership functions.
- **mutFlipBitPop1**: the probability that a bit of an individual is mutated in the population of the membership functions.
- **elitePop2**: the number of elite individuals in the population of the rules.
- **popSizePop2**: the size of the population of the rules.
- **cxProbPop2**: the crossover probability for the population of the rules.
- **mutFlipIndPop2**: the probability that an individual is a target for a mutation in the population of the rules.
- **mutFlipBitPop2**: the probability that a bit of an individual is mutated in the population of the rules.


## Fitness parameters

- **sensitivityW**: the weight of the sensitivity in the fitness.
- **specificityW**: the weight of the specificity in the fitness.
- **accuracyW**: the weight of the accuracy in the fitness.
- **ppvW**: the weight of the ppv in the fitness.
- 
- **rmseW**: the weight of the rmse in the fitness. 2 pow(-rmse) pour maximiser la fitness
- **rrseW**: the weight of the rrse in the fitness. (relative error par rapport a la valeur attendu, moins sensible au grande valeur attendu)
- **raeW**: the weight of the rae in the fitness. (relative error en absolu)
- **mxeW**: the weight of the mxe in the fitness. 
- **distanceThresholdW**: the weight of the distance threshold in the fitness. (Adm pour moi se concentrer moins sur les grand erreur) on prend tout les correct positif on calcul la moyenne et on calcul la distance avec la limite le threshold, on essaye de maximiser cette dist, pareil pour les negatif
- **distanceMinThresholdW**: the weight of the distance min threshold in the fitness. (pareil mais on prend que le point le plus pres) prerformance est pas top
- **dontCareW**: the size of the model (1/the number of conditions in the rules). 1/nbVariable = environ la complexité
- **overLearnW**: the weight of the over learn in the fitness. 
- **threshold**: the threshold used to compute the fitness.
- **threshActivated**: a boolean value indicating whether the threshold is activated or not.
