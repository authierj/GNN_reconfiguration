def method_default_args():  # networkID
    defaults = {}

    defaults['network'] = 'node4'
    defaults['epochs'] = 300
    defaults['batchSize'] = 200
    defaults['lr'] = 1e-3  # NN learning rate
    defaults['hiddenSize'] = 100  # 200  # hidden layer for NN
    defaults['softWeight'] = 100  # this is lambda_g in the paper
    # whether lambda_g is time-varying or not
    defaults['useAdaptiveWeight'] = False
    # whether lambda_g is scalar, or vector for each constraint
    defaults['useVectorWeight'] = False
    defaults['adaptiveWeightLr'] = 1e-2
    # use 100 if useCompl=False

    # defaults['softWeightEqFrac'] = 0.5
    defaults['useCompl'] = True
    defaults['useTrainCorr'] = False
    defaults['useTestCorr'] = False
    defaults['corrTrainSteps'] = 5  # 20 for train correction
    # 5 for with train correction, 500 for without train correction
    defaults['corrTestMaxSteps'] = 5
    defaults['corrEps'] = 1e-3
    defaults['corrLr'] = 1e-4  # 1e-4  # use 1e-5 if useCompl=False
    defaults['corrMomentum'] = 0.5  # 0.5
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50

    return defaults


def change(key, value):
    defaults = method_default_args()
