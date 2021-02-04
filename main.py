from numpy.core.numeric import full
import Utilities
import sys
import numpy
from Network import Network
import time
import copy 
import matplotlib.pyplot as plt 

def get_set(folder, grid_width, grid_height):
    files = Utilities.GetFiles(folder)
    
    training_set = list()
    results_set = list()
    for f in files:
        if "Spaceships" in f:
            # skip spaceships as we use them for testing the network
            continue
        
        fullMatrix = numpy.zeros((grid_width, grid_height))
        matrix = Utilities.ParseRLEFile(f)
        new_dim = [len(matrix),len(matrix[0])]
        if new_dim[0] >grid_width or new_dim[1] >grid_height:
            # skip on larger then 100x100 files
            continue  
        
        Utilities.CopyMatrix(matrix,fullMatrix)
        n_dim_matrix = Utilities.GetNDimensionalMatrix(fullMatrix)
        
        isOscillator = 1 if "Osci" in f else 0
        
        training_set.append(n_dim_matrix)
        # training_set.append((n_dim_matrix, [[isOscillator]]))
        results_set.append(numpy.array([isOscillator]))
    training_set = [numpy.reshape(x, (grid_width*grid_height, 1)) for x in training_set]
    training_result = [numpy.array([y]) for y in results_set]

    full_set = zip(training_set, training_result)
    return full_set

def getTestInput(patternName,grid_width, grid_height):
    fullMatrix = numpy.zeros((grid_width, grid_height))
    matrix = Utilities.ParseRLEFile(f"{patternName}")
    if len(matrix)>grid_width or len(matrix[0])>grid_height :
        return None
    Utilities.CopyMatrix(matrix,fullMatrix)
    newMatrix = []
    for i in range(len(fullMatrix)):
        for j in range(len(fullMatrix[0])):
            newMatrix.append([fullMatrix[i][j]])  
    return newMatrix

    
def outputPlot(successfull_classifications ,total, netowrkConfig, filename):
    x = list(range(0,netowrkConfig['epochs']))
    y = successfull_classifications

    y_max = []
    for i in range(netowrkConfig['epochs']):
        y_max.append(total)

    plt.plot(x, y_max, label = "total patterns", color='red', linestyle='dashed', linewidth = 3)

    plt.plot(x,y, label ="succesfully classified")
    plt.xlabel("Epochs")
    plt.ylabel("Successfull classifications")
    plt.title(f"Epochs:{netowrkConfig['epochs']}|Layers:{netowrkConfig['hidden_layers']}|Batch:{netowrkConfig['batch_size']}|Train rate:{netowrkConfig['training_rate']}|Success Thresh:{netowrkConfig['success_threshold']}", fontsize=12)    
    plt.savefig(filename+".png")


def testSpaceships(net, config):
    files = Utilities.GetFiles('patterns')
    oscilators = 0
    total = 0
    for f in files:
        if not "spaceships\\" in f.lower():
            continue 
        testM = getTestInput(f,config["width"],config["height"])
        if not testM == None:
            result = net.classify(testM)[0][0]
            isOscillator = result > config['success_threshold']
            oscilators += 1 if isOscillator else 0
            total += 1
            patternName = f[f.rfind('\\')+len('\\'):f.find('.rle')]
            # print(f"{patternName} is Oscillator: {'True' if isOscillator else 'False'} {result}")
    return ((oscilators*100)/total)

if __name__ == "__main__":


    config = {
        "width" : 100,
        "height" : 100,
        "success_threshold" : 0.95,
        "hidden_layers" : [100,100,100,30],
        "epochs" : 200,
        "batch_size" : 10,
        "training_rate" : 0.5
    }

    net = Network([config["width"]*config["height"]] + config["hidden_layers"] + [1], config)

    if len(sys.argv) >= 3:
        # first argument will be network serialized data
        net.load_network(sys.argv[1])

        # rest of the arguments will be the files to check
        for i in range(2, len(sys.argv)):
            testName = sys.argv[i]
            if testName.lower() == "spaceships":
                spaceShipsRatio = testSpaceships(net, config)
                print(f"{spaceShipsRatio} SpaceShips were classified as Oscillators")
                break
            else:
                testM = getTestInput(testName,config["width"],config["height"])
                result = net.classify(testM)[0][0]
                print(f"{testName} is Oscillator: {'True' if result > config['success_threshold'] else 'False'} {result}")
    else:
        startParsing = time.time()
        training_set = get_set("patterns", config["width"],config["height"])
        test_set = copy.deepcopy(training_set)
        endParsing = time.time()
        print(f"Parsing Time: {endParsing-startParsing}")
        


        startTraining = time.time()
        successfull_classifications = net.train(training_set, test_data=test_set)
        endTraining = time.time()
        filename = f"{config['epochs']}_{config['hidden_layers']}_{config['batch_size']}_{config['training_rate']}_{config['success_threshold']}"

        net.dump_network(filename+".pickle")
        outputPlot(successfull_classifications, 823, config, filename)

        print(f"Training Time: {endTraining-startTraining}")
        
        
        # test(config["width"],config["height"])
    

    