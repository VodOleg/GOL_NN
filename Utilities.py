import re, os
import numpy as np

def ParseRLEFile(inputFile, LiveCell=1, DeadCell=0):
    """
        This function accepts file path as a parameter.
        The file is expected to be valid game of life REL file.
        This parser is only valid for files that respect the standart written in (https://www.conwaylife.com/wiki/Run_Length_Encoded)
        The function returns a Matrix of integers, where 1- live cell and 0- dead cell.
        Written by Oleg Vodsky.
    """
    x = 0
    y = 0
    matrix = []
    wholePattern = ""

    with open(inputFile, "r") as input:
        for line in input:
            # first we look for dimension line and update dimensions
            match = re.search(r'x\s*=\s(\d+)\s*,\s*y\s*=\s(\d+)', line)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                continue
            if line[0] == '#':
                continue
            # from here we join first all the lines to a single string
            wholePattern += line[:-1]
    
    rows = wholePattern.split('$')
    numbers_regex = r'(\d*b)|(\d*o)'
    for row in rows:
        matrix_row = []
        word = row
        while True:
            matches = re.search(numbers_regex, word)
            if matches:
                subgroup = ""
                if matches.group(1):
                    # found b (dead cell)
                    subgroup = matches.group(1)
                    repeats = re.search(r'(\d+)|(b)', subgroup)
                    num_of_repeats = 1
                    if repeats.group(1):
                        num_of_repeats = int(repeats.group(1))
                    for i in range(0, num_of_repeats):
                        matrix_row.append(DeadCell)
                elif matches.group(2):
                    # found o (live cell)
                    subgroup = matches.group(2)
                    repeats = re.search(r'(\d+)|(o)', subgroup)
                    num_of_repeats = 1
                    if repeats.group(1):
                        num_of_repeats = int(repeats.group(1))
                    for i in range(0, num_of_repeats):
                        matrix_row.append(LiveCell)

                
                # write the group to the matrix
                
                word = word[len(subgroup):]
            else:
                break
        
        # buffer last line with dead cells
        while len(matrix_row) < x:
            matrix_row.append(DeadCell)
        matrix.append(matrix_row)
    # buffer missing lines with dead clles
    while len(matrix)<y:
        empty_matrix_row = []
        for i in range(0,x):
            empty_matrix_row.append(DeadCell)
        matrix.append(empty_matrix_row)
    return matrix
    

def GetFiles(path):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.rle']
    return result
                    
def CopyMatrix(src, dst):
    for i in range(0, len(src)):
        for j in range(0, len(src[0])):
            dst[i][j] = src[i][j]

def GetNDimensionalMatrix(matrix):
    newMatrix = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            newMatrix.append(matrix[i][j])
    return np.array(newMatrix)

def sigmoid(var):
    return 1.0/(1.0+np.exp(-var))

def sigmoid_prime(var):
    return sigmoid(var)*(1-sigmoid(var))

def binary_step(var):
    value = list()
    for i in range(len(var)):
        for j in range(len(var[i])):
            if var[i][j] < 0:
                value.append(np.array([0]))
            else:
                value.append(np.array([1]))
    value = np.array(value)
    return value

def binary_step_prime(var):
    # derivate of binary step function always 0
    # unless the var is 0 than the deriviate is undefined
    # we will still return 0 so the execution can be complete
    return 0
