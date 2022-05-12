from ctypes import util
from distutils import core
import re
from statistics import variance
import cv2
from cv2 import threshold
import numpy as np
import random
from configparser import ConfigParser 
import os
import os.path as filePath
from time import process_time
from matplotlib import pyplot as plt
from operator import index, le
import math
import re
import pandas as pd
from sklearn.utils import shuffle

program_start = process_time()

config = ConfigParser()
config.read("userConfigurations.ini")
imagesPath = config["config"]["imagesPath"]
outputPath = config["config"]["outputPath"]

# rgbWeights = [float(num) for num in config["inputs"]["rgbWeights"].split(",")]

# linearMaskSize = config["inputs"]["linearMaskSize"]
# linearWeights = [float(num) for num in config["inputs"]["linearWeights"].split(",")]
# medianMaskSize = config["inputs"]["medianMaskSize"]
# medianWeights = [float(num) for num in config["inputs"]["medianWeights"].split(",")]

# gaussianStdDeviation = config["inputs"]["gaussianStdDeviation"]
# saltNpepperStrength = config["inputs"]["saltNpepperStrength"]

# imageQuantizationLevel = config["inputs"]["imageQuantizationLevel"]

# sobelXdirection = [float(num) for num in config["inputs"]["sobelXdirection"].split(",")]
# sobelYdirection = [float(num) for num in config["inputs"]["sobelYdirection"].split(",")]
# improvedSobelXdirection = [float(num) for num in config["inputs"]["improvedSobelXdirection"].split(",")]
# improvedsobelYdirection = [float(num) for num in config["inputs"]["improvedsobelYdirection"].split(",")]
# prewittXdirection = [float(num) for num in config["inputs"]["prewittXdirection"].split(",")]
# prewittYdirection = [float(num) for num in config["inputs"]["prewittYdirection"].split(",")]
# robertsXdirection = [float(num) for num in config["inputs"]["robertsXdirection"].split(",")]
# robertsYdirection = [float(num) for num in config["inputs"]["robertsYdirection"].split(",")]

erosionKernel = [float(num) for num in config["inputs"]["erosionKernel"].split(",")]
dilationKernel = [float(num) for num in config["inputs"]["dilationKernel"].split(",")]

featuresFileCSV = config["inputs"]["featuresFileCSV"]

numClusters = config["inputs"]["clusters"]
global clusters
clusters= [[] for i in range(int(numClusters))]

try:
    os.mkdir(outputPath)
    # os.mkdir(outputPath + "/grayImages")
    # os.mkdir(outputPath + "/colorSpecturm")
    # os.mkdir(outputPath + "/medianFilter")
    # os.mkdir(outputPath + "/linearFilter")
    # os.mkdir(outputPath + "/saltNpeper")
    # os.mkdir(outputPath + "/gaussian")
    # os.mkdir(outputPath + "/histogram")
    # os.mkdir(outputPath + "/histogramEqualizations")
    # os.mkdir(outputPath + "/imageQuantization")
    # os.mkdir(outputPath + "/sobelOperator")
    # os.mkdir(outputPath + "/improvedSobel")
    # os.mkdir(outputPath + "/prewittOperator")
    # os.mkdir(outputPath + "/compassOperator")
    # os.mkdir(outputPath + "/robertsOperator")
    # os.mkdir(outputPath + "/imageErosion")
    # os.mkdir(outputPath + "/imageDilation")
    # os.mkdir(outputPath + "/histogramThreshold")
    # os.mkdir(outputPath + "/kmeans")
    os.mkdir(outputPath + "/histogramSegmentation")
    os.mkdir(outputPath + "/contours")
except OSError as error:
    print(error)

def writeToFile(path, count, image):
    if path == 1:
        cv2.imwrite(filePath.join(outputPath + "/grayImages/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 2:
        cv2.imwrite(filePath.join(outputPath + "/colorSpecturm/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 5:
        cv2.imwrite(filePath.join(outputPath + "/medianFilter/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 6:
        cv2.imwrite(filePath.join(outputPath + "/linearFilter/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 7:
        cv2.imwrite(filePath.join(outputPath + "/saltNpeper/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 8:
        cv2.imwrite(filePath.join(outputPath + "/gaussian/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 9:
        plt.savefig(outputPath+"/histogram/"+"image.jpg".split(".")[0]+str(count)+".jpg")
    elif path == 10:
        cv2.imwrite(filePath.join(outputPath + "/histogramEqualizations/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 12:
        cv2.imwrite(filePath.join(outputPath + "/imageQuantization/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 13:
        cv2.imwrite(filePath.join(outputPath + "/sobelOperator/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 14:
        cv2.imwrite(filePath.join(outputPath + "/prewittOperator/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 15:
        cv2.imwrite(filePath.join(outputPath + "/robertsOperator/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 16:
        cv2.imwrite(filePath.join(outputPath + "/imageErosion/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 17:
        cv2.imwrite(filePath.join(outputPath + "/imageDilation/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 18:
        cv2.imwrite(filePath.join(outputPath + "/histogramThreshold/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 19:
        cv2.imwrite(filePath.join(outputPath + "/kmeans/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 20:
        cv2.imwrite(filePath.join(outputPath + "/improvedSobel/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 21:
        cv2.imwrite(filePath.join(outputPath + "/compassOperator/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 22:
        cv2.imwrite(filePath.join(outputPath + "/histogramSegmentation/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 23:
        cv2.imwrite(filePath.join(outputPath + "/contours/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    elif path == 24:
        cv2.imwrite(filePath.join(outputPath + "/hysteresis/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
    else:
        print("Path not found")

def greyScaleImage2(img): 
    height = len(img)
    widht = len(img[0])
    
    grayImage = np.empty([height, widht], dtype=np.uint8)
    for i in range(height):
            for j in range(widht):
                grayImage[i][j] = int(img[i][j][0]*0.2126 + img[i][j][1]*0.7152 + img[i][j][2] * 0.0722)
    return grayImage

def greyScaleImage(img, count): 
    height = len(img)
    widht = len(img[0])
    
    grayImage = np.empty([height, widht], dtype=np.uint8)
    for i in range(height):
            for j in range(widht):
                grayImage[i][j] = int(img[i][j][0]*0.2126 + img[i][j][1]*0.7152 + img[i][j][2] * 0.0722)
    writeToFile(1, count, grayImage)

def imagetoRGB(img, count, rgbWeights):
    blueChannel = img[:,:,0]
    greenChannel = img[:,:,1]
    redChannel = img[:,:,2]
    
    blueAvg = np.dot(blueChannel, rgbWeights[2])
    greenAvg = np.dot(greenChannel, rgbWeights[1])
    redAvg = np.dot(redChannel, rgbWeights[0])
    
    blueInt = blueAvg.astype(np.uint8)
    greenInt = greenAvg.astype(np.uint8)
    redInt = redAvg.astype(np.uint8)

    image = cv2.merge([redInt, greenInt, blueInt])
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    writeToFile(2, count, image)

def saltNpepper(img, count, strength):
    img = greyScaleImage2(img)
    height = len(img)
    widht = len(img[0])
    sNp = np.empty([height, widht], dtype=np.uint8)
    threshold = 1 - float(strength) 

    for i in range(height):
        for j in range(widht):
            randNum = random.random()
            if randNum > threshold:
                sNp[i][j] = 255
            elif randNum < float(strength):
                sNp[i][j] = 0
            else:
                sNp[i][j] = img[i][j]
    # cv2.imshow('image', sNp)
    # cv2.waitKey(0)
    writeToFile(7, count, sNp)

def gaussian(img, count, std_deviation):
    img = greyScaleImage2(img)
    
    height = len(img)
    widht = len(img[0])
    mean = 0

    # print(float(std_deviation))
    gauss = np.random.normal(mean, float(std_deviation), (height,widht))
    # print(gauss[0][0])
    gauss = gauss.reshape(height, widht)
    noissyImage = img - gauss
    noissyImage = noissyImage.astype(np.uint8)
    # print(noissyImage[0][0])
    # cv2.imshow('image', noissyImage)
    # cv2.waitKey(0)
    writeToFile(8, count, noissyImage)
    
def medianFilter(img, count, filterSize, weights):
    height = len(img)
    widht = len(img[0])
    weights = np.reshape(weights, (int(filterSize), int(filterSize)))
    weights = weights.astype(np.uint8)
    medianVal = int(np.median(range(int(filterSize))))
    testImage = cv2.copyMakeBorder(greyScaleImage2(img),medianVal,medianVal,medianVal,medianVal,cv2.BORDER_CONSTANT,value=0)
    blankImage = np.zeros((height,widht), np.uint8)

    for i in range(medianVal, height+medianVal):
        for j in range(medianVal, widht+medianVal):
            # print("Old pixel: " + str(testImage[i][j]) + " at Index " + str(i) + " " + str(j))
            testArea = testImage[i-medianVal:i+medianVal+1,j-medianVal:j+medianVal+1]
            # print(testArea)
            sortList = []
            for x in range(len(testArea)):
                for y in range(len(testArea)):
                    sortList.extend([testArea[x][y]] * weights[x][y])
            sortList.sort()
            # print(sortList)
            # print("New pixel: " + str(int(np.median(sortList))) + " at Index " + str(i) + " " + str(j))
            blankImage[i-medianVal][j-medianVal] = int(np.median(sortList))
            # print(blankImage[i-medianVal][j-medianVal])
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(5, count, blankImage)

def linearFilter(img, count, filterSize, weights):
    height = len(img)
    widht = len(img[0])
    totalWeight = sum(weights)
    weights = np.reshape(weights, (int(filterSize), int(filterSize)))
    medianVal = int(np.median(range(int(filterSize))))
    testImage = cv2.copyMakeBorder(greyScaleImage2(img),medianVal,medianVal,medianVal,medianVal,cv2.BORDER_CONSTANT,value=0)
    blankImage = np.zeros((height,widht), np.uint8)

    for i in range(medianVal, height + medianVal):
        for j in range(medianVal, widht + medianVal):
            # print("Old pixel: " + str(testImage[i][j]) + " at Index " + str(i) + " " + str(j))
            testArea = testImage[i-medianVal:i+medianVal+1, j-medianVal:j+medianVal+1]
            # print(testArea)
            d = 1/(int(totalWeight))
            foo = np.dot(weights, d)
            result = []
            for x in range(len(testArea)):
                for y in range(len(testArea)):
                    temp = testArea[x][y] * foo[x][y]
                    result.append(np.round(temp))
            # print("New pixel: " + str(int(np.median(result))) + " at Index " + str(i) + " " + str(j))
            blankImage[i-medianVal][j-medianVal] = int(sum(result))
            # print(blankImage[i-medianVal][j-medianVal])
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(6, count, blankImage)

def imageHistogram(img, count, type=None):
    height = len(img)
    widht = len(img[0])
    hist = np.zeros(shape=(256), dtype=np.uint64)
    # plt.show()
    if type == "return":
        for i in range(height):
            for j in range(widht):
                hist[img[i][j]] = hist[img[i][j]]+1
        return hist
    else: 
        greyImage = greyScaleImage2(img)
        for i in range(height):
            for j in range(widht):
                hist[greyImage[i][j]] = hist[greyImage[i][j]]+1
        plt.plot(hist)
        writeToFile(9, count, hist)
    plt.close()

def imageHistogramEqualization(img, count):
    rows = len(img)
    cols = len(img[0])
    greyImage = greyScaleImage2(img)

    hist = np.zeros(shape=(256), dtype=np.uint64)
    for i in range(rows):
        for j in range(cols):
            hist[greyImage[i][j]] = hist[greyImage[i][j]]+1
    plt.plot(hist)
    # plt.show()
    plt.close()

    totalPixels = rows * cols
    CFD = []
    prevSum = 0
    prob = 1.0/totalPixels
    blankImage = np.zeros((rows,cols), np.uint8)

    for i in range(0, 256):
        # newSum = hist[i]*1.0/totalPixels
        newSum = hist[i] * prob
        prevSum = prevSum + newSum 
        CFD.append(prevSum)

    for x in range(rows):
        for y in range(cols):
            res = greyImage[x][y]
            test = CFD[res]
            newPixel = 255 * test
            blankImage[x,y] = int(newPixel)

    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(10, count, blankImage)

def imageQuantization(img, count, level):
    height = len(img)
    widht = len(img[0])
    greyImage = greyScaleImage2(img)
    blankImage = np.zeros((height, widht), dtype=np.uint8)
    testLevel = int(level) - 1
    # print(testLevel)
    quantLevel = 255/testLevel
    # print(quantLevel)
    for i in range(height):
        for j in range(widht):
            test = round(greyImage[i,j]/quantLevel)
            # print(test)
            blankImage[i,j] = round(test * quantLevel)
            # print(blankImage[x,y])
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(11, count, blankImage)

def averageTime(timeList):
    return sum(timeList) / len(timeList)

def imageQuantization2(img, level):
    height = len(img)
    widht = len(img[0])
    greyImage = greyScaleImage2(img)
    blankImage = np.zeros((height, widht), dtype=np.uint8)
    testLevel = int(level) - 1
    # print(testLevel)
    quantLevel = 255/testLevel
    # print(quantLevel)
    for i in range(height):
        for j in range(widht):
            test = round(greyImage[i,j]/quantLevel)
            # print(test)
            blankImage[i,j] = round(test * quantLevel)
    return blankImage

def meanSquaredQuantizationError(img, count, level):
    height = len(img)
    widht = len(img[0])
    greyImage = greyScaleImage2(img)
    quantImage = imageQuantization2(img, int(level))
    result = greyImage - quantImage
    MSQE = result ** 2
    foo = 1/(height * widht)
    test = np.dot(MSQE, foo)
    aList = []
    for i in range(len(test)):
        for j in range(len(test)):
            aList.append(test[i][j])
    total = sum(aList)
    f = open("MSQE.txt", "a")
    f.writelines("The MSQE for image" + str(count) + ": " + str(total) + "\n")
    f.close()

def sobelOperator(img, sobelXdirection, sobelYdirection, count):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    height = len(img)
    width = len(img[0])
    blankImage = np.zeros((height,width), np.uint8)
    num = math.sqrt(len(sobelXdirection))
    X_direct = np.reshape(sobelXdirection, (int(num), int(num)))
    Y_direct = np.reshape(sobelYdirection, (int(num), int(num)))
    for i in range(1, height-1):
        for j in range(1, width-1):
            # print("Old pixel: " + str(test1[i][j]) + " at Index " + str(i) + " " + str(j))
            testArea = img[i-1:i+2, j-1:j+2]
            X_results, Y_results = [], []
            for x in range(len(testArea)):
                for y in range(len(testArea)):
                    Y_val = testArea[x][y] * Y_direct[x][y]
                    X_val = testArea[x][y] * X_direct[x][y]
                    Y_results.append(Y_val)
                    X_results.append(X_val)
            # # print("New pixel: " + str(int(np.median(result))) + " at Index " + str(i) + " " + str(j))
            result1 = np.sum(X_results)
            result2 = np.sum(Y_results)
            blankImage[i][j] = int(np.sqrt(result1**2 + result2**2))
            # print(blankImage[i][j])
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(13, count, blankImage)

def improvedSobelOperator(img, sobelXdirection, sobelYdirection, count):
    height = len(img)
    width = len(img[0])
    blankImage = np.zeros((height,width), np.uint8)
    num = math.sqrt(len(sobelXdirection))
    X_direct = np.reshape(sobelXdirection, (int(num), int(num)))
    Y_direct = np.reshape(sobelYdirection, (int(num), int(num)))
    # X_direct = np.multiply(np.reshape(sobelXdirection, (int(num), int(num))), (1/32))
    # Y_direct = np.multiply(np.reshape(sobelYdirection, (int(num), int(num))), (1/32))
    for i in range(1, height-1):
        for j in range(1, width-1):
            # print("Old pixel: " + str(test1[i][j]) + " at Index " + str(i) + " " + str(j))
            testArea = img[i-1:i+2, j-1:j+2]
            X_results, Y_results = [], []
            for x in range(len(testArea)):
                for y in range(len(testArea)):
                    Y_results.append(testArea[x][y] * Y_direct[x][y])
                    X_results.append(testArea[x][y] * X_direct[x][y])
            # # print("New pixel: " + str(int(np.median(result))) + " at Index " + str(i) + " " + str(j))
            result1 = np.sum(X_results)
            result2 = np.sum(Y_results)
            blankImage[i][j] = int(np.sqrt(result1**2 + result2**2))
            # print(blankImage[i][j])
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(20, count, blankImage)

def prewittOperator(img, prewittXdirection, prewittYdirection,  count):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    height = len(img)
    width = len(img[0])
    blankImage = np.zeros((height,width), np.uint8)
    num = math.sqrt(len(prewittXdirection))
    X_direct = np.reshape(prewittXdirection, (int(num), int(num)))
    Y_direct = np.reshape(prewittYdirection, (int(num), int(num)))
    for i in range(1, height-1):
        for j in range(1, width-1):
            # print("Old pixel: " + str(test1[i][j]) + " at Index " + str(i) + " " + str(j))
            testArea = img[i-1:i+2, j-1:j+2]
            X_results, Y_results = [], []
            for x in range(len(testArea)):
                for y in range(len(testArea)):
                    Y_val = testArea[x][y] * Y_direct[x][y]
                    X_val = testArea[x][y] * X_direct[x][y]
                    Y_results.append(Y_val)
                    X_results.append(X_val)
            # # print("New pixel: " + str(int(np.median(result))) + " at Index " + str(i) + " " + str(j))
            result1 = np.sum(X_results)
            result2 = np.sum(Y_results)
            blankImage[i][j] = int(np.sqrt(result1**2 + result2**2))
            # print(blankImage[i][j])
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(14, count, blankImage)

def robertsOperator(img, robertsXdirection, robertsYdirection, count):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    height = len(img)
    width = len(img[0])
    blankImage = np.zeros((height,width), np.uint8)
    num = math.sqrt(len(robertsXdirection))
    X_direct = np.reshape(robertsXdirection, (int(num), int(num)))
    Y_direct = np.reshape(robertsYdirection, (int(num), int(num)))
    for i in range(1, height-1):
        for j in range(1, width-1):
            # print("Old pixel: " + str(test1[i][j]) + " at Index " + str(i) + " " + str(j))
            testArea = img[i-1:i+1, j-1:j+1]
            X_results, Y_results = [], []
            for x in range(len(testArea)):
                for y in range(len(testArea)):
                    Y_val = testArea[x][y] * Y_direct[x][y]
                    X_val = testArea[x][y] * X_direct[x][y]
                    Y_results.append(Y_val)
                    X_results.append(X_val)
            # # print("New pixel: " + str(int(np.median(result))) + " at Index " + str(i) + " " + str(j))
            result1 = np.sum(X_results)
            result2 = np.sum(Y_results)
            blankImage[i][j] = int(np.sqrt(result1**2 + result2**2))
            # print(blankImage[i][j])
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(15, count, blankImage)

def imageErosion(img, kernel, count):
    height = len(img)
    width = len(img[0])
    num = math.sqrt(len(kernel))
    kernel = np.reshape(kernel, (int(num), int(num)))
    for itr in range(2):
        blankImage = np.zeros((height, width),dtype=np.uint8)
        for x in range(height):
            for y in range(width):
                results = []
                for i in range(3):
                    for j in range(3):
                        a = x - 1 + i
                        b = y - 1 + j
                        if(a>=0 and b>=0 and a<height and b>=0 and b<width and int(kernel[i][j])==1):
                            results.append(img[a,b])
                if(len(results) > 0):
                    blankImage[x,y] = min(results)
                else:
                    blankImage[x,y] = img[x,y]
        img = blankImage.copy()
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # writeToFile(16, count, img)
    return img    

def imageDilation(img, kernel, count):
    height = len(img)
    width = len(img[0])
    num = math.sqrt(len(kernel))
    kernel = np.reshape(kernel, (int(num), int(num)))
    for itr in range(2):
        blankImage = np.zeros((height, width),dtype=np.uint8)
        for x in range(height):
            for y in range(width):
                results = []
                for i in range(3):
                    for j in range(3):
                        a = x - 1 + i
                        b = y - 1 + j
                        if(a>=0 and a<height and b>=0 and b<width and int(kernel[i][j])==1):
                            results.append(img[a,b])
                blankImage[x,y] = max(results)
        img = blankImage.copy()
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # writeToFile(17, count, img)
    return img    

def histogramThreshold(image, count):
    height = len(image)
    width = len(image[0])
    bins = 256
    hist, edges = np.histogram(image, bins)
    list1, list2, list3, list4 = [],[],[],[]
    for i in range(bins):
        list1.append(hist[i]/hist.max())

    list2 = edges[:-1]
    list3 = edges[1:]
    list4 = np.add(list2, list3)
    quotients = []
    for number in list4:
        quotients.append(number / 2.)

    weight1 = np.cumsum(list1)
    # print(weight1)
    weight2 = np.cumsum(list1[::-1])[::-1]
    # print(weight2)

    # Get the class means mu0(t)
    mean1 = np.multiply(list1, quotients)
    mean1 = np.cumsum(mean1)
    mean1 = mean1/weight1
    # print(mean1)

    mean2 = np.multiply(list1, quotients)
    mean2 = mean2[::-1]
    mean2 = np.cumsum(mean2)
    mean2 = mean2/weight2[::-1]
    mean2 = mean2[::-1]
    # print(mean2)

    list5, list6, list7, list8 = [],[],[],[]
    x = len(weight1)
    for i in range(x-1):
        list5.append(weight1[i])
    for i in range(1, x):
        list6.append(weight2[i])
    for i in range(x-1):
        list7.append(mean1[i])
    for i in range(1, x):
        list8.append(mean2[i])

    avg = np.subtract(list7, list8)
    avg = avg ** 2
    variance = np.multiply(list5, list6)
    variance = np.multiply(variance, avg)

    maxIndex = np.argmax(variance)

    t = quotients[maxIndex]
    blankImage=np.zeros((height, width), image.dtype)
    for i in range(height):
        for j in range(width): 
            if(image[i,j] <= t):
                blankImage[i,j] = 0
            else:
                blankImage[i,j] = 255
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(18, count, blankImage)
        
def kmeans(img, count, numClusters):
    height = len(img)
    width = len(img[0])
    centroid = random.sample(range(1, 257), int(numClusters))
    # centroid = [136,34,56]
    blankImage = np.zeros((height, width), dtype=np.uint8)
    centroidList= []

    while (centroid != centroidList):
        pixelMap = {}
        for x in range(height):
            for y in range(width):
                distance = []
                temp = pixelMap.get(img[x][y])
                if (temp == None):
                    for item in centroid:
                        distance.append(math.sqrt((item - img[x][y])**2))
                        # print{str(distance) + " " + str(img[x][y])}
                        variance = np.argmin(distance)
                        clusters[variance].append(img[x][y])
                        pixelMap[img[x][y]] = variance
                    blankImage[x][y] = variance
                # print(str(blankImage[x][y]) + " " + str(img[x][y]) + " index: " + str(x) + " " + str(y))
                else:
                    clusters[pixelMap.get(img[x][y])].append(img[x][y])
                    blankImage[x][y] = pixelMap.get(img[x][y])
                # print(str(blankImage[x][y]) + " " + str(img[x][y]) + " index: " + str(x) + " " + str(y))
        centroidList = centroid.copy()
        avgList = []
        for i in range(len(clusters)):
            avgList.append(int(np.sum(clusters[i])/len(clusters[i])))
    # print(avgList)

    centers = np.array(avgList)
    #print(labels)
    segImage = centers[blankImage]
    # cv2.imshow('image', segImage)
    # cv2.waitKey(0)
    segImage = segImage.astype('uint8')
    # cv2.imshow('image', segImage)
    # cv2.waitKey(0)
    writeToFile(19, count, segImage)

def compassOperator(img, count):
    height = len(img)
    width = len(img[0])
    blankImage = np.zeros((height, width), dtype=np.uint8)
    lists = []
    directions = np.array([
        [[-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]],

        [[0.0, 1.0, 2.0],
        [-1.0, 0.0, 1.0],
        [-2.0, -1.0, 0.0]],

        [[1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]],

        [[2.0, 1.0, 0.0],
        [1.0, 0.0, -1.0],
        [0.0, -1.0, -2.0]],

        [[1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]],

        [[0.0, -1.0, -2.0],
        [1.0, 0.0, -1.0],
        [2.0, 1.0, 0.0]],

        [[-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0]],

        [[-2.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0]]
    ])

    for filter in directions:
        test = np.zeros((height, width), dtype=np.uint8)
        for i in range(1, height-1):
            for j in range(1, width-1):
                    test[i][j] = int(np.sum(np.multiply(img[i-1:i+2, j-1:j+2], filter)))
                    # print(str(x[i][j]) + " at index: " + str(i) + str(j)) 
        lists.append(test)

    for x in lists:
        blankImage = np.maximum(x, blankImage)           
        # print("Done")

    finalImage = img
    for u in range(1,height-1):
        for v in range(1, width-1):
            finalImage[u][v] = math.pi/4 * blankImage[u][v]
    # cv2.imshow('image', finalImage)
    # cv2.waitKey(0)
    writeToFile(21, count, finalImage)

def histogramSegmentation(image, count):
    image = greyScaleImage2(img)
    height = len(image)
    width = len(image[0])
    bins = 256
    hist, edges = np.histogram(image, bins)
    list1, list2, list3, list4 = [],[],[],[]
    for i in range(bins):
        list1.append(hist[i]/hist.max())

    list2 = edges[:-1]
    list3 = edges[1:]
    list4 = np.add(list2, list3)
    quotients = []
    for number in list4:
        quotients.append(number / 2.)

    weight1 = np.cumsum(list1)
    # print(weight1)
    weight2 = np.cumsum(list1[::-1])[::-1]
    # print(weight2)

    # Get the class means mu0(t)
    mean1 = np.multiply(list1, quotients)
    mean1 = np.cumsum(mean1)
    mean1 = mean1/weight1
    # print(mean1)

    mean2 = np.multiply(list1, quotients)
    mean2 = mean2[::-1]
    mean2 = np.cumsum(mean2)
    mean2 = mean2/weight2[::-1]
    mean2 = mean2[::-1]
    # print(mean2)

    list5, list6, list7, list8 = [],[],[],[]
    x = len(weight1)
    for i in range(x-1):
        list5.append(weight1[i])
    for i in range(1, x):
        list6.append(weight2[i])
    for i in range(x-1):
        list7.append(mean1[i])
    for i in range(1, x):
        list8.append(mean2[i])

    avg = np.subtract(list7, list8)
    avg = avg ** 2
    variance = np.multiply(list5, list6)
    variance = np.multiply(variance, avg)
    maxIndex = np.argmax(variance)
    t = quotients[maxIndex]

    global finalThreshold
    finalThreshold = t
    global finalVariance
    finalVariance = np.sum(variance)/256
    global finalMean
    finalMean = int(avg.max())/256

    blankImage = np.zeros((height, width), image.dtype)
    for i in range(height):
        for j in range(width): 
            if(image[i,j] <= t):
                blankImage[i,j] = image[i,j]
            else:
                blankImage[i,j] = 0
    # cv2.imshow('image', blankImage)
    # cv2.waitKey(0)
    writeToFile(22, count, blankImage)
    binaryImage = np.zeros((height, width), image.dtype)
    for i in range(height):
        for j in range(width): 
                if(image[i,j] <= t):
                    binaryImage[i,j] = 255
                else:
                    binaryImage[i,j] = 0
    return binaryImage, blankImage

def contourImage(img, count):
    errosionImage = imageErosion(img, erosionKernel, count)
    dilationImage = imageDilation(errosionImage, dilationKernel, count)
    contour = np.subtract(dilationImage, errosionImage)
    # cv2.imshow('image', contour)
    # cv2.waitKey(0)
    # writeToFile(23, count, contour)
    return contour

def featuresCollection(img, count, contour, binary, image):
    hist = imageHistogram(img, count, "return")
    values = list()
    for index,value in enumerate(hist):
        if index<=finalThreshold and index != 0:
            values.extend([index]*value)
    median = np.median(values)
    stdDiv = math.sqrt(finalVariance)
    mean = finalMean
    variance = finalVariance
    perimeter = np.count_nonzero(contour == 255)
    area = np.count_nonzero(binary == 255)
    roundness = (perimeter*perimeter)/(4*math.pi*area)
    avgR = np.mean(image[:,:,2])
    avgG = np.mean(image[:,:,1])
    avgB = np.mean(image[:,:,0])
    meanRGB = (avgR + avgG + avgB)/3    
    f = open(os.path.join(outputPath+'/'+ featuresFileCSV), "a")
    f.writelines([str(area),str(","),str(perimeter),str(","),str(mean),str(","),str(median),str(","),str(meanRGB),str(","),str(roundness),str(","),str(variance),str(","),str(stdDiv),str(","),label,str("\n")])
    f.close()

def crossValidation(csvPath):
    accuracies = []
    data = shuffle(pd.read_csv(csvPath))
    data = data.round({'area': 3, 'perimeter': 3, 'mean': 3, 'median': 3, 'meanRGB': 3, 'roundness': 3, 'variance': 3, 'stdDiv': 3})
    testPercent = .90
    trainPercent = 1 - testPercent
    cross = 10
    labels = data.iloc[:,-1]
    temp = set(labels.values.tolist())
    result, probability, percent, trainData, testData = {},{},{},{},{}
    for i in temp:
        result[i]=labels.values.tolist().count(i)
    for i in result:
        probability[i] = result[i]/len(labels.values.tolist())
        percent[i] = round(trainPercent * result[i])
    size = data.shape[0]
    listSize = int(size/cross)
    list1 = list()
    for j in range(0,cross): 
        if(j < cross - 1):
            x1 = j * listSize
            y1 = (j + 1) * listSize
            testData[j] = data.iloc[x1:y1]
            list1.append(data.iloc[:x1])
            list1.append(data.iloc[y1:])
            trainData[j] = pd.concat(list1)
            list1.clear()
        else:
            x1 = j * listSize
            testData[j] = data.iloc[x1:]
            trainData[j] = data.iloc[:x1]
    labels, trainingData, testingData = list(), list(), list()
    for i in range(0,cross):
        for j in range(0,cross):
            labels = trainData[j].iloc[:,-1:].values.tolist()
            testingData = testData[i].iloc[:,:-1].values.tolist()
            trainingData = trainData[j].values.tolist()
            accs = knn(trainingData, testingData, labels)
            accuracies.append(accs)
    accuracies = [accuracies[i:i + cross] for i in range(0, len(accuracies), cross)]
    for i in range(0,cross):
        for j in range(0,cross):
            print(f"fold = {j+1}: k = {i+1}, Accuracy = {(accuracies[i][j])}")
    for i in range(len(accuracies)):
        total = np.sum(accuracies[i])
        print(f"k = {i+1}, Average Accuracy = {(total/cross)}")

def knn(trainingData, testingData, labels):
    distances = list()
    test = {}
    list1, list2, list3 = [], [], []
    length = len(testingData)
    length2 = len(trainingData)
    for i in range(0, length2):
        for j in range(0, length):
            distance = 0
            x = trainingData[i]
            y = testingData[j]
            for k in range(0,7):
                r = x[k]
                t = y[k]
                distance += np.power(r - t, 2)
        distances.append([x[-1], math.sqrt(distance)])
    distances = sorted(distances, key=lambda a: a[1])
    for i in range(len(distances)):
        list1.append(distances[i][0])
    for i in range(len(labels)):
        list2.append(labels[i][0])
    count = 0
    # temp = set(list1)
    # for i in temp:
    #     list2.append([i, list1.count(i)])
    # list2 = sorted(list2, key=lambda a: a[1])
    # foo = list2[len(list2)-1][0]
    # return accs 
    for x in range(len(list2)):
        if list2[x] == list1[x]:
            count += 1
    return round((count/float(len(list2))) * 100.0, 3)

# greyTime, rgbTime, sNpTime, gaussTime, medianTime = [],[],[],[],[]
# linearTime, histogramTime, histEqualTime, histQauntTime, meanTime = [],[],[],[],[]
# sobelTime, prewittTime, robertsTime, erosionTime, dilationTime = [],[],[],[],[]
# threshTime, kMeansTime, improvedTime, compTime = [],[],[],[]
histSegTime, contTime, featTime, hystTime = [],[],[],[]
totalTime = 0
count = 0

print("*****Starting*****")
f = open(os.path.join(outputPath +'/'+ featuresFileCSV), "w")
f.writelines(["area",str(","),"perimeter",str(","),"mean",str(","),"median",str(","),"meanRGB",str(","),"roundness",str(","),"variance",str(","),"stdDiv",str(","),"label",str("\n")])
f.close()

for filename in os.listdir(imagesPath):
    img = cv2.imread(os.path.join(imagesPath, filename))
    if (img.any() == None):
        print("No Images found in this Folder")
        print("*****Finished*****")
    else:
        global label
        label = re.split('\d+', filename)
        label = label[0]
        count = count + 1
        print("Processing image number: " + str(count))
        
        # grey_start = process_time()
        # greyScaleImage(img, count)
        # grey_end = process_time()
        # greyTime.append(float(grey_end - grey_start))

        # rgb_start = process_time()    
        # imagetoRGB(img, count, rgbWeights)
        # rgb_stop = process_time()
        # rgbTime.append(rgb_stop - rgb_start)
        
        # sNp_start = process_time()
        # saltNpepper(img, count, saltNpepperStrength)
        # sNp_end = process_time()
        # sNpTime.append(float(sNp_end - sNp_start))
        
        # guass_start = process_time()
        # gaussian(img, count, gaussianStdDeviation)
        # gauss_end = process_time()
        # gaussTime.append(float(gauss_end - guass_start))
        
        # med_start = process_time()
        # medianFilter(img, count, medianMaskSize, medianWeights)
        # med_end = process_time()
        # medianTime.append(float(med_end - med_start))
        
        # hist_start = process_time()
        # imageHistogram(img, count)
        # hist_end = process_time()
        # histogramTime.append(float(hist_end - hist_start))
        
        # line_start = process_time()
        # linearFilter(img, count, linearMaskSize, linearWeights)
        # line_end = process_time()
        # linearTime.append(float(line_end - line_start))
        
        # equal_start = process_time()
        # imageHistogramEqualization(img, count)
        # equal_end = process_time()
        # histEqualTime.append(float(equal_end - equal_start))
        
        # quant_start = process_time()
        # imageQuantization(img, count, imageQuantizationLevel)
        # quant_end = process_time()
        # histQauntTime.append(float(quant_end - quant_start))
        
        # mean_start = process_time()
        # meanSquaredQuantizationError(img, count, imageQuantizationLevel)
        # mean_end = process_time()
        # meanTime.append(float(mean_end - mean_start)) 

        # sobel_start = process_time()
        # sobelOperator(img, sobelXdirection, sobelYdirection, count)
        # sobel_end = process_time()
        # sobelTime.append(float(sobel_end - sobel_start)) 

        # improved_start = process_time()
        # improvedSobelOperator(img, improvedSobelXdirection, improvedsobelYdirection, count)
        # improved_end = process_time()
        # improvedTime.append(float(improved_end - improved_start)) 

        # prewitt_start = process_time()
        # prewittOperator(img, prewittXdirection, prewittYdirection, count)
        # prewitt_end = process_time()
        # prewittTime.append(float(prewitt_end - prewitt_start)) 

        # roberts_start = process_time()
        # robertsOperator(img, robertsXdirection, robertsYdirection, count)
        # roberts_end = process_time()
        # robertsTime.append(float(roberts_end - roberts_start)) 

        # erosion_start = process_time()
        # imageErosion(binary, erosionKernel, count)
        # erosion_end = process_time()
        # erosionTime.append(float(erosion_end - erosion_start)) 

        # dilation_start = process_time()
        # imageDilation(binary, dilationKernel, count)
        # dilation_end = process_time()
        # dilationTime.append(float(dilation_end - dilation_start)) 

        # thresh_start = process_time()
        # histogramThreshold(img, count)  
        # thresh_end = process_time()
        # threshTime.append(float(thresh_end - thresh_start)) 

        # kMeans_start = process_time()
        # kmeans(img, count, numClusters)
        # kMeans_end = process_time()
        # kMeansTime.append(float(kMeans_end - kMeans_start))

        # comp_start = process_time()
        # compassOperator(img, count)
        # comp_end = process_time()
        # compTime.append(float(comp_end - comp_start))

        histSeg_start = process_time()
        binaryHistorgram, histSegmentatedImage = histogramSegmentation(img, count)
        histSeg_end = process_time()
        histSegTime.append(float(histSeg_end - histSeg_start)) 

        cont_start = process_time()
        contour = contourImage(binaryHistorgram, count)
        cont_end = process_time()
        contTime.append(float(cont_end - cont_start)) 

        feat_start = process_time()
        featuresCollection(histSegmentatedImage, count, contour, binaryHistorgram, img)
        feat_end = process_time()
        featTime.append(float(feat_end - feat_start)) 

print("*****Image Classification*****")
cross_start = process_time()
crossValidation(outputPath +'/'+ featuresFileCSV)
cross_end = process_time()
crossTime = cross_end - cross_start 

program_end = process_time()
totalTime = (program_end - program_start)
        
# print("*****Images Processed*****")
# print("Total number of images processed: " + str(count))
# print("Average time it took to convert to grey scale images: " + str(averageTime(greyTime)) + " seconds")
# print("Average time it took to convert to RGB images: " + str(averageTime(rgbTime)) + " seconds")
# print("Average time it took to add Salt and Pepper noise: " + str(averageTime(sNpTime)) + " seconds")
# print("Average time it took to add Gaussian noise: " + str(averageTime(gaussTime)) + " seconds")
# print("Average time for Median filter: " + str(averageTime(medianTime)) + " seconds")
# print("Average time to calaculate Histograms: " + str(averageTime(histogramTime)) + " seconds")
# print("Average time for Linear filter: " + str(averageTime(linearTime)) + " seconds")
# print("Average time for Histogram Equalization: " + str(averageTime(histEqualTime)) + " seconds")
# print("Average time for Historgram Quantization: " + str(averageTime(histQauntTime)) + " seconds")
# print("Average time for MSQE: " + str(averageTime(meanTime)) + " seconds")
# print("Average time for Sobel Operator: " + str(averageTime(sobelTime)) + " seconds")
# print("Average time for Improved Sobel Operator: " + str(averageTime(improvedTime)) + " seconds")
# print("Average time for Prewitt Operator: " + str(averageTime(prewittTime)) + " seconds")
# print("Average time for Roberts Operator: " + str(averageTime(robertsTime)) + " seconds")
# print("Average time for Compass Operator: " + str(averageTime(compTime)) + " seconds")
# print("Average time for image Erosion: " + str(averageTime(erosionTime)) + " seconds")
# print("Average time for image Dilation: " + str(averageTime(dilationTime)) + " seconds")
# print("Average time for Histogram Thresholding: " + str(averageTime(threshTime)) + " seconds")
# print("Average time for K Means Segmentation: " + str(averageTime(kMeansTime)) + " seconds")

print("Average time for Histogram Segmentation: " + str(averageTime(histSegTime)) + " seconds")
print("Average time for Contours: " + str(averageTime(contTime)) + " seconds")
print("Average time for Feature Collection: " + str(averageTime(featTime)) + " seconds")
print("Average time for Cross Validation and KNN: " + str(crossTime) + " seconds")
print("Average time it took for the entire program: " + str(totalTime) + " seconds")
print("*****Finished*****")