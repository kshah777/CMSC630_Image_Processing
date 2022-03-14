import cv2
import numpy as np
import glob
import itertools
import random
from configparser import ConfigParser 
import os
import os.path as filePath
from time import process_time
from matplotlib import pyplot as plt
from operator import index, le
import copy

program_start = process_time()

config = ConfigParser()
config.read("userConfigurations.ini")
imagesPath = config["config"]["imagesPath"]
outputPath = config["config"]["outputPath"]

rgbWeights = [float(num) for num in config["inputs"]["rgbWeights"].split(",")]

linearMaskSize = config["inputs"]["linearMaskSize"]
linearWeights = [float(num) for num in config["inputs"]["linearWeights"].split(",")]
medianMaskSize = config["inputs"]["medianMaskSize"]
medianWeights = [float(num) for num in config["inputs"]["medianWeights"].split(",")]

gaussianStdDeviation = config["inputs"]["gaussianStdDeviation"]
saltNpepperStrength = config["inputs"]["saltNpepperStrength"]

imageQuantizationLevel = config["inputs"]["imageQuantizationLevel"]

try:
    os.mkdir(outputPath)
    os.mkdir(outputPath + "/grayImages")
    os.mkdir(outputPath + "/colorSpecturm")
    os.mkdir(outputPath + "/medianFilter")
    os.mkdir(outputPath + "/linearFilter")
    os.mkdir(outputPath + "/saltNpeper")
    os.mkdir(outputPath + "/gaussian")
    os.mkdir(outputPath + "/histogram")
    os.mkdir(outputPath + "/histogramEqualizations")
    os.mkdir(outputPath + "/imageQuantization")
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
    elif path == 11:
        cv2.imwrite(filePath.join(outputPath + "/imageQuantization/", "image.jpg".split(".")[0]+str(count)+".jpg"), image)
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

def imageHistogram(img, count):
    height = len(img)
    widht = len(img[0])
    hist = np.zeros(shape=(256), dtype=np.uint64)
    greyImage = greyScaleImage2(img)
    for i in range(height):
        for j in range(widht):
            hist[greyImage[i][j]] = hist[greyImage[i][j]]+1
    plt.plot(hist)
    # plt.show()
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


greyTime = []
rgbTime = []
sNpTime = []
gaussTime = []
medianTime = []
linearTime = []
histogramTime = []
histEqualTime = []
histQauntTime = []
meanTime = []
totalTime = 0
count = 0

print("*****Starting*****")

for image in glob.glob(imagesPath):
    img = cv2.imread(image)
    # img = cv2.imread('test.jpg')
    count = count + 1
    print("Processing image number: " + str(count))
    
    grey_start = process_time()
    greyScaleImage(img, count)
    grey_end = process_time()
    greyTime.append(float(grey_end - grey_start))

    rgb_start = process_time()    
    imagetoRGB(img, count, rgbWeights)
    rgb_stop = process_time()
    rgbTime.append(rgb_stop - rgb_start)
    
    sNp_start = process_time()
    saltNpepper(img, count, saltNpepperStrength)
    sNp_end = process_time()
    sNpTime.append(float(sNp_end - sNp_start))
    
    guass_start = process_time()
    gaussian(img, count, gaussianStdDeviation)
    gauss_end = process_time()
    gaussTime.append(float(gauss_end - guass_start))
    
    med_start = process_time()
    medianFilter(img, count, medianMaskSize, medianWeights)
    med_end = process_time()
    medianTime.append(float(med_end - med_start))
    
    hist_start = process_time()
    imageHistogram(img, count)
    hist_end = process_time()
    histogramTime.append(float(hist_end - hist_start))
    
    line_start = process_time()
    linearFilter(img, count, linearMaskSize, linearWeights)
    line_end = process_time()
    linearTime.append(float(line_end - line_start))
    
    equal_start = process_time()
    imageHistogramEqualization(img, count)
    equal_end = process_time()
    histEqualTime.append(float(equal_end - equal_start))
    
    quant_start = process_time()
    imageQuantization(img, count, imageQuantizationLevel)
    quant_end = process_time()
    histQauntTime.append(float(quant_end - quant_start))
    
    mean_start = process_time()
    meanSquaredQuantizationError(img, imageQuantizationLevel)
    mean_end = process_time()
    meanTime.append(float(mean_start - mean_end)) 

print("*****Images Processed*****")
    
print("Total number of images processed: " + str(count))
print("Average time it took to convert to grey scale images: " + str(averageTime(greyTime)) + " seconds")
print("Average time it took to convert to RGB images: " + str(averageTime(rgbTime)) + " seconds")
print("Average time it took to add Salt and Pepper noise: " + str(averageTime(sNpTime)) + " seconds")
print("Average time it took to add Gaussian noise: " + str(averageTime(gaussTime)) + " seconds")
print("Average time for Median filter: " + str(averageTime(medianTime)) + " seconds")
print("Average time to calaculate Histograms: " + str(averageTime(histogramTime)) + " seconds")
print("Average time for Linear filter: " + str(averageTime(linearTime)) + " seconds")
print("Average time for Histogram Equalization: " + str(averageTime(histEqualTime)) + " seconds")
print("Average time for Historgram Quantization: " + str(averageTime(histQauntTime)) + " seconds")
print("Average time for MSQE: " + str(averageTime(meanTime)) + " seconds")

program_end = process_time()
totalTime = (program_end - program_start)
print("Average time it took for the entire program: " + str(totalTime) + " seconds")

print("*****Finished*****")