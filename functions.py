import cv2
import random
import numpy as np
import os

def initiation(POP, X1, Y1, X2, Y2):

    a, b = POP.shape[0], POP.shape[1]
    for i in range(0, a):
        for j in range(0, b):
            if j%2 == 0:
                POP[i][j] = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3))

            if j%2 == 1:
                POP[i][j] = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 3))


    return POP

def initiation_car(POP, X1, Y1, X2, Y2):

    a, b = POP.shape[0], POP.shape[1]
    for i in range(0, a):
        for j in range(0, b):
            if j%2 == 0:
                POP[i][j] = random.randint(int(X1), int(X2))

            if j%2 == 1:
                POP[i][j] = random.randint(int(Y1), int(Y2))


    return POP


def initiation_e(POP, X1, Y1, X2, Y2):

    a, b = POP.shape[0], POP.shape[1]
    for i in range(0, a):
        POP[i][0] = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3))
        POP[i][1] = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 3))
        POP[i][2] = random.randint(0, int((X2-X1)//2))
        POP[i][3] = random.randint(0, int((Y2 - Y1) // 3))


    return POP




def line_pso(img_path, POP, path_adv, K, R, num_g):

    # print('POP = ', POP)

    img = cv2.imread(img_path)
    for k in range(K):
        x1, y1 = int(POP[num_g * k + 0]), int(POP[num_g * k + 1])
        x2, y2 = int(POP[num_g * k + 2]), int(POP[num_g * k + 3])


        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), R)

    cv2.imwrite(path_adv, img)

def line_pso_ablation_C(img_path, POP, path_adv, K, R, num_g, C):

    # print('POP = ', POP)

    img = cv2.imread(img_path)
    for k in range(K):
        x1, y1 = int(POP[num_g * k + 0]), int(POP[num_g * k + 1])
        x2, y2 = int(POP[num_g * k + 2]), int(POP[num_g * k + 3])


        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)

        cv2.line(img, (x1, y1), (x2, y2), C, R)

    cv2.imwrite(path_adv, img)

def ellipse_pso(img_path, POP, path_adv, R):

    img = cv2.imread(img_path)

    x, y = int(POP[0]), int(POP[1])
    lenX, lenY = int(POP[2]), int(POP[3])

    cv2.ellipse(img, (x, y), (lenX, lenY), 0, 0, 360, (0, 0, 0), R)

    cv2.imwrite(path_adv, img)

def ellipse_pso_ablation_C(img_path, POP, path_adv, R, C):

    img = cv2.imread(img_path)

    x, y = int(POP[0]), int(POP[1])
    lenX, lenY = int(POP[2]), int(POP[3])

    cv2.ellipse(img, (x, y), (lenX, lenY), 0, 0, 360, C, R)

    cv2.imwrite(path_adv, img)



def triangle_pso(img_path, POP, path_adv, K, R):

    # print('POP = ', POP)

    pts = []
    img = cv2.imread(img_path)
    if K == 1:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)

    if K == 2:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        x4, y4 = int(POP[6]), int(POP[7])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)

    if K == 3:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        x4, y4 = int(POP[6]), int(POP[7])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        x5, y5 = int(POP[8]), int(POP[9])

        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]], np.int32)

    if K == 4:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        x4, y4 = int(POP[6]), int(POP[7])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        x5, y5 = int(POP[8]), int(POP[9])
        x6, y6 = int(POP[10]), int(POP[11])

        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]], np.int32)

    if K == 5:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        x4, y4 = int(POP[6]), int(POP[7])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        x5, y5 = int(POP[8]), int(POP[9])
        x6, y6 = int(POP[10]), int(POP[11])
        x7, y7 = int(POP[12]), int(POP[13])

        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7]], np.int32)

    if K == 6:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        x4, y4 = int(POP[6]), int(POP[7])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        x5, y5 = int(POP[8]), int(POP[9])
        x6, y6 = int(POP[10]), int(POP[11])
        x7, y7 = int(POP[12]), int(POP[13])
        x8, y8 = int(POP[14]), int(POP[15])

        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8]], np.int32)

    if K == 7:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        x4, y4 = int(POP[6]), int(POP[7])
        x5, y5 = int(POP[8]), int(POP[9])
        x6, y6 = int(POP[10]), int(POP[11])
        x7, y7 = int(POP[12]), int(POP[13])
        x8, y8 = int(POP[14]), int(POP[15])
        x9, y9 = int(POP[16]), int(POP[17])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9]], np.int32)



    cv2.polylines(img, [pts], True, (0, 0, 0), R)

    cv2.imwrite(path_adv, img)


def triangle_pso_ablation_C(img_path, POP, path_adv, K, R, C):

    # print('POP = ', POP)

    pts = []
    img = cv2.imread(img_path)
    if K == 1:
        x1, y1 = int(POP[0]), int(POP[1])
        x2, y2 = int(POP[2]), int(POP[3])
        x3, y3 = int(POP[4]), int(POP[5])
        # print('x1, y1, x2, y2 = ', x1, y1, x2, y2)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)

    cv2.polylines(img, [pts], True, C, R)

    cv2.imwrite(path_adv, img)





def clip_car(population, X1, Y1, X2, Y2):

    a, b = population.shape

    # print('a, b = ', a, b)

    for i in range(0, a):

        for j in range(0, b):

            if j%2 == 0:
                if population[i][j] not in range(int(X1), int(X2)):
                    population[i][j] = random.randint(int(X1), int(X2))
            if j%2 == 1:
                if population[i][j] not in range(int(Y1), int(Y2)):
                    population[i][j] = random.randint(int(Y1), int(Y2))





    return population


def clip(population, X1, Y1, X2, Y2):

    a, b = population.shape

    # print('a, b = ', a, b)

    for i in range(0, a):

        for j in range(0, b):

            if j%2 == 0:
                if population[i][j] not in range(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)):
                    population[i][j] = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3))
            if j%2 == 1:
                if population[i][j] not in range(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)):
                    population[i][j] = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 3))





    return population
