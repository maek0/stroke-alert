# call strokedet --> record 5s of video and get ruling based on baseline or past imgs

import cv2
import mediapipe as mp
import numpy as np
import time
from shapely.geometry import Polygon
import math
from scipy import stats
from strokedet_allfun import strokedet
from strokedet_allfun import setbase
from strokedet_allfun import handCalc

# setbase()
# result, r = strokedet()
# print(result)
# print(r)
# result = 0 --> likely no stroke
# result = 1 --> likely stroke
# r is the average of p-values of determining characteristics

hand1, hand2, wrist, relDiff = handCalc(5)

print(wrist)
print(hand1)
print(hand2)
print(relDiff)
