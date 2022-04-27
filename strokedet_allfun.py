import cv2
import mediapipe as mp
import numpy as np
import time
from shapely.geometry import Polygon
import math
import os, os.path
from scipy import stats
import glob
from PIL import Image
import platform

def facepose(sec):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    storepts = []
    
    FaceX = []
    FaceY = []
    HandX = []
    HandY = []

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    endtime = time.time() + sec

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.7)
    hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fm_result = face_mesh.process(image)
        hands_result = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if fm_result.multi_face_landmarks:
            for face_landmarks in fm_result.multi_face_landmarks:
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_tesselation_style())

                for id,lm in enumerate(face_landmarks.landmark):
                    x = lm.x
                    y = lm.y

                    shape = image.shape 
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    FaceX.append(relative_x)
                    FaceY.append(relative_y)
                    storepts.append([relative_x, relative_y])
        
        # if hands_result.multi_hand_landmarks:
        #     for hand_landmarks in hands_result.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())
                
                # for id,lm in enumerate(hand_landmarks.landmark):
                #     x = lm.x
                #     y = lm.y

                #     shape = image.shape 
                #     relative_x = int(x * shape[1])
                #     relative_y = int(y * shape[0])

                #     HandX.append(relative_x)
                #     HandY.append(relative_y)

                    # handPts.append([relative_x, relative_y])

        # cv2.imshow('MediaPipe Face Mesh',cv2.flip(image, 1))

        t = time.time()

        if time.time() > endtime:
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    return FaceX, FaceY

def calcVars(sec):
    
    FaceX, FaceY = facepose(sec)

    FaceXarray = np.array(FaceX[0:477])
    del FaceX[0:477]
    FaceYarray = np.array(FaceY[0:477])
    del FaceY[0:477]

    while len(FaceX) >= 478:
        FaceXarray = np.vstack((FaceXarray, FaceX[0:477]))
        del FaceX[0:477]
    while len(FaceY) >= 478:
        FaceYarray = np.vstack((FaceYarray, FaceY[0:477]))
        del FaceY[0:477]

    midInd = [10,9,8,168,6,197,195,5,4,1,19,94,2,164,0,17,18,200,199]
    rmouthInd = [76,184,40,39,37,0,17,84,181,91,146]
    lmouthInd = [0,267,269,270,408,306,307,321,405,314,17]
    leInd = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]
    reInd = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]

    midXshift = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    lmouthA = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    rmouthA = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    leArea = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    leDiff = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    reArea = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    reDiff = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    rmouthPt = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    lmouthPt = np.empty((FaceXarray.shape[0],0),dtype=np.float64)

    allMidShift = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    allLMouthA = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    allRMouthA = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    MouthRatioArea = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    allLEArea = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    allREArea = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    allLEDiff = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    allREDiff = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    EyeRatioArea = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    EyeRatioHeight = np.empty((FaceXarray.shape[0],0),dtype=np.float64)
    MouthCorners = np.empty((FaceXarray.shape[0],0),dtype=np.float64)

    for i in range(FaceXarray.shape[0]):
        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        g = []
        h = []
        j = []
        for id in range(len(midInd)):
            a.append(FaceXarray[i,midInd[id]])
        midXshift = np.average(np.abs(np.diff(a)))
        allMidShift = np.append(allMidShift, midXshift)

        for id in range(len(reInd)):
            b.append(FaceXarray[i,leInd[id]])
            c.append(FaceYarray[i,leInd[id]])
            d.append(FaceXarray[i,reInd[id]])
            e.append(FaceYarray[i,reInd[id]])
        leArea = Polygon(zip(b,c))
        allLEArea = np.append(allLEArea, leArea.area)
        reArea = Polygon(zip(d,e))
        allREArea = np.append(allREArea, reArea.area)
        if reArea.area > leArea.area:
            EyeRatioArea = np.append(EyeRatioArea, leArea.area/reArea.area)
        else:
            EyeRatioArea = np.append(EyeRatioArea, reArea.area/leArea.area)

        for id in range(len(lmouthInd)):
            f.append(FaceXarray[i,lmouthInd[id]])
            g.append(FaceYarray[i,lmouthInd[id]])
            h.append(FaceXarray[i,rmouthInd[id]])
            j.append(FaceYarray[i,rmouthInd[id]])
        lmouthA = Polygon(zip(f,g))
        allLMouthA = np.append(allLMouthA, lmouthA.area)
        rmouthA = Polygon(zip(h,j))
        allRMouthA = np.append(allRMouthA, rmouthA.area)
        if rmouthA.area > lmouthA.area:
            MouthRatioArea = np.append(MouthRatioArea, lmouthA.area/rmouthA.area)
        else:
            MouthRatioArea = np.append(MouthRatioArea, rmouthA.area/lmouthA.area)

        leDiff = math.dist([FaceXarray[i,386],FaceYarray[i,386]],[FaceXarray[i,374],FaceYarray[i,374]])
        allLEDiff = np.append(allLEDiff, leDiff)
        
        reDiff = math.dist([FaceXarray[i,159],FaceYarray[i,159]],[FaceXarray[i,145],FaceYarray[i,145]])
        allREDiff = np.append(allREDiff, reDiff)

        if reDiff > leDiff:
            EyeRatioHeight = np.append(EyeRatioHeight, leDiff/reDiff)
        else:
            EyeRatioHeight = np.append(EyeRatioHeight, reDiff/leDiff)
        
        rmouthPt = math.dist([FaceXarray[i,76],FaceYarray[i,76]],[FaceXarray[i,2],FaceYarray[i,2]])
        lmouthPt = math.dist([FaceXarray[i,306],FaceYarray[i,306]],[FaceXarray[i,2],FaceYarray[i,2]])

        if rmouthPt > lmouthPt:
            MouthCorners = np.append(MouthCorners, lmouthPt/rmouthPt)
        else:
            MouthCorners = np.append(MouthCorners, rmouthPt/lmouthPt)

    saveArray = np.array([allMidShift, allLEArea, allREArea, EyeRatioArea, allLMouthA, allRMouthA, MouthRatioArea, allLEDiff, allREDiff, EyeRatioHeight, MouthCorners])
    return saveArray

def setbase():
    baseArray = calcVars(10)
    savebase(baseArray)

def savebase(baseArray):
    dir_path = os.getcwd()
    opsys = platform.system()
    if opsys == 'Windows':
        arrpath = "%s\SavedArrays" % (dir_path)     # Windows
    else:
        arrpath = "%s/SavedArrays" % (dir_path)     # MAC
    
    isExist = os.path.exists(arrpath)
    if not isExist:
        os.makedirs(arrpath)

    timestr = time.strftime("%m-%d-%Y_%H-%M-%S")
    filename = "base_%s.npy" % (timestr)

    np.save(os.path.join(arrpath, filename), baseArray)

def handdet(sec):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    storepts = []

    Hand1X = []
    Hand1Y = []
    Hand2X = []
    Hand2Y = []

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    endtime = time.time() + sec

    hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_result = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if hands_result.multi_hand_landmarks:
            # for hand_landmarks in hands_result.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(
            #         image,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style())

            for hand_no, hand_lm in enumerate(hands_result.multi_hand_landmarks):
                if hand_no == 0:
                    x = hand_lm.landmark[mp_hands.HandLandmark(0).value].x
                    y = hand_lm.landmark[mp_hands.HandLandmark(0).value].y

                    shape = image.shape 
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    Hand1X.append(relative_x)
                    Hand1Y.append(relative_y)

                elif hand_no == 1:
                    x = hand_lm.landmark[mp_hands.HandLandmark(1).value].x
                    y = hand_lm.landmark[mp_hands.HandLandmark(1).value].y

                    shape = image.shape 
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    Hand2X.append(relative_x)
                    Hand2Y.append(relative_y)

                else:
                    Hand1X = []
                    Hand1Y = []
                    Hand2X = []
                    Hand2Y = []

        # cv2.imshow('MediaPipe Hand Points',cv2.flip(image, 1))

        t = time.time()

        if time.time() > endtime:
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    return Hand1X, Hand1Y, Hand2X, Hand2Y

def handCalc(sec):

    Hand1X, Hand1Y, Hand2X, Hand2Y = handdet(sec)

    Hand1Xarray = np.array(Hand1X[0:20])
    del Hand1X[0:20]
    Hand1Yarray = np.array(Hand1Y[0:20])
    del Hand1Y[0:20]
    Hand2Xarray = np.array(Hand2X[0:20])
    del Hand2X[0:20]
    Hand2Yarray = np.array(Hand2Y[0:20])
    del Hand2Y[0:20]

    while len(Hand1X) >= 21:
        Hand1Xarray = np.vstack((Hand1Xarray, Hand1X[0:20]))
        del Hand1X[0:20]
    while len(Hand1Y) >= 21:
        Hand1Yarray = np.vstack((Hand1Yarray, Hand1Y[0:20]))
        del Hand1Y[0:20]
    while len(Hand2X) >= 21:
        Hand2Xarray = np.vstack((Hand2Xarray, Hand2X[0:20]))
        del Hand2X[0:20]
    while len(Hand2Y) >= 21:
        Hand2Yarray = np.vstack((Hand2Yarray, Hand2Y[0:20]))
        del Hand2Y[0:20]
    
    wristDiff = np.empty((Hand1Yarray.shape[0],0),dtype=np.float64)

    if Hand1Yarray.shape[0] == 0 & Hand2Yarray.shape[0] == 0:
        pass
    else:
        if Hand1Yarray.shape[0] > Hand2Yarray.shape[0]:
            sz = Hand2Yarray.shape[0]
        elif Hand1Yarray.shape[0] < Hand2Yarray.shape[0]:
            sz = Hand1Yarray.shape[0]
        else:
            sz = Hand1Yarray.shape[0]

        eqH1 = Hand1Yarray[0:sz]
        eqH2 = Hand2Yarray[0:sz]
        relDiff = [a/b for a,b in zip(eqH1,eqH2)]
        wristDiff = np.abs(eqH1-eqH2)
        
    return wristDiff, relDiff

def strokedet():
    past = []
    dir_path = os.getcwd()
    arrpath = "%s/SavedArrays" % (dir_path)
    framefind = "%s/framevals_*" % (arrpath)
    basefind = "%s/base_*" % (arrpath)
    isExist = os.path.exists(arrpath)

    if not isExist:
        raise ValueError("Need to establish baseline images before running!")
        # Warning("Need to establish baseline images before running!")
    else:
        if glob.glob(framefind):
            for file in glob.glob(framefind):
                temp = np.load(file)
                if past == []:
                    past = temp
                else:
                    past = np.hstack((past, temp))
        elif glob.glob(basefind):
            for file in glob.glob(basefind):
                temp = np.load(file)
                if past == []:
                    past = temp
                else:
                    past = np.hstack((past, temp))
        else:
            raise ValueError("Need to establish baseline images before running!")
            # Warning("Need to establish baseline images before running!")

    # array order:
    # [[allMidShift], 
    # [allLEArea], 
    # [allREArea], 
    # [EyeRatioArea], 
    # [allLMouthA], 
    # [allRMouthA], 
    # [MouthRatioArea], 
    # [allLEDiff], 
    # [allREDiff], 
    # [EyeRatioHeight], 
    # [MouthCorners]]

    array = calcVars(5)
    _, p1 = stats.ttest_ind(past[3],array[3])       # EyeRatioArea p-value
    _, p3 = stats.ttest_ind(past[9],array[9])       # EyeRatioHeight p-value
    _, p4 = stats.ttest_ind(past[10],array[10])     # MouthCorners p-value

    ps = np.array([p1,p3,p4])

    r = np.mean(ps)

    if r >= 0.6:
        dir_path = os.getcwd()
        opsys = platform.system()
        
        if opsys == 'Windows':
            arrpath = "%s\SavedArrays" % (dir_path)
        else:
            arrpath = "%s/SavedArrays" % (dir_path)

        isExist = os.path.exists(arrpath)
        if not isExist:
            os.makedirs(arrpath)
        
        timestr = time.strftime("%m-%d-%Y_%H-%M-%S")
        filename = "framevals_%s.npy" % (timestr)

        np.save(os.path.join(arrpath, filename), array)

    return r

def strokedetII(r):
    wristDiff, rDiff = handCalc(5)
    rDiff = np.mean(rDiff[3:])
    
    if rDiff >= 2:
        ruling = 1
    elif rDiff <= 0.5:
        ruling = 1
    else:
        ruling = 0
    
    return ruling