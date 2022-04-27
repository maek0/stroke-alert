# from showVidFcns import strokedet
# from showVidFcns import strokedetII
# from showVidFcns import setbase
# from showVidFcns import handCalc
from strokedet_allfun import strokedet
from strokedet_allfun import strokedetII
from strokedet_allfun import setbase
from strokedet_allfun import handCalc

# setbase()

# result = 0 --> likely no stroke
# result = 1 --> likely stroke

# r is the average of p-values of determining characteristics

r = strokedet()
if r < 0.5:
    print("--------------------")
    print("\nRaise your Hands!\n")
    print("--------------------")
    ruling = strokedetII(r)
else:
    ruling = 0

if ruling == 0:
    opr = 100*(1-r)
    print("--------------------\n")
    print("No Stroke Detected\n")
    if r < 0.5:
        print("There is some uncertainty in the probability. For more accuracy, move to a place with better lighting.")
    print("There is a %.2f percent chance you are experiencing a stroke\n" % opr)
    print("--------------------")
else:
    opr = 100*(1-r)
    print("--------------------\n")
    print("Stroke Likely: call EMS\n")
    print("\nThere is a %.2f percent chance you are experiencing a stroke" % opr)
    print("--------------------")

# print(ruling)
# print(r)
