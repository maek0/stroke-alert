from showVidFcns import strokedet
from showVidFcns import strokedetII
from showVidFcns import setbase
from showVidFcns import handCalc
# from strokedet_allfun import strokedet
# from strokedet_allfun import strokedetII
# from strokedet_allfun import setbase
# from strokedet_allfun import handCalc

# setbase()

r = strokedet()
if r < 0.65:
    print("--------------------")
    print("\nRaise your Hands!\n")
    print("--------------------")
    ruling = strokedetII(r)
else:
    ruling = 0

if ruling == 0:
    opr = 100*(1-r)
    print("\n--------------------\n")
    print("No Stroke Detected\n")
    if r < 0.65:
        print("There is some uncertainty in the probability. For more accuracy, move to a place with better lighting.\nIf you feel dizzy or numbness in one arm or one side of your face, call EMS.")
    print("\nThere is a %.2f percent chance you are experiencing a stroke\n" % opr)
    print("--------------------")
else:
    opr = 100*(1-r)
    print("--------------------\n")
    print("Stroke Likely: call EMS immediately!")
    print("\nThere is a %.2f percent chance you are experiencing a stroke\n" % opr)
    print("--------------------")

# print(ruling)
# print(r)
