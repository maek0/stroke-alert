from showVidFcns import strokedet
from showVidFcns import strokedetII
from showVidFcns import setbase
from showVidFcns import handCalc

# setbase()

# result = 0 --> likely no stroke
# result = 1 --> likely stroke

# r is the average of p-values of determining characteristics

r = strokedet()
if r < 0.5:
    print("\nRaise your Hands!\n\n")
    ruling = strokedetII(r)
else:
    ruling = 0

if ruling == 0:
    opr = 100*(1-r)
    print("\n\nNo Stroke Detected\n")
    print("There is a %.2f percent chance you are experiencing a stroke\n" % opr)
else:
    opr = 100*(1-r)
    print("\n\nThere is a %.2f percent chance you are experiencing a stroke" % opr)
    print("\nStroke Likely: call EMS\n")

# print(ruling)
# print(r)
