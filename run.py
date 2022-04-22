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

ruling, r = strokedet()
