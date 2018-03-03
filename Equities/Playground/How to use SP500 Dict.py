import Finance.STOCKS.SP500 as SP
import datetime
import pandas as pd

start = datetime.date(2012,1,1)
end = datetime.date(2016,1,1)
df = SP.close(start,end)

dict = pd.Series.to_dict(df)
# dict['ZBH'].plot() #picks a specific stock