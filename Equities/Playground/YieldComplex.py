from GetData import getFRED
import datetime

"""
Go to https://fred.stlouisfed.org/categories/22 to look at other interest rate products/spreads/etc.
"""

# US Treasuries
start = datetime.date(2010, 1, 1)  # start date of analysis
end = datetime.date.today()
UST = ["DGS30", "DGS20", "DGS10", "DGS7", "DGS5", "DGS3", "DGS2", "DGS1", "DGS6MO", "DGS3MO", "DGS1MO"]
df = getFRED(UST, start, end)
df.plot()
