from GetData import balance_sheet, income_statement, ratios_download

# Explaining the script and pulling in tickers to analyze and build balance sheets
print("Script will save company balance sheet infor for the last 5 quarters as a .csv "
      "file on your desktop. You can run it for as many securities as you would like.")
print(" ")
input_list = input('Please enter tickers, separated by one space only (e.g., AAPL CSCO): ').split()
tickers = [x for x in input_list]

for x in tickers:
    name = str(x)
    df1 = balance_sheet(name, 'q')  # calling balance sheet metrics
    df2 = income_statement(name, 'q')  # calling income statement metrics
    df3 = ratios_download(name)  # calling financial ratios
    df = df1.append(df2)  # combining balance sheet and income, vertically appended
    df = df.append(df3)  # adding financial ratios

    # add columns for growth changes
    for i in range(0, len(df1.columns) - 1):
        colname = str(df1.columns[i]) + "/" + str(df1.columns[i + 1])
        df[colname] = (df1.iloc[:, i] / df1.iloc[:, i + 1]) - 1

    # write to final csv file
    df.to_csv('%s.csv' % name)


# TO DO LIST
    # save all to desktop
    # build out valuation ratios to the bottom (e.g., EV/EBITDA)


# aapl.loc['Total cash'].plot() # graph total cash over the quarters
# aapl.shape to understand dimensions of pandas dataframe

# aapl.iloc[:,0] first data column (2017 - 03)
# aapl.iloc[:,1] second data column (2016 - 12)
# appl.iloc[:,2] third data column (2016 - 09)
# appl.iloc[:,3] fourth data column (2016 - 06)
# appl.iloc[:,4] fifth data column (2016 - 03)

# aapl.columns[0] name of first data column (2017 - 03)
# ...

# # adding manual metrics
# cols = df.columns  # bring in column names from inital data call
# index = ['EV/EBITDA', 'Revenue/EBITDA', 'Net Profit Margin', 'Total Asset Turnover', 'Return on Assets', 'Leverage', 'Return on Equity']  # list of added metrics
# df = pd.DataFrame(index = index, columns=cols)  # make empty data frame to be filled in
#
# for i in range(0,len(df.columns) - 1):
#     df.iloc[0,i] = (df.loc['Common stock', df.columns[i]] + df.loc['Long-term debt', df.columns[i]] - df.loc['Cash and cash equivalents', df.columns[i]]) / (df.loc['EBITDA', df.columns[i]])   # first row, EV/EBITDA, across columns, i
#     df.iloc[1,i] = (df.loc['Revenue', df.columns[i]]) / (df.loc['EBITDA', df.columns[i]])  # Revenue/EBITDA
#     df.iloc[2,i] = (df.loc['Net income', df.columns[i]]) / (df.loc['Revenue', df.columns[i]])  # Net income/Revenue = Profit Margin
#     df.iloc[3,i] = (df.loc['Revenue', df.columns[i]]) / (df.loc['Total assets', df.columns[i]])  # Revenue/Total Assets = Asset Turnover
#     df.iloc[4,i] = (df.loc['Net income', df.columns[i]]) / (df.loc['Total assets', df.columns[i]])  # ROA
#     df.iloc[5,i] = (df.loc['Total assets', df.columns[i]]) / (df.loc['Total stockholders" equity', df.columns[i]])  # Leverage *** MIGHT BE AN ISSUE
#     df.iloc[6,i] = (df.loc['Net income', df.columns[i]]) / (df.loc['Total stockholders" equity', df.columns[i]])  # ROE
