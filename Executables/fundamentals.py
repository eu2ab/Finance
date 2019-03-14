from GetData import balance_sheet, income_statement, ratios_download

# Explaining the script and pulling in tickers to analyze and build balance sheets
print("Script will save company balance sheet information for the last 5 quarters as a .csv "
      "file on your desktop. You can run it for as many securities as you would like.")
print(" ")
input_list = input('Please enter tickers, separated by one space only (e.g., AAPL CSCO): ').split()
tickers = [x for x in input_list]

for x in tickers:
    name = str(x)
    df1 = balance_sheet(name, 'q')  # Calling balance sheet metrics
    df2 = income_statement(name, 'q')  # Calling income statement metrics
    df3 = ratios_download(name)  # Calling financial ratios
    df = df1.append(df2)  # Combining balance sheet and income, vertically appended
    df = df.append(df3)  # Adding financial ratios

    # Add columns for growth changes
    for i in range(0, len(df1.columns) - 1):
        colname = str(df1.columns[i]) + "/" + str(df1.columns[i + 1])
        df[colname] = (df1.iloc[:, i] / df1.iloc[:, i + 1]) - 1

    # Write to final csv file
    df.to_csv('%s.csv' % name)