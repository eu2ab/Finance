from urllib.request import urlopen
import pandas as pd
import pandas_datareader.data as web
import quandl


def getFRED(code, start, end):
    """
    :param code: Ticker or FRED code of parameter to be considered
    :param start: datetime.date(yyyy, m, d)
    :param end:  datetime.date(yyyy, m, d)
    :return: pandas time series
    """
    df = web.DataReader(code, 'fred', start, end)
    return df


def getGOLD(start, end):
    """
    Returns the price of Gold at London fixing
    :param start: datetime.date(yyyy, m, d)
    :param end: datetime.date(yyyy, m, d)
    :return: pandas time series
    """
    df = web.DataReader('GOLDAMGBD228NLBM', 'fred', start, end)
    return df


def getWTI(start, end):
    """
    Returns price of WTI Crude
    :param start: datetime.date(yyyy, m, d)
    :param end: datetime.date(yyyy, m, d)
    :return: pandas time series
    """
    df = web.DataReader('DCOILWTICO', 'fred', start, end)
    return df


def SP500():
    """
    :return: Returns a list of stock tickers of constituents of S&P 500
    """
    url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    sec = pd.read_csv(url)

    symbols = sec['Symbol']  # list of just symbols
    symbols = pd.Series.tolist(symbols)
    symbols = [x for x in symbols if not '.' in x] # removes symbols with period (e.g., BRK.B)
    return symbols


def QUANDL_prices(tickers, start, end):
    """
    :param tickers: list of tickers
    :param start: datetime.date(yyyy, m, d)
    :param end: datetime.date(yyyy, m, d)
    :return: dataframe with columns of different stock tickers, indexed between start and end date
    """
    # converting tickers to "code"
    i = 0
    code = []
    if len(tickers) == 1:
        code = ['WIKI/' + str(tickers[0]) + '.11']
    else:
        code = ['WIKI/' + str(tickers[0]) + '.11']
        for i in range(1, len(tickers)):
            add = 'WIKI/' + str(tickers[i]) + '.11'
            code.append(add)

    df = quandl.get(code, start_date=start, end_date=end, authtoken="63rHV5FhAYL36rmWhjP7")
    # quandl.get(['WIKI/AAPL.11']) gets only the closing prices for apple
    # quandl.get(['WIKI/AAPL.11','WIKI/MSFT.11']) gets only the closing prices for apple and microsoft
    df.columns = tickers
    return df


def QUANDL_price_pct_change(tickers, start, end):
    """
    :param tickers: list of tickers
    :param start: datetime.date(yyyy, m, d)
    :param end: datetime.date(yyyy, m, d)
    :return: dataframe with columns of different stock tickers, indexed between start and end date, w/ pct change
    """
    # converting tickers to "code"
    i = 0
    code = []
    if len(tickers) == 1:
        code = ['WIKI/' + str(tickers[0]) + '.11']
    else:
        code = ['WIKI/' + str(tickers[0]) + '.11']
        for i in range(1, len(tickers)):
            add = 'WIKI/' + str(tickers[i]) + '.11'
            code.append(add)

    df = quandl.get(code, start_date=start, end_date=end, authtoken="63rHV5FhAYL36rmWhjP7")
    # quandl.get(['WIKI/AAPL.11']) gets only the closing prices for apple
    # quandl.get(['WIKI/AAPL.11','WIKI/MSFT.11']) gets only the closing prices for apple and microsoft
    df.columns = tickers
    df = df.pct_change()
    return df


# def Fundamentals(tickers):
#     for x in tickers:
#         name = str(x)
#         df1 = balance_sheet(name, 'q')  # calling balance sheet metrics
#         df2 = income_statement(name, 'q')  # calling income statement metrics
#         df3 = ratios_download(name)  # calling financial ratios
#         df = df1.append(df2)  # combining balance sheet and income, vertically appended
#         df = df.append(df3)  # adding financial ratios
#         #
#         # # add columns for growth changes
#         # for i in range(0, len(df1.columns) - 1):
#         #     colname = str(df1.columns[i]) + "/" + str(df1.columns[i + 1])
#         #     df[colname] = (df1.iloc[:, i] / df1.iloc[:, i + 1]) - 1
#     return df


def ShortInterest(tickers, start, end):
    """
    Short interest for basket of stocks. Not as comprehensive, thus would advise against doing
    the whole S&P 500
    :param tickers: e.g., 'AAPL'
    :param start: datetime.date(yyyy, m, d)
    :param end: datetime.date(yyyy, m, d)
    :return: short interest, average volume, and Days to Cover for chosen tickers
    """
    # converting tickers to "code"
    i = 0
    code = []
    if len(tickers) == 1:
        code = ['SI/' + str(tickers[0]) + '_SI']
    else:
        code = ['SI/' + str(tickers[0]) + '_SI']
        for i in range(1, len(tickers)):
            add = 'SI/' + str(tickers[i]) + '_SI'
            code.append(add)
    df = quandl.get(code, start_date=start, end_date=end, authtoken="63rHV5FhAYL36rmWhjP7")
    return df


def QUANDL_fundCondensed(tickers):
    """
    :param tickers: list of securities to consider (e.g., ['AAPL','MSFT'])
    :return: dataframe of fundamental data for tickers considered
    """
    df = quandl.get_table('ZACKS/FC', ticker = tickers, api_key="63rHV5FhAYL36rmWhjP7")
    return df


def QUANDL_fundRatios(tickers):
    """
    :param tickers: list of securities to consider (e.g., ['AAPL','MSFT'])
    :return: dataframe of fundamental ratios for tickers considered
    """
    df = quandl.get_table('ZACKS/FC', ticker = tickers, api_key="63rHV5FhAYL36rmWhjP7")
    return df


def getFX(code, start, end):
    """
    Returns the FX pairs for trade weighted dollar, EUR/USD, GBP/USD, USD/CNH, USD/JPY, USD/CAD, USD/MXN, USD/BRL
    USD/KRW, AUD/USD, USD/INR
    :param code: currency pair
    :param start: datetime.date(yyyy, m, d)
    :param end: datetime.date(yyyy, m, d)
    :return: pandas
    """
    if code == 'DXY':
        code = 'DTWEXB'
    elif code == 'EUR/USD':
        code = 'DEXUSEU'
    elif code == 'GBP/USD':
        code = 'DEXUSUK'
    elif code == 'USD/CNH':
        code = 'DEXCHUS'
    elif code == 'USD/JPY':
        code = 'DEXJPUS'
    elif code == 'USD/CAD':
        code = 'DEXCAUS'
    elif code == 'USD/MXN':
        code = 'DEXMXUS'
    elif code == 'USD/BRL':
        code = 'DEXBZUS'
    elif code == 'USD/KRW':
        code = 'DEXKOUS'
    elif code == 'AUD/USD':
        code = 'DEXUSAL'
    elif code == 'USD/INR':
        code = 'DEXINUS'
    df = web.DataReader(code, 'fred', start, end)
    return df


def financials_download(ticker, report, frequency):
    """
    Ignore...for use with other financial formulae
    """
    if frequency == "A" or frequency == "a":
        frequency = "12"
    elif frequency == "Q" or frequency == "q":
        frequency = "3"
    url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?&t='+ticker+'&region=usa&culture=en-US&cur=USD&reportType='+report+'&period='+frequency+'&dataType=R&order=desc&columnYear=5&rounding=3&view=raw&r=640081&denominatorView=raw&number=3'
    df = pd.read_csv(url, skiprows=1, index_col=0)
    return df


def balance_sheet(ticker, frequency):
    """
    Balance sheet of basket of stocks
    :param ticker: e.g., 'AAPL' or multiple securities
    :param frequency: 'Q' or 'A'
    :return:
    """
    if frequency == "A" or frequency == "a":
        frequency = "12"
    elif frequency == "Q" or frequency == "q":
        frequency = "3"
    url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?&t='+ticker+'&region=usa&culture=en-US&cur=USD&reportType=bs&period='+frequency+'&dataType=R&order=desc&columnYear=5&rounding=3&view=raw&r=640081&denominatorView=raw&number=3'
    df = pd.read_csv(url, skiprows=1, index_col=0)
    return df


def income_statement(ticker, frequency):
    """
    Income statement of basket of stocks
    :param ticker: e.g., 'AAPL' or multiple securities
    :param frequency: 'Q' or 'A'
    :return:
    """
    if frequency == "A" or frequency == "a":
        frequency = "12"
    elif frequency == "Q" or frequency == "q":
        frequency = "3"
    url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?&t='+ticker+'&region=usa&culture=en-US&cur=USD&reportType=is&period='+frequency+'&dataType=R&order=desc&columnYear=5&rounding=3&view=raw&r=640081&denominatorView=raw&number=3'
    df = pd.read_csv(url, skiprows=1, index_col=0)
    return df


def ratios_download(ticker):
    """
    :param ticker: e.g., 'AAPL'
    :return: pd dataframe of all available ratios for security, from MorningStar
    """
    url = 'http://financials.morningstar.com/ajax/exportKR2CSV.html?&callback=?&t='+ticker+'&region=usa&culture=en-US&cur=USD&order=desc'
    df = pd.read_csv(url, skiprows=2, index_col=0)
    # df.to_csv('test.csv')
    return df


def y_stats(ticker, stat):
    """
    :param ticker: ticker of security
    :param stat: code for the stat to be chosen
    :return:
    """
    url = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (ticker, stat)
    return urlopen(url).read().decode('UTF-8').strip()


def cash_and_equivalents(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: cash and equivalents off of balance sheet
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Cash and cash equivalents']


def short_term_investments(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Short-term investments']


def total_cash(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Total cash']


def receivables(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Receivables']


def inventories(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Inventories']


def prepaid_expenses(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Prepaid expenses']


def other_current_assets(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Other current assets']


def gross_ppe(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Gross property, plant and equipment']


def accumulated_depreciation(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Accumulated Depreciation']


def net_ppe(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Net property, plant and equipment']


def equity_and_other_investments(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Equity and other investments']


def goodwill(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Goodwill']


def intangible_assets(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Intangible assets']


def other_longterm_assets(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Other long-term assets']


def total_noncurrent_assets(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Total non-current assets']


def total_assets(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Total assets']


def accounts_payable(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Accounts payable']


def shortterm_debt(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Short-term debt']


def accrued_liabilities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Accrued liabilities']


def other_current_liabilities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Other current liabilities']


def total_current_liabilities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Total current liabilities']


def longterm_debt(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Long-term debt']


def deferred_taxes_liabilities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Deferred taxes liabilities']


def pensions_postretirement_benefits(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Pensions and other post retirement benefits']


def minority_interest(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Minority interest']


def total_noncurrent_liabilities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Total non-current liabilities']


def total_liabilities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Total liabilities']


def common_stock(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Common stock']


def additional_paidin_capital(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Additional_paidin_capital']


def retained_earnings(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Retained earnings']


def treasury_stock(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Treasury stock']


def accumulated_other_comprehensive_income(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc['Accumulated other comprehensive income']


def total_stockholder_equity(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc["Total Stockholders' equity"]


def total_liabilities_and_stockholders_equity(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'bs', frequency)
    return df.loc["Total liabilities and stockholders' equity"]


def revenue_5yr(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Revenue"]


def cost_of_revenue(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Cost of revenue"]


def gross_profit(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Gross profit"]


def sales_administrative_expense(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Sales, General and administrative"]


def depreciation_amort_expense(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Depreciation and amortization"]


def interest_expense(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Interest expense"]


def other_operating_expenses(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Other operating expenses"]


def total_costs_and_expenses(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Total costs and expenses"]


def income_tax_expense(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Provision for income taxes"]


def net_income_contin_ops(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Net income from continuing operations"]


def net_income(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Net income"]


def preferred_dividend(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Preferred dividend"]


def net_income_to_common_shareholders(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["Net income available to common shareholders"]


def eps_basic(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    df = df[0:len(df)-3]
    return df.loc["Basic"]


def eps_diluted(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    df = df[0:len(df)-3]
    return df.loc["Diluted"]


def ebitda(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'is', frequency)
    return df.loc["EBITDA"]


def net_cash_from_ops(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Net cash provided by operating activities"]


def investments_in_ppe(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Investments in property, plant, and equipment"]


def net_cash_used_for_investing_activities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Net cash used for investing activities"]


def debt_issued(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Debt issued"]


def debt_repayment(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Debt repayment"]


def net_cash_from_financing_activities(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Net cash provided by (used for) financing activities"]


def net_change_in_cash(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Net change in cash"]


def cash_at_beginning_period(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Cash at beginning of period"]


def cash_at_end_period(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Cash at end of period"]


def operating_cf(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Operating cash flow"]


def capex(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Capital expenditure"]


def fcf(ticker, frequency):
    """
    :param ticker: e.g., 'AAPL' or MULTIPLE SECURITIES
    :param frequency: 'A' or 'Q' for annual or quarterly, respectively
    :return: obvious..
    """
    df = financials_download(ticker, 'cf', frequency)
    return df.loc["Free cash flow"]


def gross_margin(ticker):
    df = ratios_download(ticker)
    return df.loc["Gross Margin %"]


def operating_margin(ticker):
    df = ratios_download(ticker)
    return df.loc["Operating Margin %"]


def eps(ticker):
    df = ratios_download(ticker)
    return df.loc["Earnings Per Share USD"]


def dps(ticker):
    df = ratios_download(ticker)
    return df.loc["Dividends USD"]


def payout_ratio(ticker):
    df = ratios_download(ticker)
    return df.loc["Payout Ratio %"]


def bvps(ticker):
    df = ratios_download(ticker)
    return df.loc["Book Value Per Share USD"]


def operating_cf_usd(ticker):
    df = ratios_download(ticker)
    return df.loc["Operating Cash Flow USD Mil"]


def revenue_usd(ticker):
    df = ratios_download(ticker)
    return df.loc["Revenue USD Mil"]


def operating_income_usd(ticker):
    df = ratios_download(ticker)
    return df.loc["Operating Income USD Mil"]


def capex_usd(ticker):
    df = ratios_download(ticker)
    return df.loc["Cap Spending USD Mil"]


def fcf_usd(ticker):
    df = ratios_download(ticker)
    return df.loc["Free Cash Flow USD Mil"]


def fcf_pershare_usd(ticker):
    df = ratios_download(ticker)
    return df.loc["Free Cash Flow Per Share USD"]


def working_capital_usd(ticker):
    df = ratios_download(ticker)
    return df.loc["Working Capital USD Mil"]


def tax_rate(ticker):
    df = ratios_download(ticker)
    return df.loc["Tax Rate %"]


def net_margin(ticker):
    df = ratios_download(ticker)
    return df.loc["Net Margin %"]


def asset_turnover(ticker):
    df = ratios_download(ticker)
    return df.loc["Asset Turnover (Average)"]


def roa(ticker):
    df = ratios_download(ticker)
    return df.loc["Return on Assets %"]


def financial_leverage(ticker):
    df = ratios_download(ticker)
    return df.loc["Financial Leverage (Average)"]


def roe(ticker):
    df = ratios_download(ticker)
    return df.loc["Return on Equity %"]


def roic(ticker):
    df = ratios_download(ticker)
    return df.loc["Return on Invested Capital %"]


def interest_coverage(ticker):
    df = ratios_download(ticker)
    return df.loc["Interest Coverage"]


def revenue_yoy_growth(ticker):
    df = ratios_download(ticker)
    df = df[41:45]
    return df.loc["Year over Year"]


def revenue_3yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[41:45]
    return df.loc["3-Year Average"]


def revenue_5yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[41:45]
    return df.loc["5-Year Average"]


def revenue_10yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[41:45]
    return df.loc["10-Year Average"]


def operating_income_yoy_growth(ticker):
    df = ratios_download(ticker)
    df = df[46:50]
    return df.loc["Year over Year"]


def operating_income_3yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[46:50]
    return df.loc["3-Year Average"]


def operating_income_5yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[46:50]
    return df.loc["5-Year Average"]


def operating_income_10yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[46:50]
    return df.loc["10-Year Average"]


def net_income_yoy_growth(ticker):
    df = ratios_download(ticker)
    df = df[51:55]
    return df.loc["Year over Year"]


def net_income_3yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[51:55]
    return df.loc["3-Year Average"]


def net_income_5yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[51:55]
    return df.loc["5-Year Average"]


def net_income_10yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[51:55]
    return df.loc["10-Year Average"]


def eps_yoy_growth(ticker):
    df = ratios_download(ticker)
    df = df[56:60]
    return df.loc["Year over Year"]


def eps_3yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[56:60]
    return df.loc["3-Year Average"]


def eps_5yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[56:60]
    return df.loc["5-Year Average"]


def eps_10yr_avg_growth(ticker):
    df = ratios_download(ticker)
    df = df[56:60]
    return df.loc["10-Year Average"]


def current_ratio(ticker):
    df = ratios_download(ticker)
    return df.loc["Current Ratio"]


def debt_to_equity(ticker):
    df = ratios_download(ticker)
    return df.loc["Debt/Equity"]


def quick_ratio(ticker):
    df = ratios_download(ticker)
    return df.loc["Quick Ratio"]


def stock_exchange(symbol):
    return y_stats(symbol, 'x')


def price_current(symbol):
    return y_stats(symbol, 'l1')


def market_cap(symbol):
    return y_stats(symbol, 'j1')


def book_value(symbol):
    return y_stats(symbol, 'b4')


def dividend_yield(symbol):
    return y_stats(symbol, 'y')


def high_52_week(symbol):
    return y_stats(symbol, 'k')


def low_52_week(symbol):
    return y_stats(symbol, 'j')


def moving_average_50(symbol):
    return y_stats(symbol, 'm3')


def moving_average_200(symbol):
    return y_stats(symbol, 'm4')


def pe_ratio(symbol):
    return y_stats(symbol, 'r')


def forward_pe(symbol):
    return y_stats(symbol, 'r')


def peg_ratio(symbol):
    return y_stats(symbol, 'r5')


def price_to_sales(symbol):
    return y_stats(symbol, 'p5')


def price_to_book(symbol):
    return y_stats(symbol, 'p6')


def short_ratio(symbol):
    return y_stats(symbol, 's7')


def analyst_target(symbol):
    return y_stats(symbol, 't8')


def perc_change(symbol):
    return y_stats(symbol, 'k2')


def doll_change(symbol):
    return y_stats(symbol, 'c1')


def eps_estimate_quarter(symbol):
    return y_stats(symbol, 'e9')


def eps_estimate_year(symbol):
    return y_stats(symbol, 'e8')


def price_close(symbol):
    return y_stats(symbol, 'p')
