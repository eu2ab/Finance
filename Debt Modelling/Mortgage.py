import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def monthly_pmt(principal, rate, term):
    """
        Calculates the payment on a loan

        :param principal: Principal loan amount outstanding
        :param rate: Interest rate, shown as an APR ('.04' is 4%)
        :param term: Number of months left
        :return: Monthly payment
        """
    rate_monthly = ((1 + rate) ** (1/12)) - 1  # monthly payment rate
    pmt = round(principal * (rate_monthly * (1 + rate_monthly)**(term)) /
                ((1 + rate_monthly)**(term) - 1), 2)  # monthly payment total
    return (pmt)


def amortization_table(principal, rate, term):
    """
    Creates amortization table, including: payment no., beginning balance, payment,
    principal, interest, end balance

    :param principal: Principal loan amount outstanding
    :param rate: Interest rate, shown as an APR ('0.04' is 4%)
    :param term: Number of months left
    :return: Amortization table
    """

    # Build empty dataframe to fill in
    headers = ['Beginning Balance', 'Payment', 'Interest', 'Principal', 'Ending Balance']
    index = list(range(1, term + 1))
    df = pd.DataFrame(index=index, columns=headers)
    rate_monthly = ((1 + rate) ** (1 / 12)) - 1  # Monthly payment rate

    # Populate first row of data
    df.iloc[0,0] = principal
    df.iloc[0,1] = round(monthly_pmt(principal, rate, term), 2)
    df.iloc[0,2] = round(principal * rate_monthly, 2)
    df.iloc[0,3] = df.iloc[0,1] - round(principal * rate_monthly, 2)
    df.iloc[0,4] = principal - df.iloc[0,3]

    # Populate rest of amortization schedule based on data after first row
    for i in range(1, term):
        df.iloc[i, 0] = df.iloc[i - 1, 4]  # Takes the ending balance from the previous row
        df.iloc[i, 1] = round(monthly_pmt(principal, rate, term), 2)  # Monthly payment is always the same
        df.iloc[i, 2] = round(df.iloc[i, 0] * rate_monthly, 2)
        df.iloc[i, 3] = df.iloc[i, 1] - df.iloc[i, 2]
        df.iloc[i, 4] = df.iloc[i, 0] - df.iloc[i, 3]

    return (df)


def amortization_int_princ_graph(df):
    """
    Function returns a graph of the interest and principal payments of the amortization schedule

    :param df: Dataframe containing the amortization table, configured based on the formula above
    :param return: Graph of the interest and principal payments of the amortization schedule
    """

    # Extracting only relevant columns, in this case the data on interest and principal payments
    ts = df[["Interest", "Principal"]]

    # Find the breakeven point during which principal starts exceeding interest payments
    ts_diff = ts.Interest - ts.Principal
    ts_diff = ts_diff.astype(float)
    find_t = np.sign(ts_diff) != np.sign(ts_diff.shift(1))
    truth_num = find_t.index[find_t].tolist()  # Find index number where True (where signs will have crossed)
    truth_num = truth_num[1]  # Skips the first item since that item was made True due to a NaN earlier in the process
    int_int = ts.iloc[truth_num].Interest   # Value of interest payment at intersecting loan
    int_princ = ts.iloc[truth_num].Principal    # Value of Principal payment at intersecting loan
    midpoint = (int_int + int_princ) / 2  # Average both payments to find midpoint y coordinate

    # Start plotting out the graph
    ts.plot(title="Interest and Principal Payments of Amortization Schedule")
    plt.annotate('Breakeven: ${0} at {1} months'.format(round(midpoint, 2), truth_num),
                 xy=(truth_num, midpoint))  # Annotation for the breakeven point
    plt.xlabel("Length of Loan (months)")
    plt.ylabel("Value of Payment (in USD)")
    plt.show()  # Actual shows the plot in  interactive mode


def irr(deposit, cf_monthly, term):
    """
    Returns an IRR given outflows, inflows, and term length

    :param deposit: Initial amount paid for property
    :param cf_monthly: Amount earned each month, net of income and expenses
    :param term: Number of months on the loan
    :return: IRR calculation
    """
    df = []
    df.append(-deposit)
    for i in range(1, term):
        df.append(cf_monthly)
    calc = round(np.irr(df), 3)
    return (calc)


def mortgage_stats(rent, value, principal, interest_rate, term,
                   insurance_rate=0.035, tax_rate=0.1, vacancy_rate=0.07, hoa=150):
    """
    A summary of mortgage statistics

    :param rent: Anticipated monthly rental income
    :param value: Total value of home, represented in USD
    :param principal: Total value of loan, represented in USD
    :param interest_rate: Interest rate on loan, presented as APR (4% is '0.04')
    :param term: Number of months
    :param insurance_rate: Cost of insurance, represented as % of value (default = 0.035, from Zillow)
    :param tax_rate: Cost of tax, represented as a % of value
                    (default = 0.10 to represent VA real estate rate)
    :param vacancy_rate: Expected vacancy rate (default = 0.07, the average US rate)
    :param hoa: Monthly home owner association fees, represented in USD (default = 150)
    :return: A table of mortgage statistics
    """
    deposit = value - principal
    payment = monthly_pmt(principal, interest_rate, term) + (insurance_rate * value / 12) + \
              (tax_rate * value / 12) + hoa  # Expected monthly payment
    income = rent - (vacancy_rate * rent / 12)  # Anticipated monthly income, adjusted for vacancy rate
    cf_monthly = income - payment

    breakeven_months = deposit / cf_monthly  # Represents the number of months needed to make the deposit back
    roi = (cf_monthly * 12) / deposit  # Return on investment; cap rate is ROI if you paid full cash
    ltv_ratio = principal / value  # Loan to value ratio

    # building IRR calculation
    irr1 = irr(deposit, cf_monthly, term)

    # printing the material
    print('Total Monthly Expenses: ' + str(payment) + '\nTotal Monthly Income: ' + str(income) +
          '\nNet Monthly Cash Inflow: ' + str(cf_monthly) + '\nROI: ' + str(roi) + '\nIRR: ' + str(irr1) +
          '\nLoan-to-Value Ratio: ' + str(ltv_ratio) + '\nNumber of Months Before Breakeven: ' + str(breakeven_months))
