import pandas as pd
import numpy as np

def mortgage_payment(principal, rate, term):
    """
        Calculates the payment on a loan
        :param principal: principal
        :param rate: interest rate, shown as an APR (4% is '4')
        :param term: number of months
        :return: payment
        """
    ratePerTwelve = rate / (12 * 100.0)  # monthly payment rate
    result = principal * (ratePerTwelve / (1 - (1 + ratePerTwelve) ** (-term)))  # monthly payment total
    result = round(result, 2)
    return result

def amortization_table(principal, rate, term):
    """
    Creates amortization table, including: payment no., beginning balance, payment, principal, interest, end balance
    :param principal: principal payment on a loan
    :param rate: interest rate, shown as an APR (4% is '4')
    :param term: number of months
    :return: table
    """
    # build empty dataframe to fill in
    headers = ['Beginning Balance', 'Payment', 'Interest', 'Principal', 'Ending Balance']
    index = list(range(1, term+1, 1))
    df = pd.DataFrame(index=index, columns=headers)

    payment = mortgage_payment(principal, rate, term)
    begBal = principal

    for i in range(1, term + 1):
        df.ix[i, 0] = round(begBal, 2)
        df.ix[i, 1] = round(payment, 2)
        interest = begBal * (rate / (12 * 100))
        df.ix[i, 2] = round(interest, 2)
        principal = payment - interest
        df.ix[i, 3] = round(principal, 2)
        endBal = begBal - principal
        df.ix[i, 4] = round(endBal, 2)
        begBal = endBal

    return df

def irr(deposit, CF_monthly, term):
    """
    Returns an IRR given outflows, inflows, and term length.
    :param deposit: Initial amount paid for property
    :param CF_monthly: Amount earned each month, net of income and expenses
    :param term: Number of months on the loan
    :return: IRR calculation
    """
    df = []
    df.append(-deposit)
    for i in range(1, term):
        df.append(CF_monthly)
    calc = round(np.irr(df), 3)
    return calc

def mortgage_stats(rent, value, principal, interest_rate, term,
                   insurance_rate = 0.035, tax_rate = 1.1, vacancy_rate = 7, hoa = 150):
    """
    A summary of mortgage statistics
    :param rent: anticipated rental income
    :param value: total value of home, represented in USD
    :param principal: total value of loan, represented in USD
    :param interest_rate: interest rate on loan, presented as APR (4% is '4')
    :param term: number of months
    :param insurance_rate: cost of insurance, represented as % of value (default = 0.035, from Zillow)
    :param tax_rate: cost of tax, represented as a % of value(default = 1.1, to represent VA real estate rate)
    :param vacancy_rate: expected vacancy rate (default = 7, the average US rate)
    :param hoa: home owner association fees, represented in USD (default = 150)
    :return: A table of mortgage statistics
    """
    deposit = value - principal
    payment = mortgage_payment(principal, interest_rate, term) + (insurance_rate * value / 100) + \
              (tax_rate * value / (12 * 100)) + hoa  # expected monthly payment
    income = rent - (vacancy_rate * rent / (12 * 100))  # anticipated monthly income, adjusted for vacancy rate
    CF_monthly = income - payment

    breakeven_months = deposit / (CF_monthly)  # represents the number of months needed to make the deposit back
    roi = ((CF_monthly) * 12) / deposit  # return on investment; cap rate is ROI if you paid full cash
    ltv_ratio = principal/value  # loan to value ratio

    # building IRR calculation
    irr1 = irr(deposit, CF_monthly, term)

    #printing the material
    print('Total Monthly Expenses: '+str(payment)+'\nTotal Monthly Income: '+str(income)+
          '\nNet Monthly Cash Inflow: '+str(CF_monthly)+'\nROI: '+str(roi)+'\nIRR: '+str(irr1)+
          '\nLoan-to-Value Ratio: '+str(ltv_ratio)+'\nNumber of Months Before Breakeven: '+str(breakeven_months))