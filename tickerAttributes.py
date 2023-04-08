from yahoofinancials import YahooFinancials


def info(ticker_name):
    ticker_attr = {}

    yahoo_financials = YahooFinancials(ticker_name)
    data = yahoo_financials.get_stock_quote_type_data() 
    otherdata = yahoo_financials.get_summary_data()

    ticker_attr['longName'] = (data[ticker_name]['longName'])
    ticker_attr['timeZoneFullName'] = (data[ticker_name]['timeZoneFullName'])

    #otherdata
    ticker_attr['previousClose'] = (otherdata[ticker_name]['previousClose'])
    ticker_attr['open'] = (otherdata[ticker_name]['open'])
    ticker_attr['dayLow'] = (otherdata[ticker_name]['dayLow'])
    ticker_attr['dayHigh'] = (otherdata[ticker_name]['dayHigh'])
    ticker_attr['currency'] = (otherdata[ticker_name]['currency'])
    ticker_attr['currentPrice'] = yahoo_financials.get_current_price()
    return ticker_attr

if __name__ == "__main__":
    info = info(ticker_name)






