
import numpy as np
from numpy import zeros
import datetime as dt
from optionlab import run_strategy
import matplotlib.pyplot as plt

#
def calculate_strategy_inputs(stock_name, stock_price, stock_n, option_stock_price, option_price, option_n, volatility, start_date, target_date, interest_rate):
    """
    Generate strategy inputs based on given parameters for a covered call strategy.

    :param stock_name: Name of the stock.
    :param stock_price: Current price of the stock.
    :param stock_n: Number of stocks to buy.
    :param option_stock_price: Strike price of the call option.
    :param option_price: Premium of the call option.
    :param option_n: Number of call options to sell.
    :param volatility: Volatility of the stock.
    :param start_date: Start date of the strategy.
    :param target_date: Target date for the strategy.
    :param interest_rate: Interest rate for calculating the cost of carry.
    :return: Tuple containing stock name and strategy inputs dictionary.
    """
    min_stock = stock_price - round(stock_price * 0.5, 2)
    max_stock = stock_price + round(stock_price * 0.5, 2)
    strategy = [
        {"type": "stock", "n": stock_n, "action": "sell"},
        {"type": "call", "strike": option_stock_price, "premium": option_price, "n": option_n, "action": "buy"},
    ]

    inputs = {
        "stock_price": stock_price,
        "start_date": start_date,
        "target_date": target_date,
        "volatility": volatility,
        "interest_rate": interest_rate,
        "min_stock": min_stock,
        "max_stock": max_stock,
        "strategy": strategy,
    }

    return stock_name, inputs



def plot_strategy(out, stock_name, stock_price,option_stock_price,option_price):
    """
    Plot the profit/loss graph for the covered call strategy.

    :param out: Output object from the strategy simulation.
    :param stock_name: Name of the stock.
    :param stock_price: Current price of the stock.
    """
    s, pl_total = out.data.stock_price_array, out.data.strategy_profit
    leg = []

    for i in range(len(out.data.profit)):
        leg.append(out.data.profit[i])

    zeroline = zeros(s.shape[0])
    plt.xlabel("Stock price")
    plt.ylabel("Profit/Loss")

    min_stock = stock_price - round(stock_price * 0.5, 2)
    max_stock = stock_price + round(stock_price * 0.5, 2)

    abs_min_return = abs(out.minimum_return_in_the_domain)
    abs_max_return = abs(out.maximum_return_in_the_domain)
    max_abs_return = max(abs_min_return, abs_max_return)

    plt.xlim(min_stock, max_stock)
    plt.ylim(-max_abs_return*1.5, max_abs_return*1.5)
    plt.plot(s, zeroline, "m-")
    
   
    
    plt.plot(s, pl_total, "k-", label="Protective Call")
    #plt.plot(s, leg[0], "r--", label="Long Stock")
    #plt.plot(s, leg[1], "b--", label="Short Call")
    plt.legend(loc="upper left")

    low, high = 0, 0
    profit_ranges = out.profit_ranges  
    
    profit_range_str = ""
    for low, high in profit_ranges:
        profit_range_str += f"\n {low:.2f} ---> {high:.2f}"
        
        
        
    plt.title(
        f"Stock name: {stock_name}\n"
        f"Stock price: {stock_price:.2f}\n"
        f"Option Stock price: {option_stock_price:.2f}\n"
        f"Option price: {option_price:.2f}\n"
        f"Strategy cost: {out.strategy_cost:.2f}\n"
        f"Maximum loss: {abs(out.minimum_return_in_the_domain):.2f}\n"
        f"Maximum profit: {out.maximum_return_in_the_domain:.2f}\n"
        f"Probability of Profit (PoP): {out.probability_of_profit * 100.0:.1f}%\n"
        f"Profitable stock price range:  {profit_range_str} "
    )
    
    
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def display_results(out):
    """
    Display the calculated results of the strategy.

    :param out: Output object from the strategy simulation.
    """
    print("\nCalculation Results")
    print(f"Strategy cost: {out.strategy_cost:.2f}")
    print(f"Maximum loss: {abs(out.minimum_return_in_the_domain):.2f}")
    print(f"Maximum profit: {out.maximum_return_in_the_domain:.2f}")
    print("Profitable stock price range:")
    for low, high in out.profit_ranges:
        print(f"      {low:.2f} ---> {high:.2f}")
    print(f"Probability of Profit (PoP): {out.probability_of_profit * 100.0:.1f}%")

def main() -> None:
    """
    Main function to execute the covered call strategy calculation and visualization.

    :return: None
    """
    stock_name = "CLSK"
    stock_price = 160.00
    volatility = 0.272
    start_date = dt.date(2021, 11, 22)
    target_date = dt.date(2021, 12, 17)
    interest_rate = 0.0002
    stock_n= 100
    option_stock_price=170
    option_price = 1.5
    option_n = 200
    stock_name, inputs = calculate_strategy_inputs(stock_name, stock_price, stock_n,option_stock_price,option_price,option_n,volatility, start_date, target_date, interest_rate)
    out = run_strategy(inputs)
    plot_strategy(out, stock_name,stock_price,option_stock_price,option_price)
    display_results(out)

if __name__ == "__main__":
    main()
