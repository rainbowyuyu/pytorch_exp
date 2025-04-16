import pandas as pd
import matplotlib.pyplot as plt


# 定义一个读取股票数据的函数，输入有两个：一个是需要待读取股票的公司名称，第二个是日期创建DataFrame对象，
# 其中行标签为dates
def stocks_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
    # 利用read_csv读取股票数据的文件
    # index_col：设定索引
    # parse_dates：指定某些列为时间类型
    # usecols:读取的时候只想要使用到的列
    # na_values:该参数可以配置哪些值需要处理成 NaN
        df_temp = pd.read_csv("data/{}.us.txt".format(symbol), index_col='Date',
                    parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        # df_temp.rename：修改标签名
        df_temp = df_temp.rename(columns={'Close': symbol})
        # 将读取的df_temp依次添加至df
        df = df.join(df_temp) 
    return df


if __name__ == '__main__':
    # 创建时间数据的索引，输入分别为：日期的起点、日期的终点、指定的计时单位（D：日历日、B：每工作日）
    dates = pd.date_range('2015-01-01','2016-12-31',freq='B')
    symbols = ['goog','ibm', 'aapl']
    df = stocks_data(symbols, dates)
    # nan数据填充
    # df.fillna(method='ffill')
    df.ffill()
    axes = df.plot(figsize=(10, 6), grid=True, subplots=True)
    plt.tight_layout()
    plt.savefig('output/output_plot.png')