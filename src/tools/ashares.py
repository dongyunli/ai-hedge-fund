import akshare as ak
import pandas as pd
from datetime import datetime
import numpy as np
import logging
from typing import List, Optional

from data.cache import get_cache
from data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)

# 复用现有缓存和日志配置
_cache = get_cache()
logger = logging.getLogger(__name__)

def get_prices(ticker: str, start_date: str, end_date: str) -> List[Price]:
    """使用akshare获取A股历史行情数据"""
    if cached_data := _cache.get_prices(ticker):
        filtered_data = [Price(**price) for price in cached_data 
                        if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data
    sdate = start_date.replace('-','')
    edate = end_date.replace('-','')
    print(sdate,edate)
    try:
        # 获取A股历史行情数据
        df = ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=sdate,
            end_date=edate,
            adjust="qfq"  # 前复权
        )
        print(df)
        # 转换为Price对象列表
        prices = []
        for _, row in df.iterrows():
            prices.append(Price(
                time=str(row['日期']),
                open=float(row['开盘']),
                high=float(row['最高']),
                low=float(row['最低']),
                close=float(row['收盘']),
                volume=int(row['成交量'])
            ))
        
        # 缓存结果
        _cache.set_prices(ticker, [p.model_dump() for p in prices])
        return prices
        
    except Exception as e:
        logger.error(f"获取价格数据失败: {str(e)}")
        return []

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10
) -> List[FinancialMetrics]:
    """从Akshare获取财务指标数据"""
    if cached_data := _cache.get_financial_metrics(ticker):
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data 
                        if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        return filtered_data[:limit]

    metrics_dict = {
        "market_cap": None,  # 总市值需自行计算（股价*总股本）
        "enterprise_value": None,  # 企业价值需自行计算（市值+净负债-现金）
        "price_to_earnings_ratio": None,  # 市盈率 PE
        "price_to_book_ratio": None,  # 市净率 PB
        "price_to_sales_ratio": None,  # 市销率 PS
        "enterprise_value_to_ebitda_ratio": None,  # 市值/EBITDA, 可用毛利润替代
        "enterprise_value_to_revenue_ratio": None,  # 市值/营业收入
        "free_cash_flow_yield": None,  # 自由现金流/总市值
        "peg_ratio": None,  # PEG, 需用市盈率/净利润增长率 手动计算
        "gross_margin": "销售毛利率(%)",
        "operating_margin": "营业利润率(%)",
        "net_margin": "销售净利率(%)",
        "return_on_equity": "加权净资产收益率(%)",
        "return_on_assets": "总资产净利润率(%)",
        "return_on_invested_capital": '投资收益率(%)',  # 净利润/投资资本
        "asset_turnover": "总资产周转率(次)",
        "inventory_turnover": "存货周转率(次)",
        "receivables_turnover": "应收账款周转率(次)",
        "days_sales_outstanding": "应收账款周转天数(天)",
        "operating_cycle":['存货周转率(次)','应收账款周转率(次)'],  # 需用存货周转天数+应收周转天数 手动计算
        "working_capital_turnover": None,  # 总收入/平均营运成本
        "current_ratio": "流动比率",
        "quick_ratio": "速动比率",
        "cash_ratio": "现金比率(%)",
        "operating_cash_flow_ratio": "经营现金净流量对负债比率(%)",
        "debt_to_equity": "负债与所有者权益比率(%)",
        "debt_to_assets": "资产负债率(%)",
        "interest_coverage": "利息支付倍数",
        "revenue_growth": "主营业务收入增长率(%)",
        "earnings_growth": "净利润增长率(%)",
        "book_value_growth": "净资产增长率(%)",
        "earnings_per_share_growth": None,  # 需用摊薄每股收益增长率 手动计算
        "free_cash_flow_growth": None,  # 自由现金流增长率手动计算
        "operating_income_growth": None,  # 需用营业利润增长率 手动计算
        "ebitda_growth": '营业利润率(%)',  # 国内报表无EBITDA标准指标
        "payout_ratio": "股息发放率(%)",
        "earnings_per_share": "摊薄每股收益(元)",
        "book_value_per_share": "每股净资产_调整前(元)",
        "free_cash_flow_per_share": "每股经营性现金流(元)"
    }
    metrics = []
    try:
        # 获取财务指标数据
        df = ak.stock_financial_analysis_indicator(symbol=ticker, start_year='2000')
        df = df[df['日期'] <= datetime.strptime(end_date,'%Y-%m-%d').date()]
        df['日期'] = pd.to_datetime(df['日期'])
        if period == 'annual':
            df = df.groupby(df['日期'].dt.year).first()  # 移除reset_index()
            df = df.reset_index(drop=True)  # 使用drop=True避免列名冲突
        else:
            # TTM取最近12个月的数据，quarter都取所有数据
            pass
        #降序排列并取前limit条数据
        df = df.sort_values(by='日期', ascending=False)
        df = df.head(limit)
        # 遍历df中的每一行并修正错误数据
        df = df.fillna(0)
        df = df.replace('--', 0)
        df = df.replace('...', 0)
        df = df.replace('N/A', 0)
        df = df.replace('NaN', 0)
        df = df.replace('None', 0)
        df = df.replace('null', 0)
        df = df.replace('NULL', 0)
        
        # 获取估值分析
        svdf = ak.stock_value_em(symbol=ticker)
        # 获取df中的所有日期
        df_dates = df['日期'].unique()
        matched_rows = []
        for date in df_dates:
            # 查找小于等于当前日期的所有svdf行
            candidates = svdf[svdf['数据日期'] <= date.date()].copy()  # 使用copy()避免警告
            if not candidates.empty:
                # 取小于等于date的最大日期行
                closest_row = candidates.loc[candidates['数据日期'].idxmax()].copy()
                # 将数据日期修改为df的日期
                closest_row['数据日期'] = date
                matched_rows.append(closest_row)
        # 创建新的匹配数据框
        svdf = pd.DataFrame(matched_rows)
        svdf = svdf.sort_values(by='数据日期', ascending=False)

        # 遍历df中并组装数据
        for _, row in df.iterrows():
            kvs = dict()
            # 遍历metrics_dict中的属性
            for attr, col_name in metrics_dict.items():
                if col_name is None:
                    continue

                value = 0
                # print(attr,col_name)
                try:
                    # 处理多个列名的情况(如存货周转率+应收周转率)
                    if isinstance(col_name, list):
                        # 计算列的和
                        for col in col_name:
                            value += float(row[col]) if pd.notna(row[col]) else 0
                            # print(row[col])
                    else:
                        value = float(row[col_name]) if pd.notna(row[col_name]) else None
                        # print(row[col_name])
                    kvs[attr] = value
                except Exception as e:
                    logger.warning(f"无法获取属性 {col_name}: {str(e)}")
                    continue
            
            # 添加估值指标
            if not svdf.empty:
                sv_row = svdf[svdf['数据日期'] == row['日期']]
                if not sv_row.empty:
                    try:
                        kvs['market_cap'] = float(sv_row['总市值'])
                        kvs["enterprise_value"] = float(sv_row['总市值']) + float(row['总资产(元)'])*(float(row['资产负债率(%)']) - float(row['现金比率(%)']))/100
                        kvs["price_to_earnings_ratio"] = float(sv_row['PE(TTM)'])
                        kvs["price_to_book_ratio"] = float(sv_row['市净率'])
                        kvs["price_to_sales_ratio"] = float(sv_row['市销率'])
                        kvs["enterprise_value_to_ebitda_ratio"] = float(sv_row['总市值'])/float(row['主营业务利润(元)'])
                        kvs["enterprise_value_to_revenue_ratio"] = float(sv_row['总市值'])/(float(row['主营业务利润(元)'])*100/float(row['主营业务利润率(%)']))
                        # kvs["free_cash_flow_yield"]= float(sv_row['自由现金流'])/float(sv_row['总市值'])
                        kvs["peg_ratio"] = float(sv_row['PEG值'])
                    except Exception as e:
                        logger.warning(f"无法获取估值指标: {str(e)}")
                        continue
            metric = FinancialMetrics(
                ticker=ticker,report_period=str(row['日期']),period=period,currency='CNY',
                free_cash_flow_yield=0 ,working_capital_turnover=0, earnings_per_share_growth=0,free_cash_flow_growth=0,operating_income_growth=0,
                **kvs
            )
            #print(kvs)
            #print(metric)
            metrics.append(metric)
    except Exception as e:
        logger.error(f"获取财务指标数据失败: {str(e)}")
        return []

    _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics])
    return metrics[:limit]

def search_line_items(
    ticker: str,
    line_items: List[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10
) -> List[LineItem]:
    """
    从AkShare获取财务指标数据
    ticker: 股票代码
    line_items: 要查询的指标名称列表
    end_date: 查询的截止日期
    period: 查询的周期，默认为ttm, 可选值 annual, quarterly, ttm 
    limit: 返回的最大记录数
    """
    income_func = ak.stock_financial_benefit_ths
    balance_func = ak.stock_financial_debt_ths
    cash_func = ak.stock_financial_cash_ths
    indicator='按报告期'
    if period == "ttm":
        pass
    elif period == "quarterly":
        indicator='按季度'
    elif period == "annual":
        indicator='按年度'
    else:
        raise ValueError("Invalid period. Must be one of 'annual', 'quarterly', or 'ttm'.")
    
    benifit_dict = {
    "consolidated_income": "一、营业总收入",
    "cost_of_revenue": "其中：营业成本",
    "earnings_per_share": "（一）基本每股收益",
    "earnings_per_share_diluted": "（二）稀释每股收益",
    "ebit": "三、营业利润",
    "income_tax_expense": "减：所得税费用",
    "interest_expense": "其中：利息费用",
    "net_income": "五、净利润",
    "net_income_common_stock": "*归属于母公司所有者的净利润",
    "net_income_non_controlling_interests": "少数股东损益",
    "operating_expense": "二、营业总成本",
    "operating_income": "三、营业利润",
    "research_and_development": "研发费用",
    "revenue": "其中：营业收入",
    "gross_profit": "四、利润总额", # 计算方式
    "dividends_per_common_share": None,
    "ebit_usd": None,
    "earnings_per_share_usd": None,
    "net_income_common_stock_usd": None,
    "net_income_discontinued_operations": None,
    "preferred_dividends_impact": None,
    "revenue_usd": None,
    "selling_general_and_administrative_expenses": ['销售费用','管理费用','财务费用'],
    "weighted_average_shares": None,
    "weighted_average_shares_diluted": None
    }
    balance_dict = {
    "accumulated_other_comprehensive_income": "其他综合收益",
    "cash_and_equivalents": "货币资金",
    "current_assets": "流动资产合计",
    "current_debt": "一年内到期的非流动负债",
    "current_investments": "交易性金融资产",
    "current_liabilities": "流动负债合计",
    "deferred_revenue": "合同负债",
    "deposit_liabilities": '应付账款',
    "goodwill_and_intangible_assets": ["商誉", "无形资产"],  # 需合并计算
    "inventory": "存货",
    "investments": "非流动金融资产",
    "non_current_assets": "非流动资产合计",
    "non_current_debt": "长期借款",
    "non_current_investments": "其他非流动金融资产",
    "non_current_liabilities": "非流动负债合计",
    "outstanding_shares": "实收资本（或股本）",
    "property_plant_and_equipment": "固定资产合计",
    "retained_earnings": "未分配利润",
    "shareholders_equity": "*所有者权益（或股东权益）合计",
    "tax_assets": "递延所得税资产",
    "tax_liabilities": "递延所得税负债",
    "total_assets": "*资产合计",
    "total_debt": ["短期借款", "长期借款"],  # 需合并流动与非流动负债中的借款
    "total_liabilities": "*负债合计",
    "trade_and_non_trade_payables": "应付票据及应付账款",
    "trade_and_non_trade_receivables": "应收票据及应收账款",
    # 无直接对应项的字段
    "cash_and_equivalents_usd": None,
    "shareholders_equity_usd": None,
    "total_debt_usd": None
    }
    cash_dict = {
        "net_cash_flow_from_operations": "*经营活动产生的现金流量净额",
        "net_cash_flow_from_investing": "*投资活动产生的现金流量净额",
        "net_cash_flow_from_financing": "*籌资活动产生的现金流量净额",
        "change_in_cash_and_equivalents": "*现金及现金等价物净增加额",
        "capital_expenditure": "购建固定资产、无形资产和其他长期资产支付的现金",
        "depreciation_and_amortization": [
            "固定资产折旧、油气资产折耗、生产性生物资产折旧",
            "无形资产摊销"
        ],  # 需合并间接法中的折旧摊销项
        "dividends_and_other_cash_distributions": "分配股利、利润或偿付利息支付的现金",
        "effect_of_exchange_rate_changes": "四、汇率变动对现金及现金等价物的影响",
        "issuance_or_repayment_of_debt_securities": [
            "取得借款收到的现金",
            "偿还债务支付的现金"
        ],  # 需合并筹资活动中的债务相关现金流
        "business_acquisitions_and_disposals": [
            "处置子公司及其他营业单位收到的现金净额",
            "取得子公司及其他营业单位支付的现金净额"
        ],  # 需合并企业并购现金流
        "investment_acquisitions_and_disposals": [
            "收回投资收到的现金",
            "投资支付的现金"
        ],  # 需合并投资活动中的买卖现金流
        "share_based_compensation": None,  # 股权激励,现金流量表无直接对应项
        # 以下字段需特殊处理
        "issuance_or_purchase_of_equity_shares": [
            "吸收投资收到的现金", 
            "分配股利、利润或偿付利息支付的现金"
        ]  # 含股权融资与回购（部分需拆分）
    }
    try:
        income_df = income_func(symbol=ticker, indicator=indicator)
        balance_df = balance_func(symbol=ticker, indicator=indicator)
        cash_df = cash_func(symbol=ticker, indicator=indicator)
        if income_df.empty and balance_df.empty and cash_df.empty:
            return []
        # 预处理所有数据框中的数值
        for df in [income_df, balance_df, cash_df]:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: 
                        float(x.replace('亿',''))*100000000 if '亿' in str(x) 
                        else float(x.replace('万',''))*10000 if '万' in str(x) 
                        else float(x) if str(x).replace('.','').isdigit() 
                        else x)
        
        # 初始化结果列表
        result = []
        
        # 遍历请求的line_items
        for item in line_items:
            # 检查项目在哪个字典中
            if item in benifit_dict:
                cn_header = benifit_dict[item]
                df_source = income_df
            elif item in balance_dict:
                cn_header = balance_dict[item]
                df_source = balance_df
            elif item in cash_dict:
                cn_header = cash_dict[item]
                df_source = cash_df
            else:
                continue
                
            if not cn_header:
                continue

            # 处理多个表头的情况(如销售费用、管理费用等、暂时只支持加法)
            if isinstance(cn_header, list):
                values = df_source[cn_header].sum(axis=1,skipna=True)
                values = values.replace([np.inf, -np.inf], np.nan)  # 处理可能的无穷大值
            else:
                values = df_source[cn_header]
                
            # 将数据转换为LineItem对象
            for idx, row in df_source.iterrows():
                report_period = str(row['报告期'])
                value = float(values.iloc[idx]) if not pd.isna(values.iloc[idx]) else None
                lineItem = LineItem(
                    ticker=ticker,
                    report_period=report_period,
                    period=period,
                    currency=''
                    )
                setattr(lineItem, item, value)
                print(lineItem)
                result.append(lineItem)
        # 按报告期排序并限制返回数量
        result.sort(key=lambda x: x.report_period, reverse=True)
        return result[:limit]
        
    except Exception as e:
        logger.error(f"获取表数据失败: {str(e)}")
        return []

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000
) -> List[InsiderTrade]:
    """使用akshare获取内部交易数据"""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data
    try:
        # 获取高管持股变动数据
        df = ak.stock_management_change_ths(symbol=ticker)
        # 转换为InsiderTrade对象列表
        trades = []
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
        for _, row in df.iterrows():
            trade_date = row['变动日期'] 
            # 过滤日期范围
            if start_date_dt and trade_date < start_date_dt:
                continue
            if trade_date > end_date_dt:
                continue
            # 处理变动数量字符串
            change_str = row['变动数量']
            if '减持' in change_str:
                # 提取数字部分并转换为负数
                num_str = change_str.replace('减持', '').replace('万', '0000').replace('亿', '00000000')
                transaction_shares = -int(float(num_str))
            elif '增持' in change_str:
                # 提取数字部分并转换为正数
                num_str = change_str.replace('增持', '').replace('万', '0000').replace('亿', '00000000')
                transaction_shares = int(float(num_str))
            else:
                # 默认处理
                transaction_shares = int(float(change_str))

            rs_str = row['剩余股数']
            num_str = rs_str.replace('万', '0000').replace('亿', '00000000')
            shares_owned_after_transaction = int(float(num_str))
            shares_owned_before_transaction = shares_owned_after_transaction + transaction_shares

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=row['股份变动途径'],
                name=row['变动人'],
                title=row['与公司高管关系'],
                is_board_director=True,
                transaction_shares=transaction_shares,
                transaction_price_per_share=float(row['交易均价']) if row['交易均价'] else None,
                transaction_value=transaction_shares*float(row['交易均价']) if row['交易均价'] else None,
                shares_owned_after_transaction=shares_owned_after_transaction,
                shares_owned_before_transaction=shares_owned_before_transaction,
                security_title=ticker,
                transaction_date=str(row['变动日期']) if row['变动日期'] else None,
                filing_date=str(row['变动日期']) if row['变动日期'] else None
            ))
            
            if len(trades) >= limit:
                break
                
        # 按交易日期降序排序
        trades.sort(key=lambda x: x.transaction_date, reverse=True)
        
        # 缓存结果
        _cache.set_insider_trades(ticker, [t.model_dump() for t in trades])
        return trades
        
    except Exception as e:
        logger.error(f"获取内部交易数据失败: {str(e)}")
        return []

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000
) -> List[CompanyNews]:
    """使用akshare获取公司新闻数据"""
    try:
        # 获取A股公司新闻
        news_df = ak.stock_news_em(symbol=ticker)
        
        # 转换为CompanyNews对象列表
        news_list = []
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        
        for _, row in news_df.iterrows():
            news_date = datetime.strptime(row['发布时间'], '%Y-%m-%d %H:%M:%S')
            # 过滤日期范围
            if start_date_dt and news_date < start_date_dt:
                continue
            if news_date > end_date_dt:
                continue
            news_list.append(CompanyNews(
                ticker=ticker,
                author='东财网',
                date=row['发布时间'],
                title=row['新闻标题'],
                content=row['新闻内容'],
                source=row['文章来源'],
                url=row['新闻链接']
            ))
            
            if len(news_list) >= limit:
                break
                
        # 按日期降序排序
        news_list.sort(key=lambda x: x.date, reverse=True)
        
        # 缓存结果
        _cache.set_company_news(ticker, [n.model_dump() for n in news_list])
        return news_list
        
    except Exception as e:
        logger.error(f"获取公司新闻数据失败: {str(e)}")
        return []
def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from API
        df = ak.stock_individual_info_em(ticker)
        if df.empty:
            return None
        return float(df['总市值'])

    financial_metrics = get_financial_metrics(ticker, end_date)
    if not financial_metrics:
        return None
    
    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap
