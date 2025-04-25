from typing import List, Optional
import pandas as pd
import logging

from data.models import (
    Price,
    FinancialMetrics, 
    LineItem, 
    CompanyNews
    )

logger = logging.getLogger(__name__)

class DataSource:
    """统一数据源接口，可选择使用ashares或financialdatasets实现"""
    def __init__(self, source_type: str = "financialdatasets"):
        """
        初始化数据源
        :param source_type: 数据源类型，可选 'financialdatasets' 或 'ashares'
        """
        self.source_type = source_type
    
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """获取股票价格数据"""
        if self.source_type == "financialdatasets":
            from .api import get_prices
        else:
            from .ashares import get_prices
        return get_prices(ticker, start_date, end_date)
    
    def get_financial_metrics(
        self, 
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[FinancialMetrics]:
        """获取财务指标数据"""
        if self.source_type == "financialdatasets":
            from .api import get_financial_metrics
        else:
            from .ashares import get_financial_metrics
        return get_financial_metrics(ticker, end_date, period, limit)
    
    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 10
    ) -> List[CompanyNews]:
        """获取公司新闻"""
        if self.source_type == "financialdatasets":
            from .api import get_company_news
        else:
            from .ashares import get_company_news
        return get_company_news(ticker, end_date, start_date, limit)
    
    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 10
    ) -> List[dict]:
        """获取内幕交易数据"""
        if self.source_type == "financialdatasets":
            from .api import get_insider_trades
        else:
            from .ashares import get_insider_trades
        return get_insider_trades(ticker, end_date, start_date, limit)
    
    def search_line_items(
        self,
        ticker: str,
        line_items: List[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[LineItem]:
        """搜索财务指标项"""
        if self.source_type == "financialdatasets":
            from .api import search_line_items
        else:
            from .ashares import search_line_items
        return search_line_items(ticker, line_items, end_date, period, limit)
    
    def get_market_cap(
        self,
        ticker: str,
        end_date: str,
    ) -> float | None:
        """获取股票市值"""
        if self.source_type == "financialdatasets":
            from.api import get_market_cap
        else:
            from.ashares import get_market_cap
        return get_market_cap(ticker, end_date)

    def prices_to_df(prices: list[Price]) -> pd.DataFrame:
        """Convert prices to a DataFrame."""
        df = pd.DataFrame([p.model_dump() for p in prices])
        df["Date"] = pd.to_datetime(df["time"])
        df.set_index("Date", inplace=True)
        numeric_cols = ["open", "close", "high", "low", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.sort_index(inplace=True)
        return df


    # Update the get_price_data function to use the new functions
    def get_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        prices = self.get_prices(ticker, start_date, end_date)
        return self.prices_to_df(prices)