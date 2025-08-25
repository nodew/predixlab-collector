# QStock Market Data Service

一个独立的股票市场数据获取服务，从 qstock 项目中提取，使用 yfinance 和 Yahoo Query 进行数据收集。

## 功能特性

### 🚀 核心功能
- **多区域支持**: 支持美股 (US)、中国A股 (CN)、港股 (HK)、印度股市 (IN)、巴西股市 (BR)
- **多时间频率**: 支持日线数据 (1d) 和分钟数据 (1min)
- **股票指数管理**: 支持 SP500、NASDAQ100、CSI300、HSI 等主要指数
- **数据标准化**: 提供数据清洗和标准化功能
- **并发收集**: 支持多线程并发数据收集

### 📊 支持的数据源
- Yahoo Finance API
- Yahoo Query (yahooquery)
- 东方财富 API (中国市场)
- 深圳证券交易所 API (交易日历)

## 安装

### 环境要求
- Python >= 3.12
- 稳定的网络连接访问 Yahoo Finance

### 依赖安装
```bash
# 克隆项目
git clone <repository-url>
cd qstock-marketdata

# 安装依赖
pip install -e .
```

## 快速开始

### 1. 测试连接
```bash
python main.py test-connection
```

### 2. 查看服务信息
```bash
python main.py info
```

### 3. 获取股票符号列表
```bash
# 获取美股符号
python main.py list-symbols --region US --limit 10

# 获取中国A股符号
python main.py list-symbols --region CN --limit 10

# 获取港股符号
python main.py list-symbols --region HK --limit 10
```

### 4. 获取交易日历
```bash
# 获取美股交易日历
python main.py get-calendar --region US

# 获取中国A股交易日历
python main.py get-calendar --region CN
```

### 5. 数据收集
```bash
# 收集美股数据
python main.py collect --region US --interval 1d

# 收集指定市场数据
python main.py collect --region US --market sp500 --interval 1d

# 收集指定时间范围数据
python main.py collect --region US --start 2024-01-01 --end 2024-12-31
```

## API 使用

### Python 代码示例

```python
from src.utils import get_us_stock_symbols, get_calendar_list
from src.collectors import YahooCollectorUS1d

# 获取美股符号
symbols = get_us_stock_symbols()
print(f"获取到 {len(symbols)} 个美股符号")

# 获取交易日历
calendar = get_calendar_list("US_ALL")
print(f"获取到 {len(calendar)} 个交易日")

# 收集数据
collector = YahooCollectorUS1d(
    save_dir="./data",
    start="2024-01-01",
    end="2024-12-31",
    interval="1d"
)
collector.collector_data()
```

### 支持的区域和市场

| 区域 | 代码 | 支持的指数 | 示例符号格式 |
|------|------|------------|-------------|
| 美国 | US | SP500, NASDAQ100 | AAPL, MSFT, GOOGL |
| 中国 | CN | CSI300, CSI500 | 600000.ss, 000001.sz |
| 香港 | HK | HSI, HSTECH | 0700.HK, 0941.HK |
| 印度 | IN | NIFTY | RELIANCE.NS |
| 巴西 | BR | IBOV | PETR4.SA |

## 项目结构

```
qstock-marketdata/
├── src/
│   ├── collectors/          # 数据收集器
│   │   ├── __init__.py
│   │   ├── base.py         # 基础收集器类
│   │   └── yahoo.py        # Yahoo Finance 收集器
│   ├── index/              # 股票指数管理
│   │   ├── __init__.py
│   │   ├── base.py         # 指数基础类
│   │   └── utils.py        # 指数工具函数
│   └── utils/              # 工具函数
│       ├── __init__.py
│       └── common.py       # 通用工具函数
├── main.py                 # 主入口文件
├── pyproject.toml          # 项目配置
└── README.md               # 项目文档
```

## 配置说明

### 数据存储
- 默认数据保存在 `./data` 目录
- 每个股票符号对应一个 CSV 文件
- 支持增量更新模式

### 网络配置
- 默认请求间隔: 0.5 秒
- 默认重试次数: 5 次
- 支持代理配置 (需要环境变量)

### 并发设置
- 默认最大工作线程: 4
- 可通过 `--max-workers` 参数调整

## 常见问题

### Q: 连接 Yahoo Finance 失败怎么办？
A:
1. 检查网络连接
2. 如在中国大陆，可能需要配置代理
3. 尝试降低请求频率 (`--delay` 参数)

### Q: 如何获取指定股票的数据？
A:
```python
from src.collectors import YahooCollectorUS1d

# 创建自定义符号列表的收集器
collector = YahooCollectorUS1d(save_dir="./data")
collector.get_data("AAPL", "1d", start_date, end_date)
```

### Q: 支持哪些数据字段？
A:
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- volume: 成交量
- adjclose: 复权收盘价 (如果可用)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目基于 MIT 许可证开源。

## 更新日志

### v0.1.0 (2025-01-23)
- 初始版本发布
- 支持基础的数据收集功能
- 支持美股、中国A股、港股数据
- 提供命令行界面和 Python API
