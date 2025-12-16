# QStock Collector

一个专业的美股市场数据收集和处理服务，从 qstock 项目中提取并独立开发，使用 Yahoo Finance 和 yahooquery 进行高效的股票数据收集。

## 🚀 功能特性

### 核心功能
- **美股数据收集**: 支持 S&P 500 和 NASDAQ 100 成分股数据收集
- **智能增量更新**: 自动检测本地数据，支持增量更新和全量下载
- **数据标准化**: 提供完整的数据清洗、异常检测和标准化处理
- **交易日历管理**: 自动获取和更新美股交易日历
- **批量处理**: 支持批量下载和并行数据处理
- **异常检测**: 智能检测价格异常并自动修正

### 数据源
- **Yahoo Finance**: 使用 yahooquery 库获取股票历史数据
- **Wikipedia**: 获取 S&P 500 和 NASDAQ 100 最新成分股列表
- **自动异常处理**: 检测和修正常见的数据异常（如价格单位错误等）

## 📋 环境要求

- **Python**: >= 3.12
- **操作系统**: Windows, macOS, Linux
- **网络**: 需要稳定的互联网连接访问 Yahoo Finance API

## 🛠 安装

### 1. 克隆项目
```bash
git clone https://github.com/nodew/qstock-collector.git
cd qstock-collector
```

### 2. 安装依赖
```bash
# 使用 pip 安装
pip install -e .

# 或使用 uv（推荐，更快）
uv sync
```

### 3. 环境配置
```bash
# 复制环境变量模板（可选）
cp .env.example .env
```

## 🚀 快速开始

### 每日数据更新（推荐）
一键完成交易日历更新、指数成分股更新、股票数据收集和标准化：

```bash
python main.py update_daily_data
```

### 分步操作

#### 1. 更新美股指数成分股
```bash
python main.py collect_us_index
```

#### 2. 更新交易日历
```bash
# 从 2015-01-01 开始更新
python main.py collect_us_calendar

# 从指定日期开始更新
python main.py collect_us_calendar --start_date "2020-01-01"
```

#### 3. 收集股票数据
```bash
# 收集所有股票的完整历史数据
python main.py collect_yahoo_data

# 收集指定日期范围的数据
python main.py collect_yahoo_data --start_date "2024-01-01" --end_date "2024-12-31"

# 限制处理的股票数量（用于测试）
python main.py collect_yahoo_data --limit_nums 10

# 调整请求间隔（秒）
python main.py collect_yahoo_data --delay 1.0
```

#### 4. 数据标准化处理
```bash
# 标准化所有数据
python main.py normalize_yahoo_data

# 指定处理的日期范围
python main.py normalize_yahoo_data --start_date "2024-01-01" --end_date "2024-12-31"

# 指定并行处理的工作进程数
python main.py normalize_yahoo_data --max_workers 8
```

## 📊 数据结构

### 目录结构
```
data/
├── calendar/           # 交易日历
│   └── us.txt         # 美股交易日历
├── instruments/        # 股票指数成分股
│   └── us.txt         # 美股指数成分股（S&P 500 + NASDAQ 100）
├── stock_data/         # 原始股票数据
│   └── us_data/       # 美股原始数据
│       ├── AAPL.csv
│       ├── MSFT.csv
│       └── ...
└── normalized_data/    # 标准化数据
    └── us_data/       # 美股标准化数据
        ├── AAPL.csv
        ├── MSFT.csv
        └── ...
```

### 数据格式

#### 原始数据（stock_data/us_data/）
```csv
date,open,high,low,close,volume,symbol
2024-01-01,150.00,152.00,149.50,151.50,1000000,AAPL
2024-01-02,151.50,153.00,150.00,152.50,1100000,AAPL
```

#### 标准化数据（normalized_data/us_data/）
```csv
date,open,high,low,close,volume,change,symbol
2024-01-01,1.0000,1.0133,0.9967,1.0100,1515000.0,0.0100,AAPL
2024-01-02,1.0100,1.0200,1.0000,1.0167,1670000.0,0.0066,AAPL
```

#### 指数成分股文件格式（instruments/us.txt）
```
AAPL	1999-01-01	2099-12-31
MSFT	1999-01-01	2099-12-31
GOOGL	2004-08-19	2099-12-31
```

## 🔧 配置说明

### 配置文件 (config.py)
主要配置项：

```python
class Settings(BaseSettings):
    # 交易日历
    calendar_dir: str = "data/calendar"
    us_calendar_path: str = "data/calendar/us.txt"

    # 指数成分股
    index_dir: str = "data/instruments"
    us_index_path: str = "data/instruments/us.txt"

    # 股票数据
    stock_data_dir: str = "data/stock_data"
    us_stock_data_dir: str = "data/stock_data/us_data"

    # 标准化数据
    normalized_data_dir: str = "data/normalized_data"
    us_normalized_data_dir: str = "data/normalized_data/us_data"

    # MongoDB 设置
    mongodb_url: str = 'mongodb://localhost:27017'
    database_name: str = 'qstock'
    jobs_collection: str = 'jobs'

    # Azure Communication Service 邮件设置
    acs_connection_string: str = ''
    acs_sender_email: str = ''
    acs_to_emails: str = ''  # 逗号分隔的收件人邮箱列表
```

### 环境变量 (.env)
可选的环境变量配置：

```bash
# 覆盖默认配置路径
US_STOCK_DATA_DIR=custom/path/to/stock/data
US_NORMALIZED_DATA_DIR=custom/path/to/normalized/data

# MongoDB 配置
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=qstock
JOBS_COLLECTION=jobs

# Azure Communication Service 邮件配置
ACS_CONNECTION_STRING=endpoint=https://your-acs-resource.communication.azure.com/;accesskey=your-access-key
ACS_SENDER_EMAIL=DoNotReply@your-domain.com
ACS_TO_EMAILS=user1@example.com,user2@example.com
```

## 📧 通知功能

### 任务状态通知
系统支持通过 Azure Communication Service (ACS) 发送任务执行状态通知邮件。

#### 配置步骤
1. 在 Azure 门户创建 Communication Service 资源
2. 配置发件人邮箱域名和验证
3. 获取连接字符串并配置到 `.env` 文件
4. 设置收件人邮箱列表

#### 通知内容
- 任务名称和状态（成功/失败）
- 开始时间、结束时间和执行时长
- 处理的股票数量和其他执行结果
- 错误信息（如果任务失败）

#### 数据库记录
任务状态会自动保存到 MongoDB 数据库中，包括：
- 任务执行记录
- 时间戳和执行时长
- 详细的执行结果
- 错误信息和警告

### 示例邮件内容
```
✅ QStock Daily Data Update - Success

Job Details
Job: QStock Daily Data Update (qstock_collector_daily_update)
Status: SUCCESS
Start Time: 2025-01-11T15:00:00
End Time: 2025-01-11T15:30:00
Duration: 30m 15s

Execution Results:
Last Trading Date: 2025-01-10
Symbols Processed: 603
Data Collected: True
Data Normalized: True
```

## 💡 高级功能

### 智能批量下载
系统根据日期范围和数据间隔自动调整批量下载的批次大小：

- **日线数据**: 根据时间跨度动态调整批次大小（5-50只股票/批次）
- **分钟数据**: 使用较小批次避免 API 限制（2-10只股票/批次）
- **自动重试**: 批量下载失败时自动降级到单个下载

### 异常数据检测与修正
- **价格异常**: 检测价格单位错误（如美分与美元混淆）
- **逻辑检验**: 验证 OHLC 数据的逻辑关系
- **极端变化**: 标记异常的价格变动（>50% 日涨跌幅）
- **自动修正**: 对检测到的异常自动进行修正处理

### 数据标准化处理
1. **基础标准化**: 时区处理、重复数据移除、异常修正
2. **价格复权**: 使用复权价格计算分割和分红调整
3. **相对标准化**: 所有数据相对于首日收盘价进行标准化

### 并行处理
- **数据收集**: 支持批量并行下载
- **数据处理**: 多进程并行标准化处理
- **自动调优**: 根据 CPU 核心数自动调整工作进程数

## 🐍 Python API 使用

### 基础使用
```python
from collectors.yahoo import collect_yahoo_data
from collectors.yahoo.normalize import normalize_yahoo_data
from collectors.us_index import collect_us_index
from collectors.us_calendar import collect_us_calendar

# 更新指数成分股
collect_us_index()

# 更新交易日历
collect_us_calendar(start_date="2020-01-01")

# 收集股票数据
collect_yahoo_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d",
    delay=0.5,
    limit_nums=100  # 仅处理前100只股票
)

# 标准化数据
normalize_yahoo_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
    max_workers=4
)
```

### 高级用法
```python
from collectors.yahoo.collector import YahooCollector
from collectors.yahoo.normalize import YahooNormalizer

# 创建收集器实例
collector = YahooCollector(
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d",
    delay=1.0,
    limit_nums=50
)

# 执行收集
collector.collect()

# 创建标准化器实例
normalizer = YahooNormalizer(
    start_date="2024-01-01",
    end_date="2024-12-31",
    max_workers=8
)

# 执行标准化
normalizer.normalize()
```

## 🔍 监控与日志

### 日志功能
- 使用 `loguru` 提供结构化日志
- 详细记录每个处理步骤的状态
- 异常股票自动记录到 `abnormal_tickers.txt`

### 进度监控
```bash
# 查看日志输出了解处理进度
python main.py collect_yahoo_data

# 示例输出：
# 2025-01-23 10:30:15 | INFO | Processing batch 1/10: ['AAPL', 'MSFT', 'GOOGL']
# 2025-01-23 10:30:20 | INFO | Retrieved 252 records for AAPL
# 2025-01-23 10:30:25 | INFO | Progress: 100/500 (Success: 98, Failed: 2)
```

## ⚠️ 注意事项

### API 限制
- Yahoo Finance 对请求频率有限制，建议保持默认的 0.5 秒间隔
- 批量下载时系统会自动处理速率限制
- 如遇到频繁的 API 错误，可增加 `--delay` 参数

### 数据质量
- 系统会自动检测和记录异常数据
- 异常股票会被标记并重新下载完整历史数据
- 建议定期查看 `abnormal_tickers.txt` 文件

### 磁盘空间
- 完整的 S&P 500 + NASDAQ 100 历史数据约需要 2-3 GB 空间
- 标准化数据会额外占用相似的存储空间
- 建议保留充足的磁盘空间

## 🚨 常见问题

### Q: 连接 Yahoo Finance 失败怎么办？
**A**:
1. 检查网络连接是否正常
2. 尝试增加请求间隔：`--delay 2.0`
3. 如果在某些地区访问受限，可考虑使用代理

### Q: 如何处理异常数据？
**A**:
1. 系统会自动检测异常数据并记录到 `abnormal_tickers.txt`
2. 异常股票会自动重新下载完整历史数据
3. 可手动删除有问题的 CSV 文件，系统会重新下载

### Q: 如何自定义股票列表？
**A**:
1. 编辑 `data/instruments/us.txt` 文件
2. 格式：`股票代码 TAB 开始日期 TAB 结束日期`
3. 重新运行数据收集命令

### Q: 数据不完整怎么办？
**A**:
1. 删除对应的 CSV 文件，系统会重新下载完整数据
2. 检查股票代码是否正确
3. 某些股票可能已退市或暂停交易

### Q: 如何提高处理速度？
**A**:
1. 增加并行工作进程数：`--max_workers 8`
2. 使用 SSD 硬盘存储数据
3. 确保网络连接稳定且带宽充足

## 📈 项目结构

```
qstock-collector/
├── collectors/                 # 数据收集器模块
│   ├── us_calendar/           # 美股交易日历收集器
│   │   └── collector.py
│   ├── us_index/              # 美股指数成分股收集器
│   │   └── collector.py
│   └── yahoo/                 # Yahoo Finance 收集器
│       ├── collector.py       # 数据收集
│       └── normalize.py       # 数据标准化
├── data/                      # 数据存储目录
├── config.py                  # 配置文件
├── main.py                    # 主入口文件
├── pyproject.toml            # 项目配置
└── README.md                 # 项目文档
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置
```bash
# 克隆项目
git clone https://github.com/nodew/qstock-collector.git
cd qstock-collector

# 安装开发依赖
uv sync --dev

# 运行测试
python -m pytest tests/
```

## 📄 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。

## 🔄 更新日志

### v0.1.0 (2025-01-25)
- ✅ 完整的美股数据收集系统
- ✅ 智能批量下载和增量更新
- ✅ 异常数据检测和自动修正
- ✅ 完整的数据标准化流程
- ✅ 并行处理和性能优化
- ✅ 综合的命令行界面

---

**QStock Collector** - 专业的美股数据收集和处理解决方案 🚀
