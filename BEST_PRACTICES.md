# 代码最佳实践指南 - QStock Collector

## 📚 目录
1. [编码规范](#编码规范)
2. [性能优化](#性能优化)
3. [错误处理](#错误处理)
4. [测试策略](#测试策略)
5. [文档规范](#文档规范)
6. [Git 提交规范](#git-提交规范)

---

## 🎯 编码规范

### 1. 类型提示

**✅ 推荐做法：**
```python
from typing import Optional, List, Dict, Any
from pathlib import Path

def process_data(
    file_path: Path,
    symbols: List[str],
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """处理数据文件"""
    pass
```

**❌ 避免：**
```python
def process_data(file_path, symbols, config=None):
    """处理数据文件"""
    pass
```

### 2. 文档字符串

**✅ 推荐做法（Google 风格）：**
```python
def calculate_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    """计算收益率序列。

    Args:
        prices: 价格序列
        period: 计算周期，默认为 1

    Returns:
        收益率序列

    Raises:
        ValueError: 当价格序列为空时

    Examples:
        >>> prices = pd.Series([100, 105, 103])
        >>> returns = calculate_returns(prices)
        >>> print(returns)
    """
    if prices.empty:
        raise ValueError("价格序列不能为空")
    
    return prices.pct_change(period)
```

### 3. 常量命名

**✅ 推荐做法：**
```python
# 常量使用大写字母和下划线
DEFAULT_START_DATE = "2015-01-01"
MAX_RETRY_ATTEMPTS = 3
API_TIMEOUT_SECONDS = 30

# 配置类常量
class Config:
    DATABASE_URL = "mongodb://localhost:27017"
    MAX_WORKERS = 8
```

**❌ 避免：**
```python
default_start_date = "2015-01-01"  # 应该用大写
MaxRetryAttempts = 3  # 不要用驼峰命名常量
```

### 4. 函数长度

**原则：** 一个函数应该只做一件事，理想长度 < 50 行

**✅ 推荐做法：**
```python
def update_stock_data(symbol: str) -> bool:
    """更新股票数据"""
    data = fetch_data(symbol)
    if not validate_data(data):
        return False
    
    normalized = normalize_data(data)
    return save_data(normalized, symbol)

def fetch_data(symbol: str) -> pd.DataFrame:
    """获取数据"""
    # 专注于数据获取
    pass

def validate_data(data: pd.DataFrame) -> bool:
    """验证数据"""
    # 专注于数据验证
    pass
```

**❌ 避免：**
```python
def update_stock_data(symbol: str) -> bool:
    """更新股票数据"""
    # 100+ 行代码做所有事情
    # 数据获取、验证、处理、保存全在一个函数里
    pass
```

---

## ⚡ 性能优化

### 1. 向量化优先

**✅ 推荐做法（快 10-100x）：**
```python
import pandas as pd
import numpy as np

# 使用向量化操作
df['returns'] = df['close'].pct_change()
df['is_positive'] = df['returns'] > 0

# 使用 numpy 的向量化函数
with np.errstate(divide='ignore', invalid='ignore'):
    result = np.log(df['price'] / df['price'].shift(1))
```

**❌ 避免（慢）：**
```python
# 避免使用 apply 和 lambda
df['returns'] = df['close'].apply(lambda x: ...)

# 避免使用循环
for i in range(len(df)):
    df.loc[i, 'returns'] = calculate_return(df.loc[i, 'close'])
```

### 2. 内存管理

**✅ 推荐做法：**
```python
# 使用 copy() 明确表示需要副本
df_filtered = df[df['volume'] > 0].copy()
df_filtered['new_column'] = calculate_value(df_filtered)

# 及时删除不需要的大对象
large_df = load_large_data()
process_data(large_df)
del large_df  # 释放内存

# 使用生成器处理大文件
def read_large_file(file_path: Path):
    with open(file_path) as f:
        for line in f:
            yield process_line(line)
```

**❌ 避免：**
```python
# 不必要的数据复制
df_copy = df.copy()  # 如果不需要，不要复制
df_copy2 = df_copy.copy()  # 多次复制

# 在循环中创建大对象
for symbol in symbols:
    full_history = load_all_data(symbol)  # 可能导致内存溢出
```

### 3. 批处理

**✅ 推荐做法：**
```python
# 批量处理以减少 I/O
BATCH_SIZE = 100

for i in range(0, len(symbols), BATCH_SIZE):
    batch = symbols[i:i + BATCH_SIZE]
    data = fetch_batch_data(batch)  # 一次请求多个
    process_batch(data)
```

**❌ 避免：**
```python
# 逐个处理导致频繁 I/O
for symbol in symbols:
    data = fetch_data(symbol)  # 每次一个请求
    process_data(data)
```

---

## 🛡️ 错误处理

### 1. 具体的异常类型

**✅ 推荐做法：**
```python
from typing import Optional

def read_config(path: Path) -> Optional[Dict]:
    """读取配置文件"""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {e}")
        return None
    except PermissionError:
        logger.error(f"没有权限读取配置文件: {path}")
        return None
```

**❌ 避免：**
```python
def read_config(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # 太宽泛
        logger.error(f"错误: {e}")
        return None
```

### 2. 资源清理

**✅ 推荐做法：**
```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def get_db_connection() -> Generator[MongoClient, None, None]:
    """数据库连接上下文管理器"""
    client = None
    try:
        client = MongoClient(settings.mongodb_url)
        yield client
    finally:
        if client:
            client.close()

# 使用
with get_db_connection() as client:
    db = client[settings.database_name]
    # 使用数据库
    # 自动清理
```

**❌ 避免：**
```python
def save_to_db(data):
    client = MongoClient(settings.mongodb_url)
    db = client[settings.database_name]
    db.collection.insert_one(data)
    # 忘记关闭连接！
```

### 3. 自定义异常

**✅ 推荐做法：**
```python
class DataCollectionError(Exception):
    """数据收集错误基类"""
    pass

class DataValidationError(DataCollectionError):
    """数据验证错误"""
    pass

class APIRateLimitError(DataCollectionError):
    """API 限流错误"""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"API rate limit exceeded. Retry after {retry_after}s")

# 使用
try:
    data = fetch_data(symbol)
except APIRateLimitError as e:
    logger.warning(f"遇到限流，等待 {e.retry_after} 秒")
    time.sleep(e.retry_after)
```

---

## 🧪 测试策略

### 1. 单元测试结构

```python
import pytest
from pathlib import Path
from utils import read_last_trading_date

class TestReadLastTradingDate:
    """测试 read_last_trading_date 函数"""
    
    def test_normal_file(self, tmp_path):
        """测试正常的日历文件"""
        # Arrange
        calendar_file = tmp_path / "calendar.txt"
        calendar_file.write_text("2024-01-01\n2024-01-02\n2024-01-03\n")
        
        # Act
        result = read_last_trading_date(calendar_file)
        
        # Assert
        assert result == "2024-01-02"
    
    def test_empty_file(self, tmp_path):
        """测试空文件"""
        calendar_file = tmp_path / "empty.txt"
        calendar_file.write_text("")
        
        result = read_last_trading_date(calendar_file, default_date="2020-01-01")
        
        assert result == "2020-01-01"
    
    def test_missing_file(self, tmp_path):
        """测试不存在的文件"""
        calendar_file = tmp_path / "missing.txt"
        
        result = read_last_trading_date(calendar_file, default_date="2020-01-01")
        
        assert result == "2020-01-01"
    
    def test_invalid_date_format(self, tmp_path):
        """测试无效的日期格式"""
        calendar_file = tmp_path / "invalid.txt"
        calendar_file.write_text("not-a-date\n")
        
        with pytest.raises(ValueError):
            read_last_trading_date(calendar_file)
```

### 2. 集成测试

```python
@pytest.mark.integration
class TestDataCollection:
    """集成测试：完整的数据收集流程"""
    
    @pytest.fixture
    def collector(self):
        """创建收集器实例"""
        return YahooCollector(
            start_date="2024-01-01",
            end_date="2024-01-31",
            interval="1d",
            limit_nums=5  # 限制测试数据量
        )
    
    def test_full_collection_pipeline(self, collector, tmp_path):
        """测试完整的收集流程"""
        # 设置测试目录
        collector.us_stock_data_dir = tmp_path / "stock_data"
        
        # 执行收集
        collector.collect()
        
        # 验证结果
        csv_files = list(collector.us_stock_data_dir.glob("*.csv"))
        assert len(csv_files) > 0
        
        # 验证数据格式
        df = pd.read_csv(csv_files[0])
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in df.columns for col in required_cols)
```

### 3. 性能测试

```python
import pytest
import time

def test_vectorized_filter_performance():
    """测试向量化过滤的性能"""
    # 创建大数据集
    n = 100000
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='1min'),
        'close': np.random.randn(n)
    })
    
    collector = YahooCollector(interval="1d")
    
    # 测量性能
    start = time.time()
    result = collector._filter_data(df)
    duration = time.time() - start
    
    # 应该在合理时间内完成（< 1秒）
    assert duration < 1.0
    assert len(result) > 0
```

---

## 📖 文档规范

### 1. README 更新

每次重要功能添加后，更新 README：

```markdown
## 新功能：批量数据收集

### 使用方法
```python
from collectors.yahoo import YahooCollector

# 创建收集器
collector = YahooCollector(
    start_date="2024-01-01",
    interval="1d",
    limit_nums=100  # 限制处理数量
)

# 执行收集
collector.collect()
```

### 性能特点
- 支持批量下载，速度提升 5-10x
- 自动异常检测和重试
- 内存优化，支持大规模数据处理
```

### 2. 变更日志

维护 `CHANGELOG.md`：

```markdown
## [0.2.0] - 2025-01-XX

### Added
- 新增批量数据收集功能
- 添加了工具模块 `utils.py`
- 配置验证功能

### Changed
- 重构了 `main.py`，减少代码重复 60%
- 优化了数据过滤性能（10-100x 提升）

### Fixed
- 修复了数据库连接未关闭的问题
- 修复了异常检测中的除零警告

### Deprecated
- `old_function()` 将在 v0.3.0 中移除

### Removed
- 移除了不再使用的 `legacy_module.py`
```

---

## 📝 Git 提交规范

### 1. 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 2. Type 类型

- `feat`: 新功能
- `fix`: 修复 bug
- `refactor`: 重构（不改变功能）
- `perf`: 性能优化
- `docs`: 文档更新
- `test`: 添加测试
- `chore`: 构建/工具链更新

### 3. 示例

```bash
# 好的提交信息
git commit -m "feat(collector): add batch download support

- Implement batch data fetching for improved performance
- Add dynamic batch size calculation based on date range
- Include retry logic for failed batches

Closes #123"

# 简单的提交
git commit -m "fix(config): add MongoDB URL validation"

# 重构提交
git commit -m "refactor(main): extract common job status logic

- Create _prepare_job_status helper method
- Create _finalize_job_status helper method
- Reduce code duplication by 60%"
```

**❌ 避免：**
```bash
git commit -m "update"
git commit -m "fix bug"
git commit -m "改了一些东西"
```

---

## 🔒 安全实践

### 1. 敏感信息

**✅ 推荐做法：**
```python
# 使用环境变量
from config import settings

connection_string = settings.mongodb_url  # 从 .env 读取

# .env 文件（不要提交到 Git）
MONGODB_URL=mongodb://localhost:27017
ACS_CONNECTION_STRING=endpoint=...
```

**❌ 避免：**
```python
# 硬编码敏感信息
connection_string = "mongodb://user:password@host:27017"
api_key = "sk-1234567890abcdef"  # 绝对不要这样做！
```

### 2. .gitignore

确保包含：
```gitignore
# 环境变量
.env
.env.local

# 敏感数据
*.key
*.pem
secrets/

# 数据文件
data/
*.csv
*.xlsx
```

---

## 📊 代码审查清单

提交 PR 前检查：

- [ ] 所有函数都有类型提示
- [ ] 所有公共函数都有文档字符串
- [ ] 没有硬编码的敏感信息
- [ ] 遵循 DRY 原则（无重复代码）
- [ ] 使用向量化操作（避免循环）
- [ ] 适当的错误处理
- [ ] 资源正确清理（连接、文件等）
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 提交信息清晰规范

---

## 🎓 学习资源

### Python 最佳实践
- [PEP 8 - Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Effective Python](https://effectivepython.com/)

### Pandas 性能
- [Pandas Performance Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Modern Pandas](https://tomaugspurger.github.io/modern-1-intro.html)

### 测试
- [Pytest Documentation](https://docs.pytest.org/)
- [Test-Driven Development](https://testdriven.io/)

---

**最后更新**: 2025-01-XX  
**维护者**: 开发团队
