# ZMQ 无数据诊断日志 — 2026-05-18 06:38 UTC (北京 14:38)

## 1. 我是否改过 config / sql_writer？
**没有。**
- `config.json` 最后修改时间：**May 11 08:56**（今天未动）
- `sql_writer_dce.py` 最后修改时间：**May 11 08:56**（今天未动）
- 我今天改过的文件仅限以下（均与 ZMQ 无关）：
  - `data_service.py` (分钟 gap filling、time import bug 修复)
  - `data_function.py` (weighted_s 回退到 shift)
  - `strategies.py` (_save_min_data 精简、字段新增)
  - `monitor_web.py` (独立监控页面)
  - `orchestrator.py` (auto-archive)

## 2. sql_writer 当前状态
- **PID**: `55286`
- **状态**: `S (sleeping)` — 正常等待 ZMQ 数据
- **stdout/stderr**: 挂在 `/dev/pts/6`（你之前的终端窗口）
- **数据库**: `/home/strategy_PAMY_dev/tick_data.db` 已打开，有 wal 文件，但**没有任何合约表被创建**
  - 说明从启动到现在**零条 tick 被收到**

## 3. 配置确认
```json
合约: ["a2605","b2605","c2605","cs2605","m2605","y2605","p2605","lh2605"]
ZMQ:  tcp://172.17.0.6:7779
```
与你所说的模拟数据源配置一致。

## 4. 网络/端口测试
- `python zmq.SUB → connect(172.17.0.6:7779)`：**连接成功**
- 但 `socket.recv()` 在 2s 超时后返回 **无数据**
- **docker 无容器运行**（`docker ps` 无输出）
- **无进程监听 7779**（`netstat` 未找到）

## 5. 结论
**模拟数据源进程本身没有启动或已崩溃。**
- ZMQ 端口能连上，通常意味着对端有进程绑定了 `*:7779`（或者网络层有端口映射），但没有任何数据推送。
- 由于 docker 未运行，该数据源可能是宿主机上的一个独立 Python/Java 进程，目前已退出。

## 6. 建议下一步（请在新窗口执行）

### A. 检查模拟数据源进程是否还活着
```bash
# 在你的数据源服务器上执行（172.17.0.6 所在机器）
ps aux | grep -E "7779|pub|simul|行情|zmq"
netstat -tlnp | grep 7779
```

### B. 如果进程已死，直接重启模拟数据源
```bash
# 示例（请替换为真实启动命令）
cd <模拟数据源目录>
python sim_publisher.py   # 或 nohup python ... &
```

### C. 重启 sql_writer 并持久化日志
当前 sql_writer 挂在 pts/6，若你关闭旧窗口会丢失输出。建议：
```bash
cd /home/strategy_PAMY_dev
# 先停旧进程
kill 55286
# 重新启动并写日志
python sql_writer_dce.py > logs/sql_writer_$(date +%Y%m%d).log 2>&1 &
```

### D. 快速验证数据是否恢复
```bash
# 等待 10 秒后检查
cd /home/strategy_PAMY_dev
sqlite3 tick_data.db "SELECT instrument, COUNT(*) FROM tick_data_a2605;"
```

---
*本文件由 agent 自动生成于 2026-05-18，供新窗口速查。*
