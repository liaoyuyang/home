# Terra Feed 接收端调试方案

> 发送端当前状态：已启动，ZMQ PUB 绑定在 `tcp://*:7779`，进程运行正常。
> 发送端环境：Docker 容器内（容器 IP: `172.17.0.7`）

---

## 一、确认接收端连接的目标地址

请先确认你的订阅代码里填写的连接地址。以下是几种常见情况：

| 场景 | 应连接的地址 | 说明 |
|------|-------------|------|
| 接收端与发送端在同一 Docker 网络 | `tcp://172.17.0.7:7779` | 容器间互通可直接用容器 IP |
| 接收端在宿主机/外部机器，有端口映射 | `tcp://<宿主机IP>:7779` | 需确认宿主机做了 `-p 7779:7779` |
| 接收端在宿主机/外部机器，host 网络模式 | `tcp://<宿主机IP>:7779` | 容器使用 `--network host` |

**请先检查你的订阅配置（代码或配置文件）里写的是哪个 IP。**

---

## 二、基础网络连通性检查

在接收端机器上执行以下命令，替换 `<目标IP>` 为实际要连接的 IP：

```bash
# 1. 测试 TCP 端口是否通
nc -vz <目标IP> 7779

# 如果 nc 没有，用 telnet
telnet <目标IP> 7779

# 如果都没有，用 Python 简单测试
python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(('<目标IP>', 7779))
    print('TCP 连接成功')
except Exception as e:
    print('连接失败:', e)
finally:
    s.close()
"
```

**预期结果：**
- 如果显示 `Connected` / `TCP 连接成功` → 网络层是通的，问题在 ZMQ 订阅层。
- 如果显示 `Connection refused` / `Connection timed out` → 网络不通或端口未暴露，看第三节。

---

## 三、如果网络不通：排查 Docker/防火墙

### 3.1 检查发送端容器是否有端口映射
在**发送端的宿主机**上执行：

```bash
docker inspect e8309b1a3acd | grep -A 20 '"PortBindings"'
docker inspect e8309b1a3acd | grep '"NetworkMode"'
```

- 如果 `NetworkMode` 是 `host` → 外部机器应连宿主机 IP。
- 如果 `NetworkMode` 是 `bridge` 且没有 `PortBindings` → **这就是问题！** 外部机器无法直接访问容器内部端口。需要在宿主机上重启容器并加端口映射：
  ```bash
  docker run -p 7779:7779 ...
  ```

### 3.2 检查宿主机防火墙
在**发送端宿主机**上：

```bash
# CentOS/RHEL
sudo firewall-cmd --list-ports | grep 7779
sudo firewall-cmd --add-port=7779/tcp --permanent
sudo firewall-cmd --reload

# 或者 iptables
sudo iptables -L -n | grep 7779
```

---

## 四、ZMQ 订阅端代码调试

如果 TCP 端口能通，但收不到数据，请用以下最小化代码测试：

```python
import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.SUB)

# 填写实际 IP
socket.connect("tcp://<目标IP>:7779")

# 订阅所有主题（空字符串表示不过滤）
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("已连接，等待数据...")

# 设置接收超时，方便观察
socket.setsockopt(zmq.RCVTIMEO, 5000)

try:
    for i in range(20):
        try:
            msg = socket.recv()
            print(f"[{i}] 收到数据，长度: {len(msg)}")
        except zmq.Again:
            print(f"[{i}] 超时，未收到数据")
except KeyboardInterrupt:
    pass
finally:
    socket.close()
    context.term()
```

**运行后观察：**
- 如果持续打印 `超时，未收到数据` → 发送端确实没有推送数据（可能是因为非交易时间，见第五节）。
- 如果打印 `收到数据` → 网络和订阅都正常，检查你的业务代码是否过滤了 topic。

---

## 五、确认发送端是否在推送数据

当前发送端配置 `instrument_class=CF`（棉花），**棉花当前非交易时间**（交易日 13:30-15:00、21:00-23:00）。如果现在是白天收盘后，即使网络和订阅都正常，也不会收到实时 tick 数据。

### 5.1 检查发送端日志是否有行情推送
在发送端执行：

```bash
# 找到最新日志
cd /home/liaoyuyang/terra_feed/bin
ls -lt .2026*.log | head -n 3

# 检查是否有行情推送关键字（tick、depth、market、push）
grep -i "tick\|depth\|marketdata\|push\|send" <最新日志文件> | tail -n 20
```

如果日志里只有 instrument 加载信息，没有行情推送 → **没有数据可收**。

### 5.2 临时验证：用回测端口测试
发送端 `bt_trade` 还开了 `tcp://*:8882`（回测数据发布）。如果接收端在内部网络，可以连 8882 验证网络和 ZMQ 订阅是否正常：

```python
socket.connect("tcp://<目标IP>:8882")
```

---

## 六、信息收集清单

如果以上步骤仍未解决，请把以下信息发回给发送端排查：

1. 接收端机器的 IP 地址：_________
2. 订阅代码里填写的连接地址：_________
3. `nc -vz <目标IP> 7779` 的输出结果：_________
4. 接收端是否在 Docker 容器内？是 / 否
5. 运行最小化订阅代码后的输出（贴前 20 行）：_________
6. 发送端当前时间是否在交易时间内？是 / 否

---

## 七、快速决策树

```
nc/telnet 端口通吗？
├── 不通
│   ├── 接收端与发送端同 Docker 网络？→ 检查是否连的容器 IP (172.17.0.7)
│   └── 接收端在宿主机/外部？→ 检查宿主机是否做了 -p 7779:7779 端口映射
│
└── 通
    └── 最小化 ZMQ 代码能收到数据吗？
        ├── 能 → 你的业务代码有 topic 过滤或解析问题
        └── 不能 → 发送端当前无行情（非交易时间或 CTP 未登录成功）
            └── 检查发送端日志是否有行情推送记录
```
