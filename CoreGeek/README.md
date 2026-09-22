# CoreGeek — 《未来战争》参赛程序

基于官方 Demo `CoreGeek` 骨架重构的第一版参赛程序。

## 快速开始

```bash
# 判题器方式启动(比赛环境)
bash run.sh 8080

# 任务状态机回归测试
python -m unittest discover -s tests -p 'test_*.py' -v
```

比赛平台会固定执行 `python main3.py <port>`，因此提交包根目录必须直接包含
`main3.py`。若平台要求上传 tar.gz，压缩包内应是 `CoreGeek/main3.py`。

需要 Python ≥ 3.11, 零第三方依赖(纯标准库)。

## 目录结构

```
CoreGeek/
  main3.py              比赛平台入口: python main3.py <port>
  main.py               服务启动实现
  run.sh                判题器拉起脚本
  src/agent/
    server.py           HTTP 服务: 4s deadline 守卫 + 异常时返回空指令
    protocol.py         协议层: 请求解析 + 全部 12 种动作码构造器
    grid.py             8 向 A* 寻路（含交互目标邻接寻路）+ Bresenham 弹道
    memory.py           跨回合记忆: 建造区试错/新闻日志/SOP 知识库/宝藏线索
    brain.py            主决策编排: 白天分派经济+任务, 夜晚分派防御
    economy.py          经济: 性价比选矿/批量卖矿/建造/升级券/修墙
    defense.py          夜战: 三炮火力分配(火箭群伤/电磁穿透/加特林锥形)+集火超杀转移
    tasks.py            任务引擎: 自进化 SOP 流水线 + LLM 异步 + 宝藏召唤
  tests/
    test_task_agent.py  沙盒命令/答案状态机回归测试
```

## 已实现能力(对照任务书)

| 模块 | 能力 | 对应得分 |
|---|---|---|
| server.py | 5s/10s 超时保护、空指令安全兜底、HTTP/1.1 | 避免异常淘汰 |
| economy.py | 批量采卖、新闻停产预期联动、围墙圈预留入口、基地/武器/围墙升级、修复包 | 经济 + 生存分 |
| defense.py | 火箭选最密落点、电磁穿透弹道、加特林 90° 锥形多目标、伤害预扣集火 | 击杀分 |
| tasks.py | 领取任务→读取文档→LLM 选择查询命令→executeCmd 回传真实结果→提交答案；动态任务用品与宝藏时间/返回码处理 | 任务分(上限最高) |
| memory.py | 建造失败位置学习(不再重复违规)、每日新闻存档、SOP 知识库 | 全局 |

## 首版已知局限(下一步计划)

1. 自进化任务题型并未公开完整样本；当前使用受限的通用命令循环，尚需实战验证不同题型及其沙盒 API；
2. 建造区(蓝/黄)数据接口未下发, 目前靠 `lastRoundRoleActionResults` 试错学习;
3. 宝藏解谜仍依赖 LLM 从自然语言传闻提取位置、物品和开放时间，正式传闻格式需要实战校准；
4. 未实现召唤令骚扰/眩晕法宝反制等对抗性玩法;
5. 夜间角色低血量撤退/喝药逻辑未接入。

## 本地联调

无官方模拟器时，可运行上面的任务状态机测试；真实联调需判题器或自行
mock：`POST http://localhost:<port>/` 提交仓库 `docs/request.txt` 内容。
