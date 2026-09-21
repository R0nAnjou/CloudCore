# OurBot — 《未来战争》参赛程序 v0.1

基于官方 Demo `CoreGeek` 骨架重构的第一版参赛程序。

## 快速开始

```bash
# 判题器方式启动(比赛环境)
bash run.sh 8080

# 本地验证(冒烟测试: 模拟 1300 回合)
python tests/smoke_test.py
```

需要 Python ≥ 3.11, 零第三方依赖(纯标准库)。

## 目录结构

```
OurBot/
  main.py               入口: python main.py <port>
  run.sh                判题器拉起脚本
  src/agent/
    server.py           HTTP 服务: 4s deadline 守卫 + 异常兜底(重发上回合指令)
    protocol.py         协议层: 请求解析 + 全部 12 种动作码构造器
    grid.py             8 向 A* 寻路 + Bresenham 弹道
    memory.py           跨回合记忆: 建造区试错/新闻日志/SOP 知识库/宝藏线索
    brain.py            主决策编排: 白天分派经济+任务, 夜晚分派防御
    economy.py          经济: 性价比选矿/批量卖矿/建造/升级券/修墙
    defense.py          夜战: 三炮火力分配(火箭群伤/电磁穿透/加特林锥形)+集火超杀转移
    tasks.py            任务引擎: 自进化 SOP 流水线 + LLM 异步 + 宝藏召唤
  tests/
    smoke_test.py       冒烟测试
```

## 已实现能力(对照任务书)

| 模块 | 能力 | 对应得分 |
|---|---|---|
| server.py | 5s/10s 超时保护、异常兜底重发、HTTP/1.1 | 避免异常淘汰 |
| economy.py | 矿价性价比选矿、新闻停产预期联动、批量卖矿、围墙圈预留入口、基地升级券、修复包 | 经济 + 生存分 |
| defense.py | 火箭选最密落点、电磁穿透弹道、加特林 90° 锥形多目标、伤害预扣集火 | 击杀分 |
| tasks.py | 任务点领任务→executeCmd 探索→SOP 沉淀复用→submitAnswer; LLM 每日 3 次限额管理; 民间传闻宝藏召唤+返回码纠错 | 任务分(上限最高) |
| memory.py | 建造失败位置学习(不再重复违规)、每日新闻存档、SOP 知识库 | 全局 |

## 首版已知局限(下一步计划)

1. `tasks.py` 沙盒探索脚本是固定 5 步模板, 需按真实任务文档自适应;
2. 建造区(蓝/黄)数据接口未下发, 目前靠 `lastRoundRoleActionResults` 试错学习;
3. 宝藏解谜依赖 LLM 推断, 每日 3 次额度下的提示词还需打磨;
4. 未实现召唤令骚扰/眩晕法宝反制等对抗性玩法;
5. 夜间角色低血量撤退/喝药逻辑未接入。

## 本地联调

无官方模拟器时, 可用 `tests/smoke_test.py` 的样例数据直接调 `decide()`;
真实联调需判题器或自行 mock: `POST http://localhost:<port>/` 提交 `docs/request.txt` 内容。
