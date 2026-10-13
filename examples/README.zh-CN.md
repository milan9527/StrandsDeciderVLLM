# 测试输入

[English](README.md) | 简体中文

- `payment.json`：一个状态、三个独立问题，覆盖 choice / noul / score。
- `payment-corpus.json`：现有支付演示的优化输入，四场景 × 五步骤，共 20 个独立请求。
- `payment-response.json`：2026-10-13 部署的 GPU API 对 `payment.json` 返回的实际响应，调用方式见 [公网 API 文档](../docs/API.zh-CN.md)。

四个场景为到账待确认、少付补款、退款审批、商户结算。每个请求只有业务事实、问题和候选项，不执行资金交易。

20 个输入来自原项目的 `docs/results/concurrency-2026-10-13/corpus.json`，只提取 `model_input`。这些输入用于适配一致性与速度比较，不能据此声称普遍准确率。
