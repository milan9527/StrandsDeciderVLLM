# Test inputs

English | [简体中文](README.zh-CN.md)

- `payment.json`: one state and three independent questions, covering choice / noul / score.
- `payment-corpus.json`: optimized inputs from the payment demo, with four scenarios × five steps, for 20 independent requests.
- `payment-response.json`: actual response to `payment.json` from the deployed GPU API on 2026-10-13. See [public API usage](../docs/API.md).

The scenarios are payment awaiting confirmations, underpayment recovery, refund approval, and merchant settlement. Each request contains business facts, a question, and candidate options. These inputs do not execute financial transactions.

The 20 inputs were extracted from `model_input` in the original project's `docs/results/concurrency-2026-10-13/corpus.json`. They measure adaptation consistency and speed; they do not establish general decision accuracy.
