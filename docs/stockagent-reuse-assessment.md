# StockAgent reuse assessment (for M4)

Source read: `~/Desktop/Github Projects/Stockagent-main/` — `agent.py` (541 lines),
`main.py` (208), `secretary.py` (241), `stock.py` (31), `record.py` (141), `util.py` (50),
`prompt/agent_prompt.py` (290), `log/custom_logger.py`. `PromptCoder/` is a vendored copy of
`procoder` (prompt-composition helpers: `NamedBlock`, `NamedVariable`, `Collection`,
`format_prompt`). Nothing has been copied yet; that is M4.

## 1. What the code does

**Simulation loop (`main.py:75-201`).** Two synthetic stocks A and B (`main.py:78-80`, initial
prices from `util.py:14-15`). For each of `util.TOTAL_DATE` days: clear order books
(`main.py:101-106`), loan repayment / interest / bankruptcy (`main.py:114-129`), two scripted
macro events flip loan rates (`main.py:132-137`), each agent decides on a loan (`main.py:141-143`),
then for each of `util.TOTAL_SESSION` sessions agents are polled in random order and each returns
one limit order (`main.py:145-164`) which is matched immediately (`handle_action`); after the
session the stock price becomes the last trade price (`stock.py:21-26`); end of day agents post a
"forum" message and a next-day estimate (`main.py:172-189`).

**Order matching (`main.py:20-72`, `handle_action`).** A toy continuous book: an incoming buy is
matched against resting sells only at *exactly equal* price (`main.py:25`), partial fills reduce
the resting order, remainder rests (`main.py:46/69`). No price-time priority beyond list order, no
crossing (a buy at 31 does not hit a sell at 30). **Bug:** `Agent.buy_stock(stock_name, price,
amount)` (`agent.py:435`) is called as `buy_stock(stock.name, close_amount, action["price"])`
(`main.py:28,30,53,54`) — price and amount are swapped, so cash/position accounting in the
original simulator is wrong whenever `price != amount`. Do not inherit this.

**Portfolio state (`agent.py:52-67, 241-254, 435-510`).** Per agent: `cash`, `stock_a_amount`,
`stock_b_amount`, `loans` list, `is_bankrupt/quit`; `get_total_proper()` = mark-to-market equity;
`buy_stock/sell_stock` guard cash and holdings; `loan_repayment/interest_payment/bankrupt_process`
implement leverage with forced liquidation. Hard-coded to exactly two instruments named A/B.

**LLM providers (`agent.py:69-239`).** `run_api()` dispatches on model-name substring
(`agent.py:69-78`): OpenAI Chat Completions with the SDK (`agent.py:134-161`), Gemini via raw REST
`generateContent` (`agent.py:80-131`), Anthropic Messages SDK with a fixed system prompt
(`agent.py:163-188`), Ollama `/api/generate` on localhost with chat history flattened into the
prompt (`agent.py:190-239`). Each keeps a per-agent `chat_history` that is cleared daily
(`main.py:115`), retries twice, and returns `""` on failure (callers treat `""` as "no action").
Temperature defaults to **1** (`agent.py:69`). Keys are read from `util.py:4-6` module constants —
**`util.py:5` contains a live-looking Google API key committed to the repo; rotate it and move to
`.env`.** `secretary.py:21-32` has a second, dead `run_api` with `api_key=""`.

**Output validation (`secretary.py:35-241`).** `Secretary.check_loan / check_action /
check_estimate` extract the single `{...}` from the response (`secretary.py:116-122`), repair
unquoted keys (`repair_json`, `secretary.py:8-18`), parse JSON, then check semantics: action_type in
{buy,sell,no}, stock in {A,B}, integer amount, price > 0, buy notional <= cash, sell qty <= holding
(`secretary.py:136-195`). On failure the agent is re-prompted with the reason up to 3 times
(`agent.py:392-406`). Note the retry path in `plan_loan` calls `check_loan(date, resp)` with the
wrong arguments (`agent.py:322`).

**Prompts (`prompt/agent_prompt.py`).** procoder blocks: `BACKGROUND_PROMPT` (l.12),
`LASTDAY_FORUM_AND_STOCK_PROMPT` (l.30), loan prompts (l.49-111), `DECIDE_BUY_STOCK_PROMPT`
(l.127-140, the per-session trading prompt), `BUY_STOCK_RETRY_PROMPT` (l.153), fictional
fundamentals `FIRST_DAY_FINANCIAL_REPORT` / `FIRST_DAY_BACKGROUND_KNOWLEDGE` (l.175-215),
`SEASONAL_FINANCIAL_REPORT` (l.226), forum post and next-day estimate prompts (l.242-291).
Composition happens in `Agent.plan_stock` (`agent.py:339-382`) via `Collection(...)` +
`format_prompt(prompt, inputs)`.

**Records (`record.py`).** Every trade/price/agent-session row is appended by
read-whole-xlsx / concat / write-whole-xlsx (`record.py:15-27` etc.) — O(n^2) and Excel-only.

## 2. How prompts currently receive price data (matters for the no-lookahead audit)

The LLM sees **only scalars, no history**:

- `DECIDE_BUY_STOCK_PROMPT` gets `stock_a_price`, `stock_b_price` = `stock.get_price()` at call
  time (`agent.py:348-349, 364-365, 377-378`) — the last trade price of the *previous* session
  (`stock.py:24`), plus `stock_a_deals` / `stock_b_deals`, the *current* open order book dicts
  (`agent.py:350-351`, passed from `main.py:155`). The book is what other agents have already
  submitted this session, so ordering within a session leaks intent between agents (by design in
  the paper — but irrelevant to a single-agent replay).
- The loan prompt gets yesterday's close-of-day prices via `LASTDAY_FORUM_AND_STOCK_PROMPT`
  (`agent.py:293-294`).
- Portfolio state (`stock_a`, `stock_b`, `cash`, `debt`, `max_loan`) is passed alongside
  (`agent.py:343-354`).
- No return series, no OHLCV, no dates in calendar form (`date` is an integer day counter,
  `agent.py:344`), no ticker names — the model cannot recognise real assets or recall real history.

Consequences for the backtester: (a) the prompt builder must be rewritten to accept a
point-in-time window of real bars (e.g. last N Adj Close / returns) — nothing here does that;
(b) because prices are injected as explicit template inputs at one call site per prompt, the
audit is easy: assert every value in `inputs` derives from `prices.loc[:t]`; (c) using real ticker
symbols and calendar dates in prompts re-introduces the leakage the paper avoided (the LLM may
"remember" 2022) — decide whether to anonymise tickers (`Stock 1..15`) and shift/omit dates; this
is an experiment knob worth reporting either way.

## 3. Reusable for a historical-replay backtester

| Piece | Where | Verdict |
|---|---|---|
| Provider wrappers (OpenAI / Gemini REST / Anthropic / Ollama) with retry | `agent.py:80-239` | **Reuse, refactored** into a `providers.py` with one `complete(prompt, model, temperature) -> str` per provider, keys from `.env`, temperature=0 default, and a `(sha256(prompt, model, temperature) -> response)` disk cache in front (the $0-replay rule). Drop per-agent chat history: replay decisions should be stateless per bar. |
| JSON extraction + repair + semantic checks + bounded retry-with-reason | `secretary.py:8-18, 113-199`; `agent.py:385-406` | **Reuse the pattern**, generalise to N tickers and to a decision schema like `{"orders":[{"ticker":..,"side":..,"weight"|"qty":..}]}` or target weights. Keep the "3 retries then no-op" fallback and *log* fallbacks — the fallback rate is a result. |
| procoder prompt composition (`NamedBlock`, `Collection`, `format_prompt`) | `prompt/agent_prompt.py`, `PromptCoder/` | **Reuse the structure** (background / market data / portfolio / instruction / strict JSON output). Fine to keep procoder as a dep or replace with f-strings — it is a light convenience. |
| `Secretary` role (validate LLM output before it touches the portfolio) | `secretary.py` | **Reuse the idea**; the engine (M2) owns cash/holding constraints, the validator only shapes JSON. |
| Portfolio accounting concepts (cash, positions, mark-to-market equity, guarded buy/sell) | `agent.py:241-248, 435-465` | **Concept only** — rewrite in M2 as `Portfolio(cash, positions: dict[ticker, qty])` with cost model. Two-instrument hard-coding makes the code itself unusable. |
| Agent "character" (Conservative/Aggressive/…) and random initial endowment | `agent.py:57-59, 19-34` | Optional knob; fixed cash start (e.g. $1M) is the right default for comparability. |

## 4. Must be replaced / dropped

- **Session-based synthetic pricing**: `Stock.update_price` (price = last trade,
  `stock.py:21-26`), the order book and `handle_action` matcher (`main.py:20-72`), multiple
  sessions per day, agent-vs-agent matching. Replaced by historical replay: decisions on bar t,
  fills at bar t+1 open against the snapshot (M2 engine).
- **Event phases**: loans/interest/bankruptcy (`agent.py:467-510`, `main.py:114-129`), scripted
  macro events (`util.py:44-51`, `main.py:132-137`), quarterly fictional financial reports
  (`util.py:33-40`, `stock.py:12-16`), forum posts and next-day estimates (`agent.py:512-539`).
  None apply to a single-agent replay on real data; leverage is out of v1 scope.
- **Excel append records** (`record.py`) -> parquet/CSV of the fills and equity curve from M2.
- **`util.py` config/keys** -> `.env` + a small config dataclass. Rotate the leaked Google key.
- **Argument-order bug** in the buy/sell calls (`main.py:28,30,53,54`) — do not port.

## 5. Recommended M4 shape (no code yet)

`sab/strategies/llm_agent.py`: build prompt from (`prices.loc[:t]` tail window, portfolio
snapshot, instruction) -> `providers.complete()` through the cache -> validate JSON -> return
target weights/orders to the engine. Log every (prompt hash, model, response, parsed decision,
fallback?) so decision-consistency (same prompt twice, temp 0 vs 1) is a query, not a rerun.
