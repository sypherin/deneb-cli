# Engine contract: one brain, many faces

Deneb answers two questions: which model to run on this box (`recommend`), and how to set it up
(`setup`). It has more than one face that asks them: the `deneb` CLI and the Deneb MCP server
(`deneb-mcp/`).

The rule that keeps them from disagreeing:

> Selection logic lives in the engine that owns the question. Every face calls that engine. No
> face re-implements a ranking, a fit check or a selection of its own.

Where each question is owned:

- Ranking the model catalog for a use-case: `deneb/recommend.py`.
- Building the ordered, platform-correct setup steps: `deneb/setup_advisor.py`.
- Picking the measured config this box can actually reproduce: `deneb/measured.py`.
- Ranking measured inference configs is a different question, and it is TokenMark's, not Deneb's.
  The deneb side reads it over the network through `deneb/tokenmark.py`, which calls the public
  `/api/recommend` on tokenmark.app and keeps no local copy of that ranking.

`deneb/api.py` is the shared orchestration for the two CLI/MCP questions. It reads the box once,
calls the engine modules above, and returns data. The CLI formats that data for a terminal; the
MCP formats the same data for an agent. Neither formats by deciding; they format by reading
attributes off what `api` returned.

What this buys: the MCP cannot drift from the CLI, because there is no second ranking to drift.
A test in `tests/test_api.py` pins that `api.recommend` returns the engine's order verbatim, so
if a future edit grows a second brain inside a face, it goes red.

What breaks the contract (do not do these):

- Re-ranking, re-scoring or re-filtering inside a formatter (`cli.py`, `deneb-mcp/`'s `shape.py`).
  If you are sorting or comparing picks in a face, the logic is in the wrong place.
- A face that answers from its own copy of the catalog or the weights instead of calling the
  engine.
- Adding a selection knob to one face only. A new knob goes on the engine function, and every
  face passes it through.
