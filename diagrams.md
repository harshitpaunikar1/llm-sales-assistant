# LLM Sales Assistant Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## Conversation flow diagram

```mermaid
flowchart TD
    N1["Step 1\nMapped discovery steps and negotiation handoffs with sales ops; defined intents, e"]
    N2["Step 2\nBuilt on-prem language stack with local embeddings and tokenization; implemented t"]
    N1 --> N2
    N3["Step 3\nOptimised models for CPU-only inference; set confidence thresholds; designed human"]
    N2 --> N3
    N4["Step 4\nValidated in sandbox and pilot; measured intent accuracy and containment; tuned pr"]
    N3 --> N4
    N5["Step 5\nDelivered in sprints; instrumented analytics for time-to-first-response, drop-offs"]
    N4 --> N5
```

## Intent classification accuracy

```mermaid
flowchart LR
    N1["Inputs\nHistorical support chats and FAQ content"]
    N2["Decision Layer\nIntent classification accuracy"]
    N1 --> N2
    N3["User Surface\nOperator-facing UI or dashboard surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
