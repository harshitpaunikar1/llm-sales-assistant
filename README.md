# LLM Sales Assistant

> **Domain:** SalesTech

## Overview

Sales teams lose hours on repetitive discovery chats and lead qualification, while buyers expect instant, accurate responses across channels. Early conversations are inconsistent, depend on each representative's script, and often miss key details needed to progress deals. Reliance on external AI services adds cost, latency, and data exposure; shaky connectivity derails responsiveness. Managers struggle enforcing process and capturing clean notes for handoff, causing rework and longer cycles. Without a scalable way to automate the first interaction while safely escalating to humans for negotiation, leads go cold, conversion drops, and customer experience suffers.

## Approach

- Mapped discovery steps and negotiation handoffs with sales ops; defined intents, entities, guardrails, success criteria using real call transcripts and outcomes
- Built on-prem language stack with local embeddings and tokenization; implemented template-driven dialog flow and intent-based routing; persisted context for downstream systems
- Optimised models for CPU-only inference; set confidence thresholds; designed human escalation for pricing/negotiation with full, traceable transcripts
- Validated in sandbox and pilot; measured intent accuracy and containment; tuned prompts, templates, fallback behaviour using error analysis
- Delivered in sprints; instrumented analytics for time-to-first-response, drop-offs, handoff quality; packaged deployment/monitoring for IT ownership

## Skills & Technologies

- Embeddings
- Tokenization
- Intent Classification
- Dialogue Flow Design
- CPU Inference Optimization
- On-Prem Deployment
- Prompt Engineering
- Python
- Hugging Face Transformers
- A/B Testing
