# AGENTS.md — Study & Knowledge Rules

## Purpose
This repository is a **living knowledge base** of AI/ML concepts learned through deep-dive Q&A sessions. Every file here represents a topic we explored interactively — question by question, concept by concept.

## How We Learn
1. **Start with a question** — pick a topic, ask a specific question
2. **Deep dive iteratively** — explore layer by layer, question by question
3. **Use analogies** — explain concepts with real-world analogies before diving into math/code
4. **Anchor to code** — after conceptual understanding, link to actual code (llama.cpp source, CUDA kernels, etc.)
5. **Document the journey** — save the Q&A as a structured markdown file for future reference

## Q&A Format

When doing a deep-dive session, structure the output as:

```markdown
## Topic Name

### Question 1: [conceptual question]
[Clear, concise answer with analogy]

### Question 2: [follow-up question]
[Deeper explanation, may reference code]

...
```

### File Naming
- `topic-name.md` — one file per topic
- If a topic grows large, split into `topic-name-part1.md`, `topic-name-part2.md`
- Use lowercase with hyphens

### Content Guidelines
- **Analogies first** — always lead with a real-world analogy before technical details
- **Code references** — link to specific files/lines in llama.cpp or other relevant codebases
- **Visual ASCII diagrams** — use simple text diagrams to illustrate flows
- **Summaries** — include a summary/cheatsheet at the end of each file
- **Progressive complexity** — start simple, get detailed

## Topics Covered So Far

| File | Topics |
|---|---|
| `ai-basics.md` | Token, forward pass, hidden state, lm_head, autoregressive, prefill vs decode, speculative decoding |
| `mtp-basics.md` | MTP architecture, hidden state vs output, verify flow, 2 lm_head, efficiency |
| `llamacpp-code.md` | server.cpp structure, MTP state machine, build setup, CUDA toolchain |

## Maintenance
- Update files when understanding deepens
- Add cross-references between related topics
- Review and refine analogies for clarity
- Commit with descriptive messages: `study: add [topic] Q&A`

## Quick Reference — Key Concepts

### Token
Potongan kata yang model baca sebagai angka. Tokenizer ubah teks → token IDs.

### Forward Pass
Satu kali proses data dari input ke output melalui semua layer neural network.

### Hidden State
Vektor float yang merupakan "pemikiran abstrak" model di suatu layer. Bukan probabilitas.

### lm_head
Fungsi yang mengubah hidden state jadi probabilitas untuk setiap token di vocabulary.

### Autoregressive
Generate token satu per satu — setiap langkah cuma nebak 1 token ke depan, terus berulang.

### Speculative Decoding
Teknik percepat inference: model kecil generate draft → model besar verify dalam batch.
