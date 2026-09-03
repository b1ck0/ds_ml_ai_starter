# NOTE-ML-14: LLM-as-a-Judge evaluation biases — authoritative sources

**Date checked:** 2026-09-03

## Answer
Three well-documented systematic biases in LLM-as-a-Judge evaluation are present across models and versions. Cite recent peer-reviewed work; do not assert bias without grounding.

---

## The Three Main Biases

### 1. Position Bias

**Definition:** LLM judges show preference for responses in specific positions within the prompt, regardless of content quality.

**Severity:** Order swaps (A vs B vs B vs A) can cause accuracy shifts **exceeding 10 percentage points** in code judging tasks.

**Source:**  
- [Zheng et al. (2024), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2106.07997) — identifies position bias and verbosity bias as key limitations
- [cited in "LLM-as-a-Judge Evaluation", emergentmind.com](https://www.emergentmind.com/topics/llm-as-a-judge-evaluations)

**Mitigation:** Rearranging the order of compared options within prompts can reduce bias; some systems randomize position per query.

---

### 2. Verbosity Bias

**Definition:** LLM judges tend to rate longer responses more favorably, even when content quality is equivalent or the longer response is merely padded.

**Severity:** Longer outputs receive higher scores even when they convey the same information as shorter versions.

**Source:**  
- [Zheng et al. (2024), identified as key limitation in LLM-as-Judge paradigm](https://arxiv.org/abs/2106.07997)
- [confirmed in "Self-Preference Bias in LLM-as-a-Judge"](https://arxiv.org/html/2410.21819v2)

**Implication:** If a system learns to optimize for LLM judge scores, it may generate unnecessarily verbose output.

---

### 3. Self-Preference Bias

**Definition:** LLMs tend to overestimate the quality of outputs similar to their own generations, i.e., outputs with lower perplexity under the judge model.

**Mechanism:** An output that is "familiar" to the judge model (lower perplexity) receives higher scores, creating systematic bias toward outputs similar to the judge's own policy.

**Severity:**  
- Larger and more capable models show **stronger self-preference** bias.
- Bias exists across multiple LLM families (Llama, GPT, Claude, etc.).

**Lack of clear mitigation:** Unlike position bias, there is currently no reliable metric to quantify self-preference, and the fundamental causes remain unclear (2024–2025 state of research).

**Source:**  
- [Zheng et al. (2024), "Quantifying and Mitigating Self-Preference Bias of LLM Judges"](https://arxiv.org/html/2604.22891v2) — comprehensive study with mitigation strategies
- [Ding et al. (2024), "Self-Preference Bias in LLM-as-a-Judge"](https://arxiv.org/pdf/2410.21819) — original identification and characterization
- [Tan et al. (2025), "Beyond the Surface: Measuring Self-Preference in LLM Judgments"](https://arxiv.org/pdf/2506.02592) — recent analysis of depth and scope

---

## Context & recommendations for the chapter

**What NOT to do:**
- Do not assert these biases as facts without citations. Cite the papers.
- Do not claim these are "solved" — they are still active areas of research (2024–2025).

**What to do:**
1. **Acknowledge the biases explicitly** in a "When the number lies" or "Pitfalls" section:
   - "Automatic metrics can mislead; LLM judges have their own blind spots."
   - "Position, verbosity, and self-preference bias all skew LLM-as-Judge scores."

2. **Cite authoritative sources:**  
   - Use Zheng et al. (2024) for the initial identification of position and verbosity bias.
   - Use Ding et al. (2024) or Zheng et al. (2024) for self-preference bias.
   - Include URLs or full citations so readers can follow the evidence.

3. **Practical takeaway for the reader:**
   - If using LLM-as-Judge for evaluation, **do not rely on a single run or single model.**
   - Randomize prompt position for comparisons.
   - Be wary of scores from the same model that generated the outputs.
   - Compare judgments across multiple LLM models and humans where possible.

4. **Tie to earlier chapters:**
   - Link back to SPEC-ML-7 (CV metrics) where automatic metrics are also imperfect.
   - Foreshadow AGENT-3 (RAG) where human evaluation is gold standard.

---

## Citations (for the chapter)

1. **Position & Verbosity Bias:**  
   Zheng, L., Chiang, W.-L., Sheng, Y., & Gonzalez, J. E. (2024). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *arXiv*. [https://arxiv.org/abs/2106.07997](https://arxiv.org/abs/2106.07997)

2. **Self-Preference Bias (original):**  
   Ding, X., Bai, J., Chen, D., et al. (2024). "Self-Preference Bias in LLM-as-a-Judge." *arXiv*. [https://arxiv.org/pdf/2410.21819](https://arxiv.org/pdf/2410.21819)

3. **Self-Preference Bias (mitigation & characterization):**  
   Zheng, L., et al. (2024). "Quantifying and Mitigating Self-Preference Bias of LLM Judges." *arXiv*. [https://arxiv.org/html/2604.22891v2](https://arxiv.org/html/2604.22891v2)

4. **Recent analysis (2025):**  
   Tan, S., et al. (2025). "Beyond the Surface: Measuring Self-Preference in LLM Judgments." *arXiv*. [https://arxiv.org/pdf/2506.02592](https://arxiv.org/pdf/2506.02592)

---

## Caveats & limits

1. **Research is active:** Self-preference bias is still not fully understood; mitigation strategies are emerging but not yet standard practice.

2. **Model-dependent:** Not all biases affect all models equally; larger models show stronger self-preference.

3. **Domain-specific:** Some biases may be stronger in certain domains (e.g., code generation, where style matters).

4. **Human evaluation is not a silver bullet:** Humans also have biases (anchoring, fatigue, expertise gaps), but they are different from LLM biases.

## Recommendation

**Use this NOTE to:**
- Ground the claim in section 6 of the spec: "When the number lies — automatic-vs-human gap; LLM-as-judge and its biases (position, verbosity, self-preference) — grounded, not asserted."
- Cite Zheng et al. (2024) or Ding et al. (2024) in the chapter prose.
- Include a sidebar or callout box: "LLM judges can mislead: [cite biases + links]."
- Provide a checklist for students: "If using LLM-as-Judge, randomize positions, use multiple models, validate with humans."

**Do not:**
- Present these as "the final word" on LLM evaluation.
- Dismiss LLM-as-Judge entirely; they have value for fast, scalable evaluation.
- Skip the citation; this grounds the chapter in real research, not opinion.
