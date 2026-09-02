# NOTE-AGENT-4: Multi-Provider LLM SDKs — Package Versions, Model IDs, and Minimal Chat APIs

**Answer:**
Anthropic SDK v1.3.0 (released 2026-09-01), OpenAI SDK v3.7.0 (released 2026-09-02), and google-genai SDK v2.22.0 (released 2026-09-02) are verified on PyPI. Latest Claude model ID is `claude-opus-5` (released 2026-07-24). google-genai has no native multi-agent orchestration; Google ADK (v2.8.0, separate package) has native multi-agent primitives (Sequential, Parallel, Loop agents). Recommend hand-rolled provider-agnostic Elder interface for testability.

**Evidence:**

*PyPI Versions (verified 2026-09-02):*
- **anthropic** v1.3.0: https://pypi.org/project/anthropic/ (released 2026-09-01)
- **openai** v3.7.0: https://pypi.org/project/openai/ (released 2026-09-02)
- **google-genai** v2.22.0: https://pypi.org/project/google-genai/ (released 2026-09-02)

*Latest Claude Model IDs (Anthropic official docs, verified 2026-09-02):*
From https://platform.claude.com/docs/en/build-with-claude/working-with-messages:
- **claude-opus-5** (latest, released 2026-07-24): Full capabilities, 200K context window
- **claude-sonnet-4-5**: Previous generation
- Sampling parameters (temperature, top_p, top_k) not supported on Claude 4.7+; use structured outputs instead

*Minimal Chat API Signatures:*

**Anthropic SDK (v1.3.0):**
```python
from anthropic import Anthropic
client = Anthropic()  # reads ANTHROPIC_API_KEY from env
message = client.messages.create(
    model="claude-opus-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude"}]
)
print(message.content[0].text)
```
Source: https://platform.claude.com/docs/en/build-with-claude/working-with-messages (official docs)

**OpenAI SDK (v3.7.0):**
```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env
completion = client.chat.completions.create(
    model="gpt-4o",  # or "gpt-5" in 2026
    messages=[{"role": "user", "content": "Hello, OpenAI"}]
)
print(completion.choices[0].message.content)
```
Source: https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create (official API ref)

**Google GenAI SDK (v2.22.0):**
```python
from google import genai
client = genai.Client(api_key="sk-...")  # requires explicit api_key or GOOGLE_API_KEY env
chat = client.chats.create(model="gemini-2.5-flash")
response = chat.send_message("Hello, Google")
print(response.text)
```
Source: https://github.com/googleapis/python-genai (official repo); https://ai.google.dev/gemini-api/docs/migrate (migration guide)

*Google ADK Multi-Agent Capabilities:*
From https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/ and https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk:
- ADK v2.8.0 (verified on PyPI 2026-09-02) provides native multi-agent orchestration patterns:
  - **SequentialAgent**: Pipeline processing (task → task → task)
  - **ParallelAgent**: Concurrent sub-tasks
  - **LoopAgent**: Iterative refinement
- ADK agent orchestration is built-in, not an add-on
- google-genai SDK itself has no multi-agent primitives; supports single-model chats, function calling, tool use only

Comparison: google-genai is a lightweight LLM SDK (direct model calls); ADK is a full agentic orchestration framework (requires GOOGLE_API_KEY, runs in managed sandbox or locally with credentials).

**Caveats / limits:**

1. **Model ID versioning**: Claude model IDs are stable strings (`claude-opus-5`, not numbered versions). OpenAI uses semantic versioning in model IDs (`gpt-4o`, `gpt-5`). Google uses model names like `gemini-2.5-flash`, which may deprecate; prefer generic aliases like `gemini-flash-latest` for forward compatibility (but less reproducible).

2. **API version**: Anthropic SDK auto-manages the API version header (`anthropic-version: 2023-06-01`). OpenAI SDK similarly abstracts versioning. google-genai currently tracks the Gemini API without explicit versioning in the SDK.

3. **Key-gating**: All three SDKs require environment variables or explicit credentials:
   - Anthropic: `ANTHROPIC_API_KEY`
   - OpenAI: `OPENAI_API_KEY`
   - Google: `GOOGLE_API_KEY` (for genai) or Workload Identity (for ADK)
   These steps cannot run in-sandbox without credentials and should be clearly marked in teaching code.

4. **google-genai vs google-cloud-generativeai**: The package name is `google-genai` (correct, PyPI verified). The older `google-generativeai` is deprecated as of June 24, 2025.

5. **ADK is separate from google-genai**: If teaching multi-agent orchestration with Google models, ADK is the right choice; google-genai alone does not provide it. However, for a provider-agnostic Elder interface (SPEC-AGENT-5 scope), a hand-rolled orchestrator is clearer and testable with mocks.

**Recommendation:**

1. **Pin versions in requirements.txt:**
   ```
   anthropic==1.3.0
   openai==3.7.0
   google-genai==2.22.0
   google-adk==2.8.0  # if demonstrating ADK multi-agent primitives
   ```

2. **Prefer a hand-rolled, provider-agnostic Elder interface** for the multi-agent orchestrator in SPEC-AGENT-5:
   - Define an abstract `Elder` base class with a `respond(context)` method.
   - Implement concrete subclasses: `AnthropicElder`, `OpenAIElder`, `GoogleElder` (using google-genai).
   - Unit-test the orchestrator with scripted (fake) elders, no keys required.
   - Wire real providers behind the Elder interface, key-gated in a separate `run.py` or `providers.py` module.
   - This approach teaches multi-agent protocol design and orchestration clearly, without coupling to ADK's framework.

3. **Model ID recommendations:**
   - **Anthropic**: Use `claude-opus-5` (verified latest, 2026-07-24).
   - **OpenAI**: Use `gpt-4o` (or `gpt-5` once available in 2026).
   - **Google**: Use `gemini-2.5-flash` (or `gemini-flash-latest` if trading reproducibility for evergreen stability).

4. **For teaching**: Show minimal examples with mock/scripted responses first, then add key-gated real calls in a separate section marked "REQUIRES API KEY".

**Date checked:** 2026-09-02
