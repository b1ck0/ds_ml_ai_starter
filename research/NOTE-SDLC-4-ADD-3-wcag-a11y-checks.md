# NOTE-SDLC-4-ADD-3: Core WCAG / Accessibility Checks Enforced by axe-core

**Answer:**
axe-core (via axe-core-rspec / axe-core-capybara) automatically detects the following classes of
accessibility violations that matter for e-commerce checkout/catalog pages:

1. **Form labels** (WCAG 1.3.1) — inputs must have a programmatically associated `<label>`, not just
   placeholder text.
2. **Image alt text** (WCAG 1.1.1) — informative images need descriptive `alt`; decorative images get `alt=""`.
3. **Color contrast** (WCAG 1.4.3 & 1.4.11) — text 4.5:1, large text / UI components 3:1.
4. **Heading structure** (WCAG 1.3.1 & 2.4.1) — one `<h1>`, no skipped levels (h1→h3), proper hierarchy.
5. **Language attribute** (WCAG 3.1.1) — `<html lang="en">` or appropriate language code.
6. **Focus & keyboard navigation** (WCAG 2.1.1) — skip links, focus visible, tab order logical.
7. **Landmarks & semantic structure** (WCAG 1.3.1) — proper use of `<main>`, `<nav>`, `<aside>`, etc.

axe reports violations with **impact levels** (critical, serious, moderate, minor) that help prioritize
fixes and map to WCAG success criteria. Automated tools catch ~30–40% of WCAG issues; the rest require human
review for context.

## Evidence

**WCAG 2.1 (World Wide Web Consortium — official standard):**
- Source: https://www.w3.org/TR/WCAG21/
- Defines accessibility requirements for all web content; axe-core is built to this standard.

**WebAIM — Form Label Accessibility:**
- Source: https://webaim.org/articles/contrast/ (WCAG contrast explanation)
- Quote: "An input field needs an associated label. Placeholder text is not enough – it disappears as soon
  as you type and is not announced reliably."
- Form inputs must establish information and relationships programmatically via `<label for="id">` or
  wrapping the input, or via `aria-label`/`aria-labelledby`.
- Date checked: 2026-09-04

**WCAG 2.1 Image Alt Text (1.1.1):**
- Source: https://www.w3.org/TR/WCAG21/#non-text-content
- Informative images: `<img src="..." alt="Descriptive text here">`
- Decorative images: `<img src="..." alt="">`
- Functional images (links): `<a href="..."><img src="..." alt="Link destination"></a>`
- Date checked: 2026-09-04

**WCAG 2.1 Color Contrast (1.4.3 & 1.4.11):**
- Source: https://www.w3.org/TR/WCAG21/#contrast-minimum and https://webaim.org/resources/contrastchecker/
- Normal text: minimum 4.5:1 contrast ratio (text ≤ 18px or ≤ 14px bold)
- Large text: minimum 3:1 (text ≥ 18px or ≥ 14px bold)
- WCAG 2.1 Level AA (1.4.11): UI components (form borders, icons, focus indicators) also require 3:1
- Example failure: #999 placeholder on #fff = 2.85:1 (fails 4.5:1 requirement)
- Date checked: 2026-09-04

**WCAG 2.1 Heading Structure (1.3.1):**
- Source: https://www.w3.org/TR/WCAG21/#info-and-relationships
- Each page should have exactly one `<h1>` (main topic)
- No skipped levels (e.g., `<h1>` → `<h3>` without `<h2>` is a violation; confuses screen reader users)
- Heading order should match content hierarchy
- Date checked: 2026-09-04

**WCAG 2.1 Language (3.1.1):**
- Source: https://www.w3.org/TR/WCAG21/#language-of-page
- `<html lang="en">` (or other language code) must be present so screen readers know which language to use.
- Date checked: 2026-09-04

**WCAG 2.1 Focus Visible (2.4.7):**
- Source: https://www.w3.org/TR/WCAG21/#focus-visible
- All interactive elements (links, buttons, inputs) must have a visible focus indicator when using keyboard
  navigation (not removed via `outline: none` without alternative).
- Date checked: 2026-09-04

**WCAG 2.1 Skip Links (2.4.1):**
- Source: https://www.w3.org/TR/WCAG21/#bypass-blocks
- Pages with repeated content (nav, header) should have a skip link to main content, enabling keyboard
  users to bypass repetitive navigation.
- Date checked: 2026-09-04

**axe-core Rules & Impact Levels:**
- Source: https://dequeuniversity.com/rules/axe/2.1 (List of Axe 2.1 rules)
- axe impact levels:
  - **Critical:** Will cause screen reader users or keyboard-only users to be unable to access content
  - **Serious:** Will significantly degrade experience for some users
  - **Moderate:** May be difficult for some users to access
  - **Minor:** May be slightly annoying to access
- Each axe violation includes: `id`, `description`, `help`, `helpUrl`, `nodes`, `impact`, `tags` (e.g.,
  `wcag21a`, `wcag21aa`, `form`, `images`)
- Date checked: 2026-09-04

**axe-core-rspec / axe-core-capybara matcher:**
- Source: https://github.com/dequelabs/axe-core-gems
- RSpec matcher syntax:
  ```ruby
  expect(page).to be_axe_clean  # Fail on any violation
  expect(page).to be_axe_clean.checking(:label)  # Check only specific rules
  expect(page).to be_axe_clean.according_to(:wcag21a)  # Check only WCAG 2.1 Level A
  ```
- Can filter by impact level and other criteria (see gem documentation for full API).
- Date checked: 2026-09-04

## Caveats / Limits

- **Automated detection limits:** Tools like axe catch ~30–40% of WCAG issues (missing labels, alt text,
  contrast, heading order, ARIA misuse, keyboard traps). The remaining 60–70% require human review (e.g.,
  whether alt text is meaningful, logical keyboard flow, readability of instructions).
- **Context-dependent:** Some rules need human judgment (e.g., is this image truly decorative, or should
  it have alt text?).
- **Not all violations fail CI:** axe can be configured to warn on `moderate`/`minor` violations without
  failing the build, allowing gradual adoption on legacy code.
- **False negatives:** Pages may have WCAG violations that axe doesn't catch (e.g., overly complex forms
  that screen reader users find confusing despite programmatic labels).
- **Version-specific:** axe-core updates its rule set regularly; newer versions may catch violations older
  versions missed.

## Recommendation

- **For the frontend-QA agent checklist:** Include categories (form labels, images, contrast, headings,
  language, focus, landmarks) as a guide; the agent should run the `be_axe_clean` matcher and report any
  violations with their impact level and WCAG criterion.
- **For FEATURE-3 walkthrough:** Introduce a realistic violation (e.g., a checkout form input without a
  `<label>`, or skipped heading level), show the axe violation report (JSON with id, description, impact),
  and walk through fixing it.
- **For the spec suite:** Write `spec/system/accessibility_spec.rb` with examples:
  ```ruby
  describe 'Product catalog', type: :system, js: true do
    it 'has no accessibility violations' do
      visit products_path
      expect(page).to be_axe_clean.according_to(:wcag21aa)
    end
  end
  ```
- **For expectations:** Remember that axe is a complement to, not a replacement for, manual accessibility
  testing and user testing with people who use assistive technology.
