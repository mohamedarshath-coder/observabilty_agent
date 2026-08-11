/**
 * Responsible AI Policy Agent
 * Provider: OpenAI — claude-sonnet-4-6 (temperature: 0.1)
 *
 * Evaluates the response against active org policies.
 * Returns per-policy pass/fail with rationale.
 */

const TOXIC_KEYWORDS = [
  'kill', 'hate', 'stupid', 'idiot', 'dumb', 'moron', 'racist', 'sexist',
  'bomb', 'weapon', 'attack', 'violence', 'murder', 'assault',
];

const BIAS_INDICATORS = [
  /\b(always|never|all|none)\b.{0,30}\b(women|men|blacks|whites|asians|latinos|muslims|christians|jews)\b/i,
  /\b(they|those people)\b.{0,20}\b(always|never|tend to|are all)\b/i,
];

const JAILBREAK_PATTERNS = [
  /ignore.{0,30}(previous|prior|above|instructions|system)/i,
  /pretend (you are|you're|to be)/i,
  /act as (an?|your) (evil|unrestricted|jailbroken|DAN)/i,
  /do anything now/i,
  /bypass.{0,20}(safety|filter|restriction|guardrail)/i,
];

const REGULATED_DOMAINS = [
  { pattern: /\b(diagnos|prescri|medic(al|ation)|treatment|symptom|disease)\b/i, domain: 'medical' },
  { pattern: /\b(legal advice|sue|lawsuit|attorney|court ruling)\b/i, domain: 'legal' },
  { pattern: /\b(invest(ment)?|buy (stock|crypto|shares)|financial advice|portfolio)\b/i, domain: 'financial' },
];

function evaluatePolicy(policy, text) {
  const { category, name, rule_config } = policy;
  const textLower = text.toLowerCase();

  let passed = true;
  let rationale = 'No issues detected.';

  if (category === 'pii') {
    const piiPatterns = [
      /\b\d{3}-\d{2}-\d{4}\b/,
      /\b4[0-9]{12}(?:[0-9]{3})?\b/,
    ];
    const found = piiPatterns.some(p => p.test(text));
    passed = !found;
    rationale = found ? 'PII pattern detected in response.' : 'No PII detected.';
  } else if (category === 'toxicity') {
    const found = TOXIC_KEYWORDS.filter(kw => textLower.includes(kw));
    passed = found.length === 0;
    rationale = found.length > 0 ? `Toxic keywords found: ${found.join(', ')}.` : 'No toxicity detected.';
  } else if (category === 'bias') {
    const found = BIAS_INDICATORS.some(p => p.test(text));
    passed = !found;
    rationale = found ? 'Potential bias language detected.' : 'No bias indicators found.';
  } else if (category === 'jailbreak') {
    const found = JAILBREAK_PATTERNS.some(p => p.test(text));
    passed = !found;
    rationale = found ? 'Jailbreak attempt pattern detected.' : 'No jailbreak patterns found.';
  } else if (category === 'custom') {
    // Check custom keyword rules
    const rules = rule_config?.rules || [];
    const triggered = rules.filter(r => textLower.includes(r.toLowerCase()));
    passed = triggered.length === 0;
    rationale = triggered.length > 0
      ? `Custom rule(s) triggered: ${triggered.join(', ')}.`
      : 'All custom rules passed.';
  }

  // Check regulated domain guidance
  if (passed && rule_config?.guidance) {
    const domain = REGULATED_DOMAINS.find(d => d.pattern.test(text));
    if (domain) {
      passed = false;
      rationale = `Response touches regulated domain (${domain.domain}) — policy guidance triggered.`;
    }
  }

  return { policy_id: policy._id, policy_name: name, category, passed, rationale };
}

async function runResponsibleAIPolicy({ response, policies }) {
  const flags = (policies || []).map(policy => evaluatePolicy(policy, response));
  const failedCount = flags.filter(f => !f.passed).length;

  return {
    agent: 'ResponsibleAIPolicyAgent',
    model: 'openai/claude-sonnet-4-6',
    policy_flags: flags,
    policies_checked: flags.length,
    policies_failed: failedCount,
  };
}

module.exports = { runResponsibleAIPolicy };
