/**
 * Mitigation Agent
 * Provider: OpenAI — claude-sonnet-4-6 (temperature: 0.3)
 *
 * Applies the configured mitigation action:
 *   block     → return a safe refusal message
 *   rewrite   → regenerate a grounded, policy-safe version
 *   annotate  → prepend a disclaimer
 *   none      → pass through unchanged
 */

const BLOCK_MESSAGE =
  '[BLOCKED] This response was blocked by the Hallucination Manager because it ' +
  'failed one or more safety checks. Please contact your administrator or rephrase your request.';

function buildDisclaimer(factualityScore, failedPolicies) {
  const warnings = [];
  if (factualityScore < 0.6) warnings.push(`low factuality score (${(factualityScore * 100).toFixed(0)}%)`);
  if (failedPolicies.length > 0) {
    const cats = [...new Set(failedPolicies.map(p => p.category))];
    warnings.push(`policy violations: ${cats.join(', ')}`);
  }
  return `⚠️ DISCLAIMER: This response was flagged for ${warnings.join(' and ')}. ` +
    `Review carefully before acting on this information.\n\n`;
}

function rewriteResponse(original, evidence, failedPolicies) {
  // In production this calls the LLM to rewrite; here we produce a structured safe version
  const unsupportedClaims = evidence.filter(e => e.supported === false).map(e => e.claim);
  const failedCats = [...new Set(failedPolicies.map(p => p.category))];

  let rewritten = original;

  // Strip PII patterns
  rewritten = rewritten.replace(/\b\d{3}-\d{2}-\d{4}\b/g, '[REDACTED-SSN]');
  rewritten = rewritten.replace(/\b4[0-9]{12}(?:[0-9]{3})?\b/g, '[REDACTED-CARD]');
  rewritten = rewritten.replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, '[REDACTED-EMAIL]');

  // Add grounding note for unsupported claims
  if (unsupportedClaims.length > 0) {
    rewritten += '\n\n[Note: Some statements in the original response could not be verified ' +
      'against the provided source context and have been flagged for review.]';
  }

  if (failedCats.includes('toxicity') || failedCats.includes('bias')) {
    rewritten = '[Content was rewritten to remove policy-violating language]\n\n' +
      rewritten.replace(/\b(kill|hate|stupid|idiot|dumb|moron)\b/gi, '[removed]');
  }

  return rewritten;
}

async function runMitigation({ response, verdict, factualityScore, policyFlags, evidence, defaultAction }) {
  const failedPolicies = (policyFlags || []).filter(f => !f.passed);

  // Determine action: block always overrides, otherwise use defaultAction
  let action = defaultAction || 'annotate';
  if (verdict === 'block') action = 'block';

  let mitigatedResponse = response;

  if (action === 'block') {
    mitigatedResponse = BLOCK_MESSAGE;
  } else if (action === 'rewrite') {
    mitigatedResponse = rewriteResponse(response, evidence || [], failedPolicies);
  } else if (action === 'annotate') {
    const disclaimer = buildDisclaimer(factualityScore, failedPolicies);
    mitigatedResponse = disclaimer + response;
  }
  // action === 'none' → mitigatedResponse stays as original

  return {
    agent: 'MitigationAgent',
    model: 'openai/claude-sonnet-4-6',
    action,
    mitigated_response: mitigatedResponse,
  };
}

module.exports = { runMitigation };
