// Regex-only redaction layer, factored out of server.js so it can run in two places that
// both need it but can't share server.js's async pipeline: the end-user-facing masking
// block (server.js, gated behind MASKING_ENABLED) and the Langfuse export-time `mask`
// hooks (instrumentation.js / eval_service's mask_otel_spans), which the SDK requires to
// be synchronous. Keeping one copy means the two enforcement points can't drift apart.
//
// This intentionally does NOT include the NER-based name detection server.js also runs
// (maskNamesWithNER) — that needs the async @xenova/transformers pipeline and can't run
// inside a synchronous export hook. The Langfuse-side hook below is therefore weaker than
// the full end-user pipeline: it catches financial figures and known/common names, not
// every name NER would catch. Documented as a known gap in LANGFUSE_OBSERVABILITY_REFERENCE.md.

const FINANCIAL_COST_REGEX = /(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|fee|revenue|cost|payment)\b)/gi;

const NUMBER_WORD = '(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and)';
const SPELLED_OUT_CURRENCY_REGEX = new RegExp(`\\b(?:${NUMBER_WORD}\\s+){1,10}(?:dollars?|USD|cents?)\\b`, 'gi');

const CORPORATE_EXCLUSIONS = new Set([
  'Adobe', 'Inc', 'LatentView', 'Analytics', 'Corporation', 'Business', 'Head', 'Sr', 'Manager',
  'Corporate', 'Services', 'Strategic', 'Sourcing', 'Director', 'Shared', 'Operations', 'LCM',
  'EMEA', 'Support', 'Contract', 'Delivery', 'Consultants', 'Work', 'Product', 'Deliverables',
  'December', 'November', 'February', 'May', 'August', 'Project', 'There', 'However', 'The', 'In'
]);

const LOCATION_EXCLUSIONS = new Set([
  'Eiffel', 'Tower', 'Taj', 'Mahal', 'Golden', 'Gate', 'Great', 'Wall', 'Niagara', 'Falls',
  'Times', 'Square', 'Central', 'Park', 'Statue', 'Liberty', 'Big', 'Ben', 'Opera', 'House',
  'New', 'York', 'Delhi', 'Las', 'Vegas', 'Los', 'Angeles', 'San', 'Francisco', 'Hong', 'Kong',
  'South', 'North', 'United', 'States', 'Kingdom', 'Sri', 'Lanka', 'Saudi', 'Arabia'
]);

const COMMON_PERSON_NAMES = new Set([
  'Kumar', 'Abhinav', 'Rajesh', 'Suresh', 'Ramesh', 'Priya', 'Anita', 'Vijay', 'Arjun',
  'Deepak', 'Sanjay', 'Ravi', 'Ajay', 'Vikram', 'Nikhil', 'Rohit', 'Amit', 'Sunil', 'Manoj',
  'Pankaj', 'Sinha', 'Sharma', 'Gupta', 'Verma', 'Nair', 'Reddy', 'Iyer', 'Menon', 'Pillai', 'Rao'
]);

const PROPER_NOUN_NAME_PATTERN = /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g;

/**
 * Redacts financial figures (numeral and spelled-out) and known/common person names.
 * Pure regex, synchronous — safe to call from a Langfuse `mask` export hook as well as
 * from the end-user-facing masking pipeline.
 */
function maskFinancialAndKnownNames(text) {
  if (!text) return text;
  let masked = text.replace(FINANCIAL_COST_REGEX, '[REDACTED COST/REVENUE Metric]');
  masked = masked.replace(SPELLED_OUT_CURRENCY_REGEX, '[REDACTED COST/REVENUE Metric]');
  masked = masked.replace(PROPER_NOUN_NAME_PATTERN, (matchedName) => {
    const singleTokens = matchedName.split(/\s+/);
    if (singleTokens.length === 1) {
      return COMMON_PERSON_NAMES.has(matchedName) ? '[REDACTED PII / NAME Block]' : matchedName;
    }
    const isCorporateEntity = singleTokens.some(token => CORPORATE_EXCLUSIONS.has(token));
    const isKnownLocation = singleTokens.some(token => LOCATION_EXCLUSIONS.has(token));
    if (singleTokens.length >= 3) return matchedName;
    return (isCorporateEntity || isKnownLocation) ? matchedName : '[REDACTED PII / NAME Block]';
  });
  return masked;
}

module.exports = { maskFinancialAndKnownNames };
