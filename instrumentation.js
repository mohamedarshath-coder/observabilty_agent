// Bootstraps OpenTelemetry + Langfuse BEFORE any other module in this app loads.
// Must be loaded via `node -r ./instrumentation.js server.js` (see package.json's
// "start" script) rather than required at the top of server.js — OpenTelemetry's
// Node SDK has to register its instrumentation hooks before the modules it patches
// are themselves required, which `-r` guarantees and a top-of-file require does not.
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { LangfuseSpanProcessor } = require('@langfuse/otel');
const { maskFinancialAndKnownNames } = require('./lib/piiMasking');

// Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL from the
// environment automatically — same .env already loaded by server.js via dotenv.
require('dotenv').config();

// Export-time safety net: redacts financial figures and known/common names from every
// observation's input/output/metadata before it leaves the process, regardless of
// whether the end-user-facing MASKING_ENABLED pipeline in server.js is on, and
// regardless of send-order within a request (this runs at export, after all attributes
// are already set). This is the same regex logic server.js's own masking uses (see
// lib/piiMasking.js) minus the NER name pass, which needs an async model call the
// Langfuse `mask` hook's synchronous contract doesn't allow — documented as a known
// gap in LANGFUSE_OBSERVABILITY_REFERENCE.md.
const spanProcessor = new LangfuseSpanProcessor({
  mask: ({ data }) => maskFinancialAndKnownNames(data),
});

const sdk = new NodeSDK({
  spanProcessors: [spanProcessor],
});

sdk.start();

module.exports = sdk;
