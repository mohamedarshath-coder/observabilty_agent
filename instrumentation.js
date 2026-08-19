// Bootstraps OpenTelemetry + Langfuse BEFORE any other module in this app loads.
// Must be loaded via `node -r ./instrumentation.js server.js` (see package.json's
// "start" script) rather than required at the top of server.js — OpenTelemetry's
// Node SDK has to register its instrumentation hooks before the modules it patches
// are themselves required, which `-r` guarantees and a top-of-file require does not.
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { LangfuseSpanProcessor } = require('@langfuse/otel');

// Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL from the
// environment automatically — same .env already loaded by server.js via dotenv.
require('dotenv').config();

const sdk = new NodeSDK({
  spanProcessors: [new LangfuseSpanProcessor()],
});

sdk.start();

module.exports = sdk;
