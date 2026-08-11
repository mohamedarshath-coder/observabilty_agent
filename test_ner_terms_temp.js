const { pipeline } = require('@xenova/transformers');

(async () => {
  const ner = await pipeline('token-classification', 'Xenova/bert-base-NER');
  const tests = [
    'Media Partner obtains reporting APIs for Customer transactions.',
    'Adobe Media Optimizer: Ad Serving and Tracking is an add-on service.',
    'The Media Spend calculation depends on Search Engine data.',
  ];
  for (const t of tests) {
    const entities = await ner(t);
    console.log('INPUT:', t);
    console.log('ENTITIES:', JSON.stringify(entities.map(e => ({ entity: e.entity, word: e.word }))));
    console.log();
  }
})();
