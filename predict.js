async function run() {
    const { bert_multilingual_encode } = require('./tfjs-node-tiny/bert-tokenizer');
    const {loadSavedModel, tensor} = require('./tfjs-node-tiny/node');
    const model = await loadSavedModel('./bert-small-multilingual');
    const input = bert_multilingual_encode(`What's up?`);
    let t = tensor(input, [192], 'int32');
    const prediction = model.predict({
        input_ids: t
    })['output_0'];
    console.log(prediction.dataSync()[0])
}
run()