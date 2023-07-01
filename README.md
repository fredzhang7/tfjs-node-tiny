# tfjs node tiny
A light-weight, 193MB version of `@tensorflow/tfjs-node` to perform inference on any TensorFlow model in the SavedModel format. 
This repository trims all built-in TensorFlow components used for model training, while still allowing for faster model inference.

With ≈450 MB reduction in module size, ≈150%-200% the speed to load a model, and slightly faster model inference, this repository outperforms the `@tensorflow/tfjs-node` module in resource efficiency.


### Code Comparison
`@tensorflow/tfjs-node`:

```js
async function run() {
    const { node, tensor} = require('@tensorflow/tfjs-node')
    const { bert_multilingual_encode } = require('./tfjs-node-tiny/bert-tokenizer')
    const model = await node.loadSavedModel('./bert-small-multilingual');
    const input = bert_multilingual_encode(`What's up?`);
    while (input.length < 192) input.push(0);
    let t = tensor(input, [192], 'int32');
    const prediction = model.predict({
        input_ids: t
    })['output_0'];
    console.log(prediction.max().arraySync());
}
run()
```
<br>

`tfjs-node-tiny`:

```js
async function run() {
    const { bert_multilingual_encode } = require('./tfjs-node-tiny/bert-tokenizer');
    const { loadSavedModel, tensor} = require('./tfjs-node-tiny/node');
    const model = await loadSavedModel('./bert-small-multilingual');
    const input = bert_multilingual_encode(`What's up?`);
    let t = tensor(input, [192], 'int32');
    const prediction = model.predict({
        input_ids: t
    })['output_0'];
    console.log(prediction.dataSync()[0])
}
run()
```

### Setup

```bash
node setup.js
```


### Model Release
