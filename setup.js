const fs = require('fs');
const { pipeline } = require('stream');
const { promisify } = require('util');


const download = async (url, path) => {
    // Taken from https://levelup.gitconnected.com/how-to-download-a-file-with-node-js-e2b88fe55409

    const streamPipeline = promisify(pipeline);
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`unexpected response ${response.statusText}`);
    }

    await streamPipeline(response.body, fs.createWriteStream(path));
};

url = "https://huggingface.co/FredZhang7/bert-multilingual-toxicity-v2/resolve/main/tensorflow.dll"

download(url, "tfjs-node-tiny/tfnapi-v8/tensorflow.dll")