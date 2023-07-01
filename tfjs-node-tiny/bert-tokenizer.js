const symbols = new Set(['!', '`', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '~' , '_', '+', '=', '[', ']', '{', '}', '\\', '|', ':', ';', '\'', '"', '<', '>', ',', '.', '?', '/']);
const toxicityTokens = require('./bert-multilingual-uncased.json');

/**
 * @param {string} text The content of a Discord message
 */
function bert_multilingual_encode(text) {
    let unclean_words = text.replace('`', '').replace('\n', ' ').toLowerCase().split(' '), ids = [101]; // [CLS]
    for (let i = 0; i < unclean_words.length && ids.length < 192; i++) {
        let cur = unclean_words[i];
        if (cur === '') continue;
        let curToken = getToken(cur);
        if (curToken) ids.push(curToken);
        else {
            let splitBySymbols = [], preSymbol = ''
            for (let j = 0; j < cur.length; j++) {
                if (symbols.has(cur[j])) {
                    if (preSymbol !== '') splitBySymbols.push(preSymbol);
                    splitBySymbols.push(cur[j]);
                    preSymbol = '';
                } else {
                    preSymbol += cur[j]
                }
            }
            splitBySymbols.push(preSymbol);
            for (let word of splitBySymbols) {
                let k = word.length, hangman = '', checkpoint = 0;
                while (k !== checkpoint && hangman !== word) {
                    for (k; k >= checkpoint + 1; k--) {
                        let checkpointJourney = word.substring(checkpoint, k), token = getToken(hangman.length === 0 ? checkpointJourney : '##' + checkpointJourney);
                        if (token) {
                            ids.push(token);
                            hangman += checkpointJourney;
                            checkpoint = k;
                            k = word.length;
                            break;
                        }
                    }
                }
                if (k === checkpoint && hangman !== word) ids.push(100) // [UNK]
                let lastcheckpointToken = getToken(word.substring(checkpoint));
                if (lastcheckpointToken) ids.push(lastcheckpointToken);
            }
        }
    }
    if (ids.length < 192) {
        ids.push(102);
        while (ids.length < 192) {
            ids.push(0);
        }
    } else if (ids.length === 192) {
        ids[191] = 102 // [SEP]
    } else {
        ids = ids.slice(0, 192);
        ids[191] = 102 // [SEP]
    }
    return ids;
}

/**
 * @param {string} string 
 * @returns {number|undefined}
 */
function getToken(string) {
    if (string === '') return undefined;
    return toxicityTokens[string];
}

module.exports.bert_multilingual_encode = bert_multilingual_encode;