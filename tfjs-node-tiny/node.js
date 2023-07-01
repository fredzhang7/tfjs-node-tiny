'use strict';
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function () { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function () { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
let globalNameSpace, trackerFn = null, opHandler = null;
var loadedSavedModelPathMap = new Map(), nextTFSavedModelId = 0;

function loadSavedModel(path, tags, signature) {
    if (tags === void 0) { tags = ['serve']; }
    if (signature === void 0) { signature = 'serving_default'; }
    return __awaiter(this, void 0, void 0, function () {
        var backend, savedModelInfo, signatureDefEntry, sessionId, _i, _a, id_1, modelInfo, tagsString, id, savedModel;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    assert(ENGINE.backendName === 'tensorflow', function () { return "Expect the current backend to be \"tensorflow\", but got \"" + ENGINE.backendName + "\""; });
                    backend = ENGINE.findBackend('tensorflow')
                    return [4 /*yield*/, getMetaGraphsFromSavedModel(path)];
                case 1:
                    savedModelInfo = _b.sent();
                    signatureDefEntry = getSignatureDefEntryFromMetaGraphInfo(savedModelInfo, tags, signature);
                    for (_i = 0, _a = Array.from(loadedSavedModelPathMap.keys()); _i < _a.length; _i++) {
                        id_1 = _a[_i];
                        modelInfo = loadedSavedModelPathMap.get(id_1);
                        if (modelInfo.path === path &&
                            stringArraysHaveSameElements(modelInfo.tags, tags)) {
                            sessionId = modelInfo.sessionId;
                        }
                    }
                    if (sessionId == null) {
                        tagsString = tags.join(',');
                        sessionId = backend.loadSavedModelMetaGraph(path, tagsString);
                    }
                    id = nextTFSavedModelId++;
                    savedModel = new TFSavedModel(sessionId, id, signatureDefEntry, backend);
                    loadedSavedModelPathMap.set(id, { path: path, tags: tags, sessionId: sessionId });
                    return [2 /*return*/, savedModel];
            }
        });
    });
}
function assert(expr, msg) {
    if (!expr) {
        throw new Error(typeof msg === 'string' ? msg : msg());
    }
}
var messages = require('./api_pb');
function getTFDType(dataType) {
    var binding = nodeBackend().binding;
    switch (dataType) {
        case 'float32':
            return binding.TF_FLOAT;
        case 'int32':
            return binding.TF_INT32;
        case 'bool':
            return binding.TF_BOOL;
        case 'complex64':
            return binding.TF_COMPLEX64;
        case 'string':
            return binding.TF_STRING;
        case 'int64':
            return binding.TF_INT64;
        default:
            var errorMessage = "Unknown dtype: " + dataType;
            throw new Error(errorMessage);
    }
}
function createTensorsTypeOpAttr(attrName, tensorsOrDtype) {
    if (util_1.isNullOrUndefined(tensorsOrDtype)) {
        throw new Error('Invalid input tensors value.');
    }
    return {
        name: attrName,
        type: nodeBackend().binding.TF_ATTR_TYPE,
        value: (tensorsOrDtype instanceof Tensor || Array.isArray(tensorsOrDtype)) ?
            getTFDTypeForInputs(tensorsOrDtype) :
            getTFDType(tensorsOrDtype)
    };
}
function getTFDTypeForInputs(tensors) {
    if (util_1.isNullOrUndefined(tensors)) {
        throw new Error('Invalid input tensors value.');
    }
    if (util_1.isArray(tensors)) {
        for (var i = 0; i < tensors.length; i++) {
            return getTFDType(tensors[i].dtype);
        }
        return -1;
    }
    else {
        return getTFDType(tensors.dtype);
    }
}
function getOrMakeEngine() {
    const ns = getGlobalNamespace();
    if (ns._tfengine == null) {
        const environment = new Environment(ns);
        ns._tfengine = new Engine(environment);
    }
    setEnvironmentGlobal(ns._tfengine.ENV);
    setTensorTracker(() => ns._tfengine);
    return ns._tfengine;
}
const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
function isPromise(object) {
    return object && object.then && typeof object.then === 'function';
}
function getQueryParams(queryString) {
    const params = {};
    queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => {
        decodeParam(params, t[0], t[1]);
        return t.join('=');
    });
    return params;
}
function decodeParam(params, name, value) {
    params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}
function parseValue(flagName, value) {
    value = value.toLowerCase();
    if (value === 'true' || value === 'false') {
        return value === 'true';
    }
    else if (`${+value}` === value) {
        return +value;
    }
    throw new Error(`Could not parse value flag value ${value} for flag ${flagName}.`);
}
function env() {
    return ENV;
}
let ENV = null;
function setEnvironmentGlobal(environment) {
    ENV = environment;
}
const kernelRegistry = getGlobal('kernelRegistry', () => new Map());
function getKernelsForBackend(backendName) {
    const it = kernelRegistry.entries();
    const result = [];
    while (true) {
        const { done, value } = it.next();
        if (done) {
            break;
        }
        const [key, config] = value;
        const [backend,] = key.split('_');
        if (backend === backendName) {
            result.push(config);
        }
    }
    return result;
}
function getGlobalNamespace() {
    if (globalNameSpace == null) {
        let ns;
        if (typeof (window) !== 'undefined') {
            ns = window;
        }
        else if (typeof (global) !== 'undefined') {
            ns = global;
        }
        else if (typeof (process) !== 'undefined') {
            ns = process;
        }
        else if (typeof (self) !== 'undefined') {
            ns = self;
        }
        else {
            throw new Error('Could not find a global object');
        }
        globalNameSpace = ns;
    }
    return globalNameSpace;
}
function getGlobalMap() {
    const ns = getGlobalNamespace();
    if (ns._tfGlobals == null) {
        ns._tfGlobals = new Map();
    }
    return ns._tfGlobals;
}
function getGlobal(key, init) {
    const globalMap = getGlobalMap();
    if (globalMap.has(key)) {
        return globalMap.get(key);
    }
    else {
        const singleton = init();
        globalMap.set(key, singleton);
        return globalMap.get(key);
    }
}
function setTensorTracker(fn) {
    trackerFn = fn;
}
function sizeFromShape(shape) {
    if (shape.length === 0) {
        return 1;
    }
    let size = shape[0];
    for (let i = 1; i < shape.length; i++) {
        size *= shape[i];
    }
    return size;
}
function computeStrides(shape) {
    const rank = shape.length;
    if (rank < 2) {
        return [];
    }
    // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
    // strides.
    const strides = new Array(rank - 1);
    strides[rank - 2] = shape[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
function assertNonNegativeIntegerDimensions(shape) {
    shape.forEach(dimSize => {
        assert(Number.isInteger(dimSize) && dimSize >= 0, () => `Tensor must have a shape comprised of positive integers but got shape [${shape}].`);
    });
}
function flatten(arr, result = [], skipTypedArray = false) {
    if (result == null) {
        result = [];
    }
    if (Array.isArray(arr) || isTypedArray(arr) && !skipTypedArray) {
        for (let i = 0; i < arr.length; ++i) {
            flatten(arr[i], result, skipTypedArray);
        }
    }
    else {
        result.push(arr);
    }
    return result;
}
function isTypedArray(a) {
    return a instanceof Float32Array || a instanceof Int32Array || a instanceof Uint8Array || a instanceof Uint8ClampedArray;
}
function noConversionNeeded(a, dtype) {
    return (a instanceof Float32Array && dtype === 'float32') || (a instanceof Int32Array && dtype === 'int32') || (a instanceof Uint8Array && dtype === 'bool');
}
function checkConversionForErrors(vals, dtype) {
    for (let i = 0; i < vals.length; i++) {
        const num = vals[i];
        if (isNaN(num) || !isFinite(num)) {
            throw Error(`A tensor of type ${dtype} being uploaded contains ${num}.`);
        }
    }
}
function toTypedArray(a, dtype) {
    if (dtype === 'string') {
        throw new Error('Cannot convert a string[] to a TypedArray');
    }
    if (Array.isArray(a)) {
        a = flatten(a);
    }
    // if (env().getBool('DEBUG')) {
    //     checkConversionForErrors(a, dtype);
    // }
    if (noConversionNeeded(a, dtype)) {
        return a;
    }
    if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
        return new Float32Array(a);
    }
    else if (dtype === 'int32') {
        return new Int32Array(a);
    }
    else if (dtype === 'bool') {
        const bool = new Uint8Array(a.length);
        for (let i = 0; i < bool.length; ++i) {
            if (Math.round(a[i]) !== 0) {
                bool[i] = 1;
            }
        }
        return bool;
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
function isString(value) {
    return typeof value === 'string' || value instanceof String;
}
function isBoolean(value) {
    return typeof value === 'boolean';
}
function isNumber(value) {
    return typeof value === 'number';
}
function isFunction(f) {
    return !!(f && f.constructor && f.call && f.apply);
}
function inferDtype(values) {
    if (Array.isArray(values)) {
        return inferDtype(values[0]);
    }
    if (values instanceof Float32Array) {
        return 'float32';
    }
    else if (values instanceof Int32Array || values instanceof Uint8Array || values instanceof Uint8ClampedArray) {
        return 'int32';
    }
    else if (isNumber(values)) {
        return 'float32';
    }
    else if (isString(values)) {
        return 'string';
    }
    else if (isBoolean(values)) {
        return 'bool';
    }
    return 'float32';
}
function scalar(value, dtype) {
    if (((isTypedArray(value) && dtype !== 'string') || Array.isArray(value)) &&
        dtype !== 'complex64') {
        throw new Error('Error creating a new Scalar: value must be a primitive ' +
            '(number|boolean|string)');
    }
    if (dtype === 'string' && isTypedArray(value) &&
        !(value instanceof Uint8Array)) {
        throw new Error('When making a scalar from encoded string, ' +
            'the value must be `Uint8Array`.');
    }
    const shape = [];
    const inferredShape = [];
    return makeTensor(value, shape, inferredShape, dtype);
}
function tensor(values, shape, dtype) {
    const inferredShape = inferShape(values, dtype);
    return makeTensor(values, shape, inferredShape, dtype);
}
function inferShape(val, dtype) {
    let firstElem = val;
    if (isTypedArray(val)) {
        return dtype === 'string' ? [] : [val.length];
    }
    if (!Array.isArray(val)) {
        return [];
    }
    const shape = [];
    while (Array.isArray(firstElem) ||
        isTypedArray(firstElem) && dtype !== 'string') {
        shape.push(firstElem.length);
        firstElem = firstElem[0];
    }
    // if (Array.isArray(val) &&
    //     env().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')) {
    //     deepAssertShapeConsistency(val, shape, []);
    // }
    return shape;
}
function deepAssertShapeConsistency(val, shape, indices) {
    indices = indices || [];
    if (!(Array.isArray(val)) && !isTypedArray(val)) {
        assert(shape.length === 0, () => `Element arr[${indices.join('][')}] is a primitive, ` +
            `but should be an array/TypedArray of ${shape[0]} elements`);
        return;
    }
    assert(shape.length > 0, () => `Element arr[${indices.join('][')}] should be a primitive, ` +
        `but is an array of ${val.length} elements`);
    assert(val.length === shape[0], () => `Element arr[${indices.join('][')}] should have ${shape[0]} ` +
        `elements, but has ${val.length} elements`);
    const subShape = shape.slice(1);
    for (let i = 0; i < val.length; ++i) {
        deepAssertShapeConsistency(val[i], subShape, indices.concat(i));
    }
}
function makeTensor(values, shape, inferredShape, dtype) {
    if (dtype == null) {
        dtype = inferDtype(values);
    }
    if (dtype === 'complex64') {
        throw new Error(`Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).`);
    }
    if (!isTypedArray(values) && !Array.isArray(values) &&
        typeof values !== 'number' && typeof values !== 'boolean' &&
        typeof values !== 'string') {
        throw new Error('values passed to tensor(values) must be a number/boolean/string or ' +
            'an array of numbers/booleans/strings, or a TypedArray');
    }
    if (shape != null) {
        assertNonNegativeIntegerDimensions(shape);
        const providedSize = sizeFromShape(shape);
        const inferredSize = sizeFromShape(inferredShape);
        assert(providedSize === inferredSize, () => `Based on the provided shape, [${shape}], the tensor should have ` +
            `${providedSize} values but has ${inferredSize}`);
        for (let i = 0; i < inferredShape.length; ++i) {
            const inferred = inferredShape[i];
            const flatDimsDontMatch = i === inferredShape.length - 1 ?
                inferred !== sizeFromShape(shape.slice(i)) :
                true;
            assert(inferredShape[i] === shape[i] || !flatDimsDontMatch, () => `Error creating a new Tensor. Inferred shape ` +
                `(${inferredShape}) does not match the provided shape (${shape}). `);
        }
    }
    if (!isTypedArray(values) && !Array.isArray(values)) {
        values = [values];
    }
    shape = shape || inferredShape;
    values = dtype !== 'string' ?
        toTypedArray(values, dtype) :
        flatten(values, [], true);
    return ENGINE.makeTensor(values, shape, dtype);
}
function setOpHandler(handler) {
    opHandler = handler;
}
function decodeString(bytes, encoding) {
    if (bytes.length === 0) {
        return '';
    }
    return new TextDecoder(encoding).decode(bytes);
}
function toNestedArray(shape, a, isComplex = false) {
    if (shape.length === 0) {
        return a[0];
    }
    const size = shape.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
    if (size === 0) {
        return [];
    }
    if (size !== a.length) {
        throw new Error(`[${shape}] does not match the input size ${a.length}${isComplex ? ' for a complex tensor' : ''}.`);
    }
    return createNestedArray(0, shape, a, isComplex);
}
function createNestedArray(offset, shape, a, isComplex = false) {
    const ret = new Array();
    if (shape.length === 1) {
        const d = shape[0] * (isComplex ? 2 : 1);
        for (let i = 0; i < d; i++) {
            ret[i] = a[offset + i];
        }
    }
    else {
        const d = shape[0];
        const rest = shape.slice(1);
        const len = rest.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
        for (let i = 0; i < d; i++) {
            ret[i] = createNestedArray(offset + i * len, rest, a, isComplex);
        }
    }
    return ret;
}
function getEnumKeyFromValue(object, value) {
    return Object.keys(object).find(function (key) { return object[key] === value; });
}
function getSignatureDefEntryFromMetaGraphInfo(savedModelInfo, tags, signature) {
    for (var i = 0; i < savedModelInfo.length; i++) {
        var metaGraphInfo = savedModelInfo[i];
        if (stringArraysHaveSameElements(tags, metaGraphInfo.tags)) {
            if (metaGraphInfo.signatureDefs[signature] == null) {
                throw new Error('The SavedModel does not have signature: ' + signature);
            }
            return metaGraphInfo.signatureDefs[signature];
        }
    }
    throw new Error("The SavedModel does not have tags: " + tags);
}
function stringArraysHaveSameElements(arrayA, arrayB) {
    if (arrayA.length === arrayB.length &&
        arrayA.sort().join() === arrayB.sort().join()) {
        return true;
    }
    return false;
}
function mapTFDtypeToJSDtype(tfDtype) {
    switch (tfDtype) {
        case 'DT_FLOAT':
            return 'float32';
        case 'DT_INT64':
        case 'DT_INT32':
        case 'DT_UINT8':
            return 'int32';
        case 'DT_BOOL':
            return 'bool';
        case 'DT_COMPLEX64':
            return 'complex64';
        case 'DT_STRING':
            return 'string';
        default:
            throw new Error(`Unsupported tensor DataType: ${tfDtype}. Try to modify the model in python to convert the datatype`);
    }
}
var SAVED_MODEL_INIT_OP_KEY = '__saved_model_init_op';
function getMetaGraphsFromSavedModel(path) {
    return __awaiter(this, void 0, void 0, function () {
        var result, modelMessage, metaGraphList, i, metaGraph, tags, signatureDef, signatureDefMap, signatureDefKeys, key, signatureDefEntry, inputsMapMessage, inputsMapKeys, inputs, inputsMapKey, inputTensor, inputTensorInfo, dtype, outputsMapMessage, outputsMapKeys, outputs, outputsMapKey, outputTensor, outputTensorInfo, dtype;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    result = [];
                    return [4 /*yield*/, readSavedModelProto(path)];
                case 1:
                    modelMessage = _a.sent();
                    metaGraphList = modelMessage.getMetaGraphsList();
                    for (i = 0; i < metaGraphList.length; i++) {
                        metaGraph = {};
                        tags = metaGraphList[i].getMetaInfoDef().getTagsList();
                        metaGraph.tags = tags;
                        signatureDef = {};
                        signatureDefMap = metaGraphList[i].getSignatureDefMap();
                        signatureDefKeys = signatureDefMap.keys();
                        while (true) {
                            key = signatureDefKeys.next();
                            if (key.done) {
                                break;
                            }
                            if (key.value === SAVED_MODEL_INIT_OP_KEY) {
                                continue;
                            }
                            signatureDefEntry = signatureDefMap.get(key.value);
                            inputsMapMessage = signatureDefEntry.getInputsMap();
                            inputsMapKeys = inputsMapMessage.keys();
                            inputs = {};
                            while (true) {
                                inputsMapKey = inputsMapKeys.next();
                                if (inputsMapKey.done) {
                                    break;
                                }
                                inputTensor = inputsMapMessage.get(inputsMapKey.value);
                                inputTensorInfo = {};
                                dtype = getEnumKeyFromValue(messages.DataType, inputTensor.getDtype());
                                inputTensorInfo.dtype = mapTFDtypeToJSDtype(dtype);
                                inputTensorInfo.tfDtype = dtype;
                                inputTensorInfo.name = inputTensor.getName();
                                inputTensorInfo.shape = inputTensor.getTensorShape().getDimList();
                                inputs[inputsMapKey.value] = inputTensorInfo;
                            }
                            outputsMapMessage = signatureDefEntry.getOutputsMap();
                            outputsMapKeys = outputsMapMessage.keys();
                            outputs = {};
                            while (true) {
                                outputsMapKey = outputsMapKeys.next();
                                if (outputsMapKey.done) {
                                    break;
                                }
                                outputTensor = outputsMapMessage.get(outputsMapKey.value);
                                outputTensorInfo = {};
                                dtype = getEnumKeyFromValue(messages.DataType, outputTensor.getDtype());
                                outputTensorInfo.dtype = mapTFDtypeToJSDtype(dtype);
                                outputTensorInfo.tfDtype = dtype;
                                outputTensorInfo.name = outputTensor.getName();
                                outputTensorInfo.shape = outputTensor.getTensorShape().getDimList();
                                outputs[outputsMapKey.value] = outputTensorInfo;
                            }
                            signatureDef[key.value] = { inputs: inputs, outputs: outputs };
                        }
                        metaGraph.signatureDefs = signatureDef;
                        result.push(metaGraph);
                    }
                    return [2 /*return*/, result];
            }
        });
    });
}
var util_1 = require('util'), os = require("os");;
var readFile = util_1.promisify(fs.readFile);
var SAVED_MODEL_FILE_NAME = '/saved_model.pb';
function readSavedModelProto(path) {
    return __awaiter(this, void 0, void 0, function () {
        var modelFile, array;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    try {
                        fs.accessSync(path + SAVED_MODEL_FILE_NAME, fs.constants.R_OK);
                    }
                    catch (error) {
                        throw new Error('There is no saved_model.pb file in the directory: ' + path);
                    }
                    return [4, readFile(path + SAVED_MODEL_FILE_NAME)];
                case 1:
                    modelFile = _a.sent();
                    array = new Uint8Array(modelFile);
                    return [2, messages.SavedModel.deserializeBinary(array)];
            }
        });
    });
}
function now() {
    return env().platform.now();
}
function rightPad(a, size) {
    if (size <= a.length) {
        return a;
    }
    return a + ' '.repeat(size - a.length);
}
function checkComputationForErrors(vals, dtype, kernelName) {
    if (dtype !== 'float32') {
        return false;
    }
    for (let i = 0; i < vals.length; i++) {
        const num = vals[i];
        if (isNaN(num) || !isFinite(num)) {
            console.warn(`Found ${num} in the result of '${kernelName}'`);
            return true;
        }
    }
    return false;
}
function notYetImplemented(kernelName) {
    throw new Error(`'${kernelName}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`);
}
function encodeInt32ArrayAsInt64(value) {
    if (os.endianness() !== 'LE') {
        throw new Error("Int64Scalar does not support endianness of this machine: " + ("" + os.endianness()));
    }
    var buffer = new Int32Array(value.length * 2);
    for (var i = 0; i < value.length; i++) {
        buffer[i * 2] = value[i];
    }
    return buffer;
}
function bytesPerElement(dtype) {
    if (dtype === 'float32' || dtype === 'int32') {
        return 4;
    }
    else if (dtype === 'complex64') {
        return 8;
    }
    else if (dtype === 'bool') {
        return 1;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
function encodeString(s, encoding = 'utf-8') {
    encoding = encoding || 'utf-8';
    return env().platform.encode(s, encoding);
}
function makeZerosTypedArray(size, dtype) {
    if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
        return new Float32Array(size);
    }
    else if (dtype === 'int32') {
        return new Int32Array(size);
    }
    else if (dtype === 'bool') {
        return new Uint8Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
function bytesFromStringArray(arr) {
    if (arr == null) {
        return 0;
    }
    let bytes = 0;
    arr.forEach(x => bytes += x.length);
    return bytes;
}

class EngineState {
    constructor() {
        this.registeredVariables = {};
        this.nextTapeNodeId = 0;
        this.numBytes = 0;
        this.numTensors = 0;
        this.numStringTensors = 0;
        this.numDataBuffers = 0;
        this.gradientDepth = 0;
        this.kernelDepth = 0;
        this.scopeStack = [];
        this.numDataMovesStack = [];
        this.nextScopeId = 0;
        this.tensorInfo = new WeakMap();
        this.profiling = false;
        this.activeProfile = {
            newBytes: 0,
            newTensors: 0,
            peakBytes: 0,
            kernels: [],
            result: null,
            get kernelNames() {
                return Array.from(new Set(this.kernels.map(k => k.name)));
            }
        };
    }
    dispose() {
        for (const variableName in this.registeredVariables) {
            this.registeredVariables[variableName].dispose();
        }
    }
}
class Profiler {
    constructor(backendTimer, logger) {
        this.backendTimer = backendTimer;
        this.logger = logger;
        if (logger == null) {
            this.logger = new Logger();
        }
    }
    profileKernel(kernelName, inputs, f) {
        let outputs;
        const holdResultWrapperFn = () => {
            outputs = f();
        };
        let timer;
        const start = now();
        if (this.backendTimer.timerAvailable()) {
            timer = this.backendTimer.time(holdResultWrapperFn);
        }
        else {
            holdResultWrapperFn();
            for (const output of outputs) {
                output.dataSync();
            }
            timer = Promise.resolve({ kernelMs: now() - start });
        }
        if (env().getBool('CHECK_COMPUTATION_FOR_ERRORS')) {
            for (let i = 0; i < outputs.length; i++) {
                const output = outputs[i];
                output.data().then(tensorVals => {
                    checkComputationForErrors(tensorVals, output.dtype, kernelName);
                });
            }
        }
        const kernelProfile = {
            kernelName,
            outputs,
            inputs,
            timeMs: timer.then(timing => timing.kernelMs),
            extraInfo: timer.then(timing => timing.getExtraProfileInfo != null ? timing.getExtraProfileInfo() : '')
        };
        return kernelProfile;
    }
    logKernelProfile(kernelProfile) {
        const { kernelName, outputs, timeMs, inputs, extraInfo } = kernelProfile;
        outputs.forEach(result => {
            Promise.all([result.data(), timeMs, extraInfo]).then(valueContainer => {
                this.logger.logKernelProfile(kernelName, result, valueContainer[0], valueContainer[1], inputs, valueContainer[2]);
            });
        });
    }
}
class Logger {
    logKernelProfile(name, result, vals, timeMs, inputs, extraInfo) {
        const time = typeof timeMs === 'number' ? rightPad(`${timeMs}ms`, 9) : timeMs['error'];
        const paddedName = rightPad(name, 25);
        const rank = result.rank;
        const size = result.size;
        const shape = rightPad(result.shape.toString(), 14);
        let inputShapesDescription = '';
        for (const name in inputs) {
            const input = inputs[name];
            if (input != null) {
                const inputShape = input.shape || result.shape;
                const inputRank = inputShape.length;
                inputShapesDescription +=
                    `${name}: ${inputRank}D ${inputRank > 0 ? inputShape : ''} `;
            }
        }
        console.log(`%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}\t%c${inputShapesDescription}\t%c${extraInfo}`, 'font-weight:bold', 'color:red', 'color:blue', 'color: orange', 'color: green', 'color: steelblue');
    }
}
class Engine {
    constructor(ENV) {
        this.ENV = ENV;
        this.registry = {};
        this.registryFactory = {};
        this.pendingBackendInitId = 0;
        this.state = new EngineState();
    }
    async ready() {
        if (this.pendingBackendInit != null) {
            return this.pendingBackendInit.then(() => { });
        }
        if (this.backendInstance != null) {
            return;
        }
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const success = await this.initializeBackend(backendName).success;
            if (success) {
                await this.setBackend(backendName);
                return;
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations failed.`);
    }
    get backend() {
        if (this.pendingBackendInit != null) {
            throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure ` +
                `to await tf.ready() or await tf.setBackend() before calling other methods`);
        }
        if (this.backendInstance == null) {
            const { name, asyncInit } = this.initializeBackendsAndReturnBest();
            if (asyncInit) {
                throw new Error(`The highest priority backend '${name}' has not yet been ` +
                    `initialized. Make sure to await tf.ready() or ` +
                    `await tf.setBackend() before calling other methods`);
            }
            this.setBackend(name);
        }
        return this.backendInstance;
    }
    backendNames() {
        return Object.keys(this.registryFactory);
    }
    findBackend(backendName) {
        if (!(backendName in this.registry)) {
            if (backendName in this.registryFactory) {
                const { asyncInit } = this.initializeBackend(backendName);
                if (asyncInit) {
                    return null;
                }
            }
            else {
                return null;
            }
        }
        return this.registry[backendName];
    }
    findBackendFactory(backendName) {
        if (!(backendName in this.registryFactory)) {
            return null;
        }
        return this.registryFactory[backendName].factory;
    }
    registerBackend(backendName, factory, priority = 1) {
        if (backendName in this.registryFactory) {
            console.warn(`${backendName} backend was already registered. Reusing existing backend factory.`);
            return false;
        }
        this.registryFactory[backendName] = { factory, priority };
        return true;
    }
    async setBackend(backendName) {
        if (this.registryFactory[backendName] == null) {
            throw new Error(`Backend name '${backendName}' not found in registry`);
        }
        this.backendName = backendName;
        if (this.registry[backendName] == null) {
            this.backendInstance = null;
            const { success, asyncInit } = this.initializeBackend(backendName);
            const result = asyncInit ? await success : success;
            if (!result) {
                return false;
            }
        }
        this.backendInstance = this.registry[backendName];
        this.setupRegisteredKernels();
        this.profiler = new Profiler(this.backendInstance);
        return true;
    }
    setupRegisteredKernels() {
        const kernels = getKernelsForBackend(this.backendName);
        kernels.forEach(kernel => {
            if (kernel.setupFunc != null) {
                kernel.setupFunc(this.backendInstance);
            }
        });
    }
    disposeRegisteredKernels(backendName) {
        const kernels = getKernelsForBackend(backendName);
        kernels.forEach(kernel => {
            if (kernel.disposeFunc != null) {
                kernel.disposeFunc(this.registry[backendName]);
            }
        });
    }
    initializeBackend(backendName) {
        const registryFactoryEntry = this.registryFactory[backendName];
        if (registryFactoryEntry == null) {
            throw new Error(`Cannot initialize backend ${backendName}, no registration found.`);
        }
        try {
            const backend = registryFactoryEntry.factory();
            if (backend && !(backend instanceof KernelBackend) &&
                typeof backend.then === 'function') {
                const promiseId = ++this.pendingBackendInitId;
                const success = backend
                    .then(backendInstance => {
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.registry[backendName] = backendInstance;
                    this.pendingBackendInit = null;
                    return true;
                })
                    .catch(err => {
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.pendingBackendInit = null;
                    console.warn(`Initialization of backend ${backendName} failed`);
                    console.warn(err.stack || err.message);
                    return false;
                });
                this.pendingBackendInit = success;
                return { success, asyncInit: true };
            }
            else {
                this.registry[backendName] = backend;
                return { success: true, asyncInit: false };
            }
        }
        catch (err) {
            console.warn(`Initialization of backend ${backendName} failed`);
            console.warn(err.stack || err.message);
            return { success: false, asyncInit: false };
        }
    }
    removeBackend(backendName) {
        if (!(backendName in this.registryFactory)) {
            throw new Error(`${backendName} backend not found in registry`);
        }
        if (this.backendName === backendName && this.pendingBackendInit != null) {
            this.pendingBackendInitId++;
        }
        if (backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        delete this.registryFactory[backendName];
        if (this.backendName === backendName) {
            this.pendingBackendInit = null;
            this.backendName = null;
            this.backendInstance = null;
        }
    }
    getSortedBackends() {
        if (Object.keys(this.registryFactory).length === 0) {
            throw new Error('No backend found in registry.');
        }
        return Object.keys(this.registryFactory).sort((a, b) => {
            return this.registryFactory[b].priority - this.registryFactory[a].priority;
        });
    }
    initializeBackendsAndReturnBest() {
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const { success, asyncInit } = this.initializeBackend(backendName);
            if (asyncInit || success) {
                return { name: backendName, asyncInit };
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations failed.`);
    }
    moveData(backend, dataId) {
        const info = this.state.tensorInfo.get(dataId);
        const srcBackend = info.backend;
        const values = this.readSync(dataId);
        const refCount = srcBackend.refCount(dataId);
        srcBackend.disposeData(dataId, true);
        info.backend = backend;
        backend.move(dataId, values, info.shape, info.dtype, refCount);
        if (this.shouldCheckForMemLeaks()) {
            this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
        }
    }
    tidy(nameOrFn, fn) {
        let name = null;
        if (fn == null) {
            if (typeof nameOrFn !== 'function') {
                throw new Error('Please provide a function to tidy()');
            }
            fn = nameOrFn;
        }
        else {
            if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
                throw new Error('When calling with two arguments, the first argument ' +
                    'to tidy() must be a string');
            }
            if (typeof fn !== 'function') {
                throw new Error('When calling with two arguments, the 2nd argument ' +
                    'to tidy() must be a function');
            }
            name = nameOrFn;
        }
        let result;
        return this.scopedRun(() => this.startScope(name), () => this.endScope(result), () => {
            result = fn();
            if (result instanceof Promise) {
                console.error('Cannot return a Promise inside of tidy.');
            }
            return result;
        });
    }
    scopedRun(start, end, f) {
        start();
        try {
            const res = f();
            end();
            return res;
        }
        catch (ex) {
            end();
            throw ex;
        }
    }
    nextTensorId() {
        return Engine.nextTensorId++;
    }
    nextVariableId() {
        return Engine.nextVariableId++;
    }
    clone(x) {
        const y = ENGINE.runKernel(Identity, { x });
        const inputs = { x };
        const grad = (dy) => ({
            x: () => {
                const dtype = 'float32';
                const gradInputs = { x: dy };
                const attrs = { dtype };
                return ENGINE.runKernel(Cast, gradInputs, attrs);
            }
        });
        const saved = [];
        this.addTapeNode(this.state.activeScope.name, inputs, [y], grad, saved, {});
        return y;
    }
    runKernel(kernelName, inputs, attrs) {
        if (this.backendName == null) {
            this.backend;
        }
        const hasKernel = getKernel(kernelName, this.backendName) != null;
        if (!hasKernel) {
            throw new Error(`Kernel '${kernelName}' not registered for backend '${this.backendName}'`);
        }
        return this.runKernelFunc({ kernelName, inputs, attrs });
    }
    shouldCheckForMemLeaks() {
        return this.ENV.getBool('IS_TEST');
    }
    checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos) {
        const numDataIdsAfter = this.backend.numDataIds();
        let numOutputDataIds = 0;
        outInfos.forEach(info => {
            numOutputDataIds += (info.dtype === 'complex64' ? 3 : 1);
        });
        const numMoves = this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
        const dataIdsLeaked = numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
        if (dataIdsLeaked > 0) {
            throw new Error(`Backend '${this.backendName}' has an internal memory leak ` +
                `(${dataIdsLeaked} data ids) after running '${kernelName}'`);
        }
    }
    runKernelFunc(kernelParams) {
        let outputs;
        let saved = [];
        const isTapeOn = this.isTapeOn();
        const startingBytecount = this.state.numBytes;
        const startingNumTensors = this.state.numTensors;
        if (this.shouldCheckForMemLeaks()) {
            this.state.numDataMovesStack.push(0);
        }
        let kernelFunc;
        if (this.backendName == null) {
            this.backend;
        }
        let out;
        const kernelOrScopeName = isRegisteredKernelInvocation(kernelParams) ?
            kernelParams.kernelName :
            this.state.activeScope != null ? this.state.activeScope.name : '';
        if (isRegisteredKernelInvocation(kernelParams)) {
            const { kernelName, inputs, attrs } = kernelParams;
            if (this.backendName == null) {
                this.backend;
            }
            const kernel = getKernel(kernelName, this.backendName);
            assert(kernel != null, () => `Cannot find registered kernel '${kernelName}' for backend '${this.backendName}'`);
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = kernel.kernelFunc({ inputs, attrs, backend: this.backend });
                const outInfos = Array.isArray(out) ? out : [out];
                if (this.shouldCheckForMemLeaks()) {
                    this.checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos);
                }
                const outTensors = outInfos.map((outInfo) => {
                    if (outInfo.rank != null) {
                        return outInfo;
                    }
                    return this.makeTensorFromTensorInfo(outInfo);
                });
                if (isTapeOn) {
                    const tensorsToSave = this.getTensorsForGradient(kernelName, inputs, outTensors);
                    saved = this.saveTensorsForBackwardMode(tensorsToSave);
                }
                return outTensors;
            };
        }
        else {
            const { forwardFunc } = kernelParams;
            const saveFunc = (tensors) => {
                if (!isTapeOn) {
                    return;
                }
                saved = tensors.map(tensor => this.keep(this.clone(tensor)));
            };
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = this.tidy(() => forwardFunc(this.backend, saveFunc));
                const outs = (Array.isArray(out) ? out : [out]);
                if (this.shouldCheckForMemLeaks()) {
                    this.checkKernelForMemLeak(kernelOrScopeName, numDataIdsBefore, outs);
                }
                return outs;
            };
        }
        const { inputs, attrs } = kernelParams;
        const backwardsFunc = isRegisteredKernelInvocation(kernelParams) ?
            null :
            kernelParams.backwardsFunc;
        let kernelProfile;
        this.scopedRun(
        () => this.state.kernelDepth++, () => this.state.kernelDepth--, () => {
            if (!this.ENV.getBool('DEBUG') && !this.state.profiling) {
                outputs = kernelFunc();
            }
            else {
                kernelProfile = this.profiler.profileKernel(kernelOrScopeName, inputs, () => kernelFunc());
                if (this.ENV.getBool('DEBUG')) {
                    this.profiler.logKernelProfile(kernelProfile);
                }
                outputs = kernelProfile.outputs;
            }
        });
        if (isTapeOn) {
            this.addTapeNode(kernelOrScopeName, inputs, outputs, backwardsFunc, saved, attrs);
        }
        if (this.state.profiling) {
            this.state.activeProfile.kernels.push({
                name: kernelOrScopeName,
                bytesAdded: this.state.numBytes - startingBytecount,
                totalBytesSnapshot: this.state.numBytes,
                tensorsAdded: this.state.numTensors - startingNumTensors,
                totalTensorsSnapshot: this.state.numTensors,
                inputShapes: Object.keys(inputs).map(key => inputs[key] != null ? inputs[key].shape : null),
                outputShapes: outputs.map(item => item.shape),
                kernelTimeMs: kernelProfile.timeMs,
                extraInfo: kernelProfile.extraInfo
            });
        }
        return (Array.isArray(out) ? outputs : outputs[0]);
    }
    saveTensorsForBackwardMode(tensors) {
        const saved = tensors.map(tensor => this.keep(this.clone(tensor)));
        return saved;
    }
    getTensorsForGradient(kernelName, inputs, outputs) {
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            const inputsToSave = gradConfig.inputsToSave || [];
            const outputsToSave = gradConfig.outputsToSave || [];
            let inputTensorsToSave;
            if (gradConfig.saveAllInputs) {
                assert(Array.isArray(inputs), () => 'saveAllInputs is true, expected inputs to be an array.');
                inputTensorsToSave = Object.keys(inputs).map((key) => inputs[key]);
            }
            else {
                inputTensorsToSave = inputsToSave.map((inputName) => inputs[inputName]);
            }
            const outputTensorsToSave = outputs.filter((_, i) => outputsToSave[i]);
            return inputTensorsToSave.concat(outputTensorsToSave);
        }
        return [];
    }
    makeTensor(values, shape, dtype, backend) {
        if (values == null) {
            throw new Error('Values passed to engine.makeTensor() are null');
        }
        dtype = dtype || 'float32';
        backend = backend || this.backend;
        let backendVals = values;
        if (dtype === 'string' && isString(values[0])) {
            backendVals = values.map(d => encodeString(d));
        }
        const dataId = backend.write(backendVals, shape, dtype);
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        if (dtype === 'string') {
            const info = this.state.tensorInfo.get(dataId);
            const newBytes = bytesFromStringArray(backendVals);
            this.state.numBytes += newBytes - info.bytes;
            info.bytes = newBytes;
        }
        return t;
    }
    makeTensorFromDataId(dataId, shape, dtype, backend) {
        dtype = dtype || 'float32';
        const tensorInfo = { dataId, shape, dtype };
        return this.makeTensorFromTensorInfo(tensorInfo, backend);
    }
    makeTensorFromTensorInfo(tensorInfo, backend) {
        const { dataId, shape, dtype } = tensorInfo;
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        return t;
    }
    makeVariable(initialValue, trainable = true, name, dtype) {
        name = name || this.nextVariableId().toString();
        if (dtype != null && dtype !== initialValue.dtype) {
            initialValue = initialValue.cast(dtype);
        }
        const v = new Variable(initialValue, trainable, name, this.nextTensorId());
        if (this.state.registeredVariables[v.name] != null) {
            throw new Error(`Variable with name ${v.name} was already registered`);
        }
        this.state.registeredVariables[v.name] = v;
        this.incRef(v, this.backend);
        return v;
    }
    trackTensor(a, backend) {
        this.state.numTensors++;
        if (a.dtype === 'string') {
            this.state.numStringTensors++;
        }
        let bytes = 0;
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            bytes = a.size * bytesPerElement(a.dtype);
        }
        this.state.numBytes += bytes;
        if (!this.state.tensorInfo.has(a.dataId)) {
            this.state.numDataBuffers++;
            this.state.tensorInfo.set(a.dataId, {
                backend: backend || this.backend,
                dtype: a.dtype,
                shape: a.shape,
                bytes
            });
        }
        if (!(a instanceof Variable)) {
            this.track(a);
        }
    }
    incRef(a, backend) {
        this.trackTensor(a, backend);
        this.backend.incRef(a.dataId);
    }
    removeDataId(dataId, backend) {
        if (this.state.tensorInfo.has(dataId) &&
            this.state.tensorInfo.get(dataId).backend === backend) {
            this.state.tensorInfo.delete(dataId);
            this.state.numDataBuffers--;
        }
    }
    disposeTensor(a) {
        if (!this.state.tensorInfo.has(a.dataId)) {
            return;
        }
        const info = this.state.tensorInfo.get(a.dataId);
        this.state.numTensors--;
        if (a.dtype === 'string') {
            this.state.numStringTensors--;
            this.state.numBytes -= info.bytes;
        }
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            const bytes = a.size * bytesPerElement(a.dtype);
            this.state.numBytes -= bytes;
        }
        if (info.backend.disposeData(a.dataId)) {
            this.removeDataId(a.dataId, info.backend);
        }
    }
    disposeVariables() {
        for (const varName in this.state.registeredVariables) {
            const v = this.state.registeredVariables[varName];
            this.disposeVariable(v);
        }
    }
    disposeVariable(v) {
        this.disposeTensor(v);
        if (this.state.registeredVariables[v.name] != null) {
            delete this.state.registeredVariables[v.name];
        }
    }
    memory() {
        const info = this.backend.memory();
        info.numTensors = this.state.numTensors;
        info.numDataBuffers = this.state.numDataBuffers;
        info.numBytes = this.state.numBytes;
        if (this.state.numStringTensors > 0) {
            info.unreliable = true;
            if (info.reasons == null) {
                info.reasons = [];
            }
            info.reasons.push('Memory usage by string tensors is approximate (2 bytes per character)');
        }
        return info;
    }
    async profile(query) {
        this.state.profiling = true;
        const startBytes = this.state.numBytes;
        const startNumTensors = this.state.numTensors;
        this.state.activeProfile.kernels = [];
        this.state.activeProfile.result = await query();
        this.state.profiling = false;
        this.state.activeProfile.peakBytes = Math.max(...this.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
        this.state.activeProfile.newBytes = this.state.numBytes - startBytes;
        this.state.activeProfile.newTensors =
            this.state.numTensors - startNumTensors;
        for (const kernel of this.state.activeProfile.kernels) {
            kernel.kernelTimeMs = await kernel.kernelTimeMs;
            kernel.extraInfo = await kernel.extraInfo;
        }
        return this.state.activeProfile;
    }
    isTapeOn() {
        return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
    }
    addTapeNode(kernelName, inputs, outputs, gradientsFunc, saved, attrs) {
        const tapeNode = { id: this.state.nextTapeNodeId++, kernelName, inputs, outputs, saved };
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            gradientsFunc = gradConfig.gradFunc;
        }
        if (gradientsFunc != null) {
            tapeNode.gradient = (dys) => {
                dys = dys.map((dy, i) => {
                    if (dy == null) {
                        const output = outputs[i];
                        const vals = makeZerosTypedArray(output.size, output.dtype);
                        return this.makeTensor(vals, output.shape, output.dtype);
                    }
                    return dy;
                });
                return gradientsFunc(dys.length > 1 ? dys : dys[0], saved, attrs);
            };
        }
        this.state.activeTape.push(tapeNode);
    }
    keep(result) {
        result.kept = true;
        return result;
    }
    startTape() {
        if (this.state.gradientDepth === 0) {
            this.state.activeTape = [];
        }
        this.state.gradientDepth++;
    }
    endTape() {
        this.state.gradientDepth--;
    }
    startScope(name) {
        const scopeInfo = {
            track: [],
            name: 'unnamed scope',
            id: this.state.nextScopeId++
        };
        if (name) {
            scopeInfo.name = name;
        }
        this.state.scopeStack.push(scopeInfo);
        this.state.activeScope = scopeInfo;
    }
    endScope(result) {
        const tensorsToTrackInParent = getTensorsInContainer(result);
        const tensorsToTrackInParentSet = new Set(tensorsToTrackInParent.map(t => t.id));
        for (let i = 0; i < this.state.activeScope.track.length; i++) {
            const tensor = this.state.activeScope.track[i];
            if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
                tensor.dispose();
            }
        }
        const oldScope = this.state.scopeStack.pop();
        this.state.activeScope = this.state.scopeStack.length === 0 ?
            null :
            this.state.scopeStack[this.state.scopeStack.length - 1];
        tensorsToTrackInParent.forEach(tensor => {
            if (!tensor.kept && tensor.scopeId === oldScope.id) {
                this.track(tensor);
            }
        });
    }
    gradients(f, xs, dy, allowNoGradients = false) {
        assert(xs.length > 0, () => 'gradients() received an empty list of xs.');
        if (dy != null && dy.dtype !== 'float32') {
            throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
        }
        const y = this.scopedRun(() => this.startTape(), () => this.endTape(), () => this.tidy('forward', f));
        assert(y instanceof Tensor, () => 'The result y returned by f() must be a tensor.');
        const filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
        if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
            throw new Error('Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
                'that the f you passed encloses all operations that lead from x to y.');
        }
        return this.tidy('backward', () => {
            const accumulatedGradientMap = {};
            accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
            backpropagateGradients(accumulatedGradientMap, filteredTape, 
            f => this.tidy(f), 
            add);
            const grads = xs.map(x => accumulatedGradientMap[x.id]);
            if (this.state.gradientDepth === 0) {
                this.state.activeTape.forEach(node => {
                    for (const tensor of node.saved) {
                        tensor.dispose();
                    }
                });
                this.state.activeTape = null;
            }
            return { value: y, grads };
        });
    }
    customGrad(f) {
        assert(isFunction(f), () => 'The f passed in customGrad(f) must be a function.');
        return (...inputs) => {
            assert(inputs.every(t => t instanceof Tensor), () => 'The args passed in customGrad(f)(x1, x2,...) must all be tensors');
            let res;
            const inputMap = {};
            inputs.forEach((input, i) => {
                inputMap[i] = input;
            });
            const forwardFunc = (_, save) => {
                res = f(...[...inputs, save]);
                assert(res.value instanceof Tensor, () => 'The function f passed in customGrad(f) must return an object where `obj.value` is a tensor');
                assert(isFunction(res.gradFunc), () => 'The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function.');
                return res.value;
            };
            const backwardsFunc = (dy, saved) => {
                const gradRes = res.gradFunc(dy, saved);
                const grads = Array.isArray(gradRes) ? gradRes : [gradRes];
                assert(grads.length === inputs.length, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'the same number of tensors as inputs passed to f(...).');
                assert(grads.every(t => t instanceof Tensor), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns a list of only tensors.');
                const gradMap = {};
                grads.forEach((grad, i) => {
                    gradMap[i] = () => grad;
                });
                return gradMap;
            };
            return this.runKernelFunc({
                forwardFunc,
                backwardsFunc,
                inputs: inputMap,
            });
        };
    }
    readSync(dataId) {
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readSync(dataId);
    }
    read(dataId) {
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.read(dataId);
    }
    readToGPU(dataId, options) {
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readToGPU(dataId, options);
    }
    async time(query) {
        const start = now();
        const timingInfo = await this.backend.time(query);
        timingInfo.wallMs = now() - start;
        return timingInfo;
    }
    track(result) {
        if (this.state.activeScope != null) {
            result.scopeId = this.state.activeScope.id;
            this.state.activeScope.track.push(result);
        }
        return result;
    }
    get registeredVariables() {
        return this.state.registeredVariables;
    }
    reset() {
        this.pendingBackendInitId++;
        this.state.dispose();
        this.ENV.reset();
        this.state = new EngineState();
        for (const backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        this.backendName = null;
        this.backendInstance = null;
        this.pendingBackendInit = null;
    }
}
Engine.nextTensorId = 0;
Engine.nextVariableId = 0;
class Tensor {
    constructor(shape, dtype, dataId, id) {
        this.kept = false;
        this.isDisposedInternal = false;
        this.shape = shape.slice();
        this.dtype = dtype || 'float32';
        this.size = sizeFromShape(shape);
        this.strides = computeStrides(shape);
        this.dataId = dataId;
        this.id = id || 0;
        this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
    }
    get rank() {
        return this.shape.length;
    }
    async buffer() {
        const vals = await this.data();
        return opHandler.buffer(this.shape, this.dtype, vals);
    }
    bufferSync() {
        return opHandler.buffer(this.shape, this.dtype, this.dataSync());
    }
    async array() {
        const vals = await this.data();
        return toNestedArray(this.shape, vals, this.dtype === 'complex64');
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * synchronously.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    arraySync() {
        return toNestedArray(this.shape, this.dataSync(), this.dtype === 'complex64');
    }
    async data() {
        this.throwIfDisposed();
        const data = trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            const bytes = await data;
            try {
                return bytes.map(b => decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    dataToGPU(options) {
        this.throwIfDisposed();
        return trackerFn().readToGPU(this.dataId, options);
    }
    dataSync() {
        this.throwIfDisposed();
        const data = trackerFn().readSync(this.dataId);
        if (this.dtype === 'string') {
            try {
                return data.map(b => decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /** Returns the underlying bytes of the tensor's data. */
    async bytes() {
        this.throwIfDisposed();
        const data = await trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            return data;
        }
        else {
            return new Uint8Array(data.buffer);
        }
    }
    /**
     * Disposes `tf.Tensor` from memory.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dispose() {
        if (this.isDisposed) {
            return;
        }
        trackerFn().disposeTensor(this);
        this.isDisposedInternal = true;
    }
    get isDisposed() {
        return this.isDisposedInternal;
    }
    throwIfDisposed() {
        if (this.isDisposed) {
            throw new Error(`Tensor is disposed.`);
        }
    }
    print(verbose = false) {
        return opHandler.print(this, verbose);
    }
    clone() {
        this.throwIfDisposed();
        return opHandler.clone(this);
    }
    toString(verbose = false) {
        const vals = this.dataSync();
        return tensorToString(vals, this.shape, this.dtype, verbose);
    }
    cast(dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    }
    variable(trainable = true, name, dtype) {
        this.throwIfDisposed();
        return trackerFn().makeVariable(this, trainable, name, dtype);
    }
}
const EPSILON_FLOAT32 = 1e-7, EPSILON_FLOAT16 = 1e-4;
class KernelBackend {
    refCount(dataId) {
        return notYetImplemented('refCount');
    }
    incRef(dataId) {
        return notYetImplemented('incRef');
    }
    timerAvailable() {
        return true;
    }
    time(f) {
        return notYetImplemented('time');
    }
    read(dataId) {
        return notYetImplemented('read');
    }
    readSync(dataId) {
        return notYetImplemented('readSync');
    }
    readToGPU(dataId, options) {
        return notYetImplemented('readToGPU');
    }
    numDataIds() {
        return notYetImplemented('numDataIds');
    }
    disposeData(dataId, force) {
        return notYetImplemented('disposeData');
    }
    write(values, shape, dtype) {
        return notYetImplemented('write');
    }
    move(dataId, values, shape, dtype, refCount) {
        return notYetImplemented('move');
    }
    memory() {
        return notYetImplemented('memory');
    }
    floatPrecision() {
        return notYetImplemented('floatPrecision');
    }
    epsilon() {
        return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
    }
    dispose() {
        return notYetImplemented('dispose');
    }
}
class Environment {
    constructor(global) {
        this.global = global;
        this.flags = {};
        this.flagRegistry = {};
        this.urlFlags = {};
        this.getQueryParams = getQueryParams;
        this.populateURLFlags();
    }
    setPlatform(platformName, platform) {
        if (this.platform != null) {
            if (!(env().getBool('IS_TEST') || env().getBool('PROD'))) {
                console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${platformName}.`);
            }
        }
        this.platformName = platformName;
        this.platform = platform;
    }
    registerFlag(flagName, evaluationFn, setHook) {
        this.flagRegistry[flagName] = { evaluationFn, setHook };
        if (this.urlFlags[flagName] != null) {
            const flagValue = this.urlFlags[flagName];
            if (!(env().getBool('IS_TEST') || env().getBool('PROD'))) {
                console.warn(`Setting feature override from URL ${flagName}: ${flagValue}.`);
            }
            this.set(flagName, flagValue);
        }
    }
    async getAsync(flagName) {
        if (flagName in this.flags) {
            return this.flags[flagName];
        }
        this.flags[flagName] = await this.evaluateFlag(flagName);
        return this.flags[flagName];
    }
    get(flagName) {
        if (flagName in this.flags) {
            return this.flags[flagName];
        }
        const flagValue = this.evaluateFlag(flagName);
        if (isPromise(flagValue)) {
            throw new Error(`Flag ${flagName} cannot be synchronously evaluated. Please use getAsync() instead.`);
        }
        this.flags[flagName] = flagValue;
        return this.flags[flagName];
    }
    getNumber(flagName) {
        return this.get(flagName);
    }
    getBool(flagName) {
        return this.get(flagName);
    }
    getFlags() {
        return this.flags;
    }
    get features() {
        return this.flags;
    }
    set(flagName, value) {
        if (this.flagRegistry[flagName] == null) {
            throw new Error(`Cannot set flag ${flagName} as it has not been registered.`);
        }
        this.flags[flagName] = value;
        if (this.flagRegistry[flagName].setHook != null) {
            this.flagRegistry[flagName].setHook(value);
        }
    }
    evaluateFlag(flagName) {
        if (this.flagRegistry[flagName] == null) {
            throw new Error(`Cannot evaluate flag '${flagName}': no evaluation function found.`);
        }
        return this.flagRegistry[flagName].evaluationFn();
    }
    setFlags(flags) {
        this.flags = Object.assign({}, flags);
    }
    reset() {
        this.flags = {};
        this.urlFlags = {};
        this.populateURLFlags();
    }
    populateURLFlags() {
        if (typeof this.global === 'undefined' || typeof this.global.location === 'undefined' || typeof this.global.location.search === 'undefined') {
            return;
        }
        const urlParams = this.getQueryParams(this.global.location.search);
        if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
            const keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
            keyValues.forEach(keyValue => {
                const [key, value] = keyValue.split(':');
                this.urlFlags[key] = parseValue(key, value);
            });
        }
    }
}
class DataStorage {
    constructor(backend, dataMover) {
        this.backend = backend;
        this.dataMover = dataMover;
        this.data = new WeakMap();
        this.dataIdsCount = 0;
    }
    get(dataId) {
        if (!this.data.has(dataId)) {
            this.dataMover.moveData(this.backend, dataId);
        }
        return this.data.get(dataId);
    }
    set(dataId, value) {
        this.dataIdsCount++;
        this.data.set(dataId, value);
    }
    has(dataId) {
        return this.data.has(dataId);
    }
    delete(dataId) {
        this.dataIdsCount--;
        return this.data.delete(dataId);
    }
    numDataIds() {
        return this.dataIdsCount;
    }
}
var TFSavedModel = /** @class */ (function () {
    function TFSavedModel(sessionId, jsid, signature, backend) {
        this.sessionId = sessionId;
        this.jsid = jsid;
        this.signature = signature;
        this.backend = backend;
        this.disposed = false;
    }
    Object.defineProperty(TFSavedModel.prototype, "inputs", {
        /**
         * Return the array of input tensor info.
         *
         * @doc {heading: 'Models', subheading: 'SavedModel'}
         */
        get: function () {
            var entries = this.signature.inputs;
            var results = Object.keys(entries).map(function (key) { return entries[key]; });
            results.forEach(function (info) {
                info.name = info.name.replace(/:0$/, '');
            });
            return results;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TFSavedModel.prototype, "outputs", {
        /**
         * Return the array of output tensor info.
         *
         * @doc {heading: 'Models', subheading: 'SavedModel'}
         */
        get: function () {
            var entries = this.signature.outputs;
            var results = Object.keys(entries).map(function (key) { return entries[key]; });
            results.forEach(function (info) {
                info.name = info.name.replace(/:0$/, '');
            });
            return results;
        },
        enumerable: true,
        configurable: true
    });
    /**
     * Delete the SavedModel from nodeBackend and delete corresponding session in
     * the C++ backend if the session is only used by this TFSavedModel.
     *
     * @doc {heading: 'Models', subheading: 'SavedModel'}
     */
    TFSavedModel.prototype.dispose = function () {
        if (!this.disposed) {
            this.disposed = true;
            loadedSavedModelPathMap.delete(this.jsid);
            for (var _i = 0, _a = Array.from(loadedSavedModelPathMap.keys()); _i < _a.length; _i++) {
                var id = _a[_i];
                var value = loadedSavedModelPathMap.get(id);
                if (value.sessionId === this.sessionId) {
                    return;
                }
            }
            this.backend.deleteSavedModel(this.sessionId);
        }
        else {
            throw new Error('This SavedModel has already been deleted.');
        }
    };
    Object.defineProperty(TFSavedModel.prototype, "outputNodeNames", {
        get: function () {
            var _this = this;
            if (this.outputNodeNames_ != null) {
                return this.outputNodeNames_;
            }
            this.outputNodeNames_ =
                Object.keys(this.signature.outputs)
                    .reduce(function (names, key) {
                        names[key] = _this.signature.outputs[key].name;
                        return names;
                    }, {});
            return this.outputNodeNames_;
        },
        enumerable: true,
        configurable: true
    });
    /**
     * Execute the inference for the input tensors.
     *
     * @param input The input tensors, when there is single input for the model,
     * inputs param should be a Tensor. For models with multiple inputs, inputs
     * params should be in either Tensor[] if the input order is fixed, or
     * otherwise NamedTensorMap format. The keys in the NamedTensorMap are the
     * name of input tensors in SavedModel signatureDef. It can be found through
     * `tf.node.getMetaGraphsFromSavedModel()`.
     *
     * For batch inference execution, the tensors for each input need to be
     * concatenated together. For example with mobilenet, the required input shape
     * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
     * If we are provide a batched data of 100 images, the input tensor should be
     * in the shape of [100, 244, 244, 3].
     *
     * @param config Prediction configuration for specifying the batch size.
     *
     * @returns Inference result tensors. The output would be single Tensor if
     * model has single output node, otherwise Tensor[] or NamedTensorMap[] will
     * be returned for model with multiple outputs.
     *
     * @doc {heading: 'Models', subheading: 'SavedModel'}
     */
    TFSavedModel.prototype.predict = function (inputs, config) {
        var _this = this;
        if (this.disposed) {
            throw new Error('The TFSavedModel has already been deleted!');
        }
        else {
            var inputTensors = [];
            if (inputs instanceof Tensor) {
                inputTensors.push(inputs);
                var result = this.backend.runSavedModel(this.sessionId, inputTensors, Object.values(this.signature.inputs), Object.values(this.outputNodeNames));
                return result.length > 1 ? result : result[0];
            }
            else if (Array.isArray(inputs)) {
                inputTensors = inputs;
                return this.backend.runSavedModel(this.sessionId, inputTensors, Object.values(this.signature.inputs), Object.values(this.outputNodeNames));
            }
            else {
                var inputTensorNames = Object.keys(this.signature.inputs);
                var providedInputNames = Object.keys(inputs);
                if (!stringArraysHaveSameElements(inputTensorNames, providedInputNames)) {
                    throw new Error("The model signatureDef input names are " + inputTensorNames.join() + ", however the provided input names are " + providedInputNames.join() + ".");
                }
                var inputNodeNamesArray = [];
                for (var i = 0; i < inputTensorNames.length; i++) {
                    inputTensors.push(inputs[inputTensorNames[i]]);
                    inputNodeNamesArray.push(this.signature.inputs[inputTensorNames[i]]);
                }
                var outputTensorNames = Object.keys(this.outputNodeNames);
                var outputNodeNamesArray = [];
                for (var i = 0; i < outputTensorNames.length; i++) {
                    outputNodeNamesArray.push(this.outputNodeNames[outputTensorNames[i]]);
                }
                var outputTensors_1 = this.backend.runSavedModel(this.sessionId, inputTensors, inputNodeNamesArray, outputNodeNamesArray);
                assert(outputTensors_1.length === outputNodeNamesArray.length, function () {
                    return 'Output tensors do not match output node names, ' +
                        ("receive " + outputTensors_1.length + ") output tensors but ") +
                        ("there are " + _this.outputNodeNames.length + " output nodes.");
                });
                var outputMap = {};
                for (var i = 0; i < outputTensorNames.length; i++) {
                    outputMap[outputTensorNames[i]] = outputTensors_1[i];
                }
                return outputMap;
            }
        }
    };
    /**
     * Execute the inference for the input tensors and return activation
     * values for specified output node names without batching.
     *
     * @param input The input tensors, when there is single input for the model,
     * inputs param should be a Tensor. For models with multiple inputs, inputs
     * params should be in either Tensor[] if the input order is fixed, or
     * otherwise NamedTensorMap format.
     *
     * @param outputs string|string[]. List of output node names to retrieve
     * activation from.
     *
     * @returns Activation values for the output nodes result tensors. The return
     * type matches specified parameter outputs type. The output would be single
     * Tensor if single output is specified, otherwise Tensor[] for multiple
     * outputs.
     *
     * @doc {heading: 'Models', subheading: 'SavedModel'}
     */
    TFSavedModel.prototype.execute = function (inputs, outputs) {
        throw new Error('execute() of TFSavedModel is not supported yet.');
    };
    return TFSavedModel;
}());
var Int64Scalar = /** @class */ (function () {
    function Int64Scalar(value) {
        this.value = value;
        this.dtype = 'int64';
        this.rank = 1;
        if (Int64Scalar.endiannessOkay_ == null) {
            if (os.endianness() !== 'LE') {
                throw new Error("Int64Scalar does not support endianness of this machine: " +
                    ("" + os.endianness()));
            }
            Int64Scalar.endiannessOkay_ = true;
        }
        assert(value > -INT32_MAX && value < INT32_MAX - 1, function () {
            return "Got a value outside of the bound of values supported for int64 " +
                ("dtype ([-" + INT32_MAX + ", " + (INT32_MAX - 1) + "]): " + value);
        });
        assert(Number.isInteger(value), function () { return "Expected value to be an integer, but got " + value; });
        var highPart = value >= 0 ? 0 : -1;
        var lowPart = value % INT32_MAX;
        this.valueArray_ = new Int32Array([lowPart, highPart]);
    }
    Object.defineProperty(Int64Scalar.prototype, "shape", {
        get: function () {
            return [];
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Int64Scalar.prototype, "valueArray", {
        get: function () {
            return this.valueArray_;
        },
        enumerable: true,
        configurable: true
    });
    return Int64Scalar;
}());
var NodeJSKernelBackend = /** @class */ (function () {
    __extends(NodeJSKernelBackend, KernelBackend);
    function NodeJSKernelBackend(binding, packageName) {
        // var _this = KernelBackend.call(this) || this;
        var _this = this;
        _this.binding = binding;
        _this.isGPUPackage = packageName === '@tensorflow/tfjs-node-gpu';
        _this.isUsingGpuDevice = _this.binding.isUsingGpuDevice();
        _this.tensorMap = new DataStorage(_this, ENGINE);
        return _this;
    }
    NodeJSKernelBackend.prototype.getDTypeInteger = function (dtype) {
        switch (dtype) {
            case 'float32':
                return this.binding.TF_FLOAT;
            case 'int32':
                return this.binding.TF_INT32;
            case 'bool':
                return this.binding.TF_BOOL;
            case 'complex64':
                return this.binding.TF_COMPLEX64;
            case 'string':
                return this.binding.TF_STRING;
            default:
                throw new Error("Unsupported DType: " + dtype);
        }
    };
    NodeJSKernelBackend.prototype.typeAttributeFromTensor = function (value) {
        return this.getDTypeInteger(value.dtype);
    };
    NodeJSKernelBackend.prototype.createOutputTensor = function (metadata) {
        var newId = {};
        this.tensorMap.set(newId, {
            shape: metadata.shape,
            dtype: metadata.dtype,
            id: metadata.id,
            values: null,
            refCount: 1
        });
        var dtype;
        switch (metadata.dtype) {
            case this.binding.TF_FLOAT:
                dtype = 'float32';
                break;
            case this.binding.TF_INT32:
                dtype = 'int32';
                break;
            case this.binding.TF_INT64:
                console.warn('INT64 output tensor will be stored as BigInt64Array.');
                dtype = 'int32';
                break;
            case this.binding.TF_BOOL:
                dtype = 'bool';
                break;
            case this.binding.TF_COMPLEX64:
                dtype = 'complex64';
                break;
            case this.binding.TF_STRING:
                dtype = 'string';
                break;
            case this.binding.TF_RESOURCE:
                dtype = 'string';
                break;
            case this.binding.TF_UINT8:
                dtype = 'int32';
                break;
            default:
                throw new Error("Unknown dtype enum " + metadata.dtype);
        }
        var tensorInfo = {
            dataId: newId, shape: metadata.shape, dtype: dtype
        };
        return ENGINE.makeTensorFromTensorInfo(tensorInfo);
    };
    NodeJSKernelBackend.prototype.getInputTensorIds = function (tensors) {
        var ids = [];
        for (var i = 0; i < tensors.length; i++) {
            if (tensors[i] instanceof Int64Scalar) {
                var value = tensors[i].valueArray;
                var id = this.binding.createTensor([], this.binding.TF_INT64, value);
                ids.push(id);
            }
            else {
                var info = this.tensorMap.get(tensors[i].dataId);
                if (info.values != null) {
                    info.id =
                        this.binding.createTensor(info.shape, info.dtype, info.values);
                    info.values = null;
                }
                ids.push(info.id);
            }
        }
        return ids;
    };
    NodeJSKernelBackend.prototype.createReductionOpAttrs = function (tensor, keepDims) {
        if (keepDims === void 0) { keepDims = false; }
        return [
            { name: 'keep_dims', type: this.binding.TF_ATTR_BOOL, value: keepDims },
            createTensorsTypeOpAttr('T', tensor.dtype),
            createTensorsTypeOpAttr('Tidx', 'int32')
        ];
    };
    NodeJSKernelBackend.prototype.floatPrecision = function () {
        return 32;
    };
    NodeJSKernelBackend.prototype.epsilon = function () {
        return _super.prototype.epsilon.call(this);
    };
    NodeJSKernelBackend.prototype.executeSingleInput = function (name, input) {
        var opAttrs = [createTensorsTypeOpAttr('T', input.dtype)];
        return this.executeSingleOutput(name, opAttrs, [input]);
    };
    NodeJSKernelBackend.prototype.executeSingleOutput = function (name, opAttrs, inputs) {
        var outputMetadata = this.binding.executeOp(name, opAttrs, this.getInputTensorIds(inputs), 1);
        return this.createOutputTensor(outputMetadata[0]);
    };
    NodeJSKernelBackend.prototype.executeMultipleOutputs = function (name, opAttrs, inputs, numOutputs) {
        var _this = this;
        var outputMetadata = this.binding.executeOp(name, opAttrs, this.getInputTensorIds(inputs), numOutputs);
        return outputMetadata.map(function (m) { return _this.createOutputTensor(m); });
    };
    NodeJSKernelBackend.prototype.numDataIds = function () {
        return this.tensorMap.numDataIds();
    };
    NodeJSKernelBackend.prototype.dispose = function () { };
    NodeJSKernelBackend.prototype.read = function (dataId) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2, this.readSync(dataId)];
            });
        });
    };
    NodeJSKernelBackend.prototype.readSync = function (dataId) {
        if (!this.tensorMap.has(dataId)) {
            throw new Error("Tensor " + dataId + " was not registered!");
        }
        var info = this.tensorMap.get(dataId);
        if (info.values != null) {
            return info.values;
        }
        else {
            return this.binding.tensorDataSync(info.id);
        }
    };
    /**
     * Dispose the memory if the dataId has 0 refCount. Return true if the memory
     * is released, false otherwise.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    NodeJSKernelBackend.prototype.disposeData = function (dataId, force) {
        if (force === void 0) { force = false; }
        // No-op if already disposed.
        if (this.tensorMap.has(dataId)) {
            var id = this.tensorMap.get(dataId).id;
            this.tensorMap.get(dataId).refCount--;
            if (!force && this.tensorMap.get(dataId).refCount > 0) {
                return false;
            }
            if (id != null && id >= 0) {
                this.binding.deleteTensor(id);
            }
            this.tensorMap.delete(dataId);
        }
        return true;
    };
    /** Return refCount of a `TensorData`. */
    NodeJSKernelBackend.prototype.refCount = function (dataId) {
        if (this.tensorMap.has(dataId)) {
            var tensorData = this.tensorMap.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    };
    NodeJSKernelBackend.prototype.incRef = function (dataId) {
        this.tensorMap.get(dataId).refCount++;
    };
    NodeJSKernelBackend.prototype.move = function (dataId, values, shape, dtype, refCount) {
        this.tensorMap.set(dataId, { shape: shape, dtype: getTFDType(dtype), values: values, id: -1, refCount: refCount });
    };
    NodeJSKernelBackend.prototype.write = function (values, shape, dtype) {
        var dataId = {};
        this.move(dataId, values, shape, dtype, 1);
        return dataId;
    };
    NodeJSKernelBackend.prototype.applyActivation = function (input, activation, preluActivationWeights, leakyreluAlpha) {
        var result = input;
        if (activation != null) {
            if (activation === 'linear') {
                // No-op
            }
            else if (activation === 'relu') {
                result = tf.relu(result);
            }
            else if (activation === 'prelu') {
                result = tf.prelu(result, preluActivationWeights);
            }
            else if (activation === 'leakyrelu') {
                result = tf.leakyRelu(result, leakyreluAlpha);
            }
            else if (activation === 'elu') {
                result = tf.elu(result);
            }
            else if (activation === 'relu6') {
                result = tf.relu6(result);
            }
            else if (activation === 'sigmoid') {
                result = tf.sigmoid(result);
            }
            else {
                throw new Error("Activation: " + activation + " has not been implemented for the Node.js backend");
            }
        }
        return result;
    };
    NodeJSKernelBackend.prototype.divide = function (a, b) {
        var opAttrs = [createTensorsTypeOpAttr('T', tfjs_1.backend_util.upcastType(a.dtype, b.dtype))];
        return this.executeSingleOutput('Div', opAttrs, [a, b]);
    };
    NodeJSKernelBackend.prototype.divNoNan = function (a, b) {
        var opAttrs = [createTensorsTypeOpAttr('T', tfjs_1.backend_util.upcastType(a.dtype, b.dtype))];
        return this.executeSingleOutput('DivNoNan', opAttrs, [a, b]);
    };
    NodeJSKernelBackend.prototype.where = function (condition) {
        return this.executeSingleOutput('Where', [], [condition]);
    };
    NodeJSKernelBackend.prototype.topKValues = function (x, k) {
        throw new Error('Method not implemented.');
    };
    NodeJSKernelBackend.prototype.topKIndices = function (x, k) {
        throw new Error('Method not implemented.');
    };
    NodeJSKernelBackend.prototype.int = function (x) {
        throw new Error('Method not implemented.');
    };
    NodeJSKernelBackend.prototype.decodeJpeg = function (contents, channels, ratio, fancyUpscaling, tryRecoverTruncated, acceptableFraction, dctMethod) {
        var opAttrs = [
            { name: 'channels', type: this.binding.TF_ATTR_INT, value: channels },
            { name: 'ratio', type: this.binding.TF_ATTR_INT, value: ratio }, {
                name: 'fancy_upscaling',
                type: this.binding.TF_ATTR_BOOL,
                value: fancyUpscaling
            },
            {
                name: 'try_recover_truncated',
                type: this.binding.TF_ATTR_BOOL,
                value: tryRecoverTruncated
            },
            {
                name: 'acceptable_fraction',
                type: this.binding.TF_ATTR_FLOAT,
                value: acceptableFraction
            },
            { name: 'dct_method', type: this.binding.TF_ATTR_STRING, value: dctMethod }
        ];
        var inputArgs = [scalar(contents, 'string')];
        return this.executeSingleOutput('DecodeJpeg', opAttrs, inputArgs);
    };
    NodeJSKernelBackend.prototype.decodePng = function (contents, channels) {
        var opAttrs = [{ name: 'channels', type: this.binding.TF_ATTR_INT, value: channels }];
        var inputArgs = [scalar(contents, 'string')];
        return this.executeSingleOutput('DecodePng', opAttrs, inputArgs);
    };
    NodeJSKernelBackend.prototype.decodeBmp = function (contents, channels) {
        var opAttrs = [{ name: 'channels', type: this.binding.TF_ATTR_INT, value: channels }];
        var inputArgs = [scalar(contents, 'string')];
        return this.executeSingleOutput('DecodeBmp', opAttrs, inputArgs);
    };
    NodeJSKernelBackend.prototype.decodeGif = function (contents) {
        var inputArgs = [scalar(contents, 'string')];
        return this.executeSingleOutput('DecodeGif', [], inputArgs);
    };
    NodeJSKernelBackend.prototype.executeEncodeImageOp = function (name, opAttrs, imageData, imageShape) {
        var inputTensorId = this.binding.createTensor(imageShape, this.binding.TF_UINT8, imageData);
        var outputMetadata = this.binding.executeOp(name, opAttrs, [inputTensorId], 1);
        var outputTensorInfo = outputMetadata[0];
        outputTensorInfo.dtype = this.binding.TF_UINT8;
        return this.createOutputTensor(outputTensorInfo);
    };
    NodeJSKernelBackend.prototype.encodeJpeg = function (imageData, imageShape, format, quality, progressive, optimizeSize, chromaDownsampling, densityUnit, xDensity, yDensity, xmpMetadata) {
        var opAttrs = [
            { name: 'format', type: this.binding.TF_ATTR_STRING, value: format },
            { name: 'quality', type: this.binding.TF_ATTR_INT, value: quality }, {
                name: 'progressive',
                type: this.binding.TF_ATTR_BOOL,
                value: progressive
            },
            {
                name: 'optimize_size',
                type: this.binding.TF_ATTR_BOOL,
                value: optimizeSize
            },
            {
                name: 'chroma_downsampling',
                type: this.binding.TF_ATTR_BOOL,
                value: chromaDownsampling
            },
            {
                name: 'density_unit',
                type: this.binding.TF_ATTR_STRING,
                value: densityUnit
            },
            { name: 'x_density', type: this.binding.TF_ATTR_INT, value: xDensity },
            { name: 'y_density', type: this.binding.TF_ATTR_INT, value: yDensity }, {
                name: 'xmp_metadata',
                type: this.binding.TF_ATTR_STRING,
                value: xmpMetadata
            }
        ];
        return this.executeEncodeImageOp('EncodeJpeg', opAttrs, imageData, imageShape);
    };
    NodeJSKernelBackend.prototype.encodePng = function (imageData, imageShape, compression) {
        var opAttrs = [
            { name: 'compression', type: this.binding.TF_ATTR_INT, value: compression }
        ];
        return this.executeEncodeImageOp('EncodePng', opAttrs, imageData, imageShape);
    };
    NodeJSKernelBackend.prototype.deleteSavedModel = function (id) {
        this.binding.deleteSavedModel(id);
    };
    NodeJSKernelBackend.prototype.loadSavedModelMetaGraph = function (path, tags) {
        return this.binding.loadSavedModel(path, tags);
    };
    NodeJSKernelBackend.prototype.getMappedInputTensorIds = function (inputs, inputTensorInfos) {
        var tensorIds = this.getInputTensorIds(inputs);
        for (var i = 0; i < inputs.length; i++) {
            if (inputTensorInfos[i] != null) {
                if (inputTensorInfos[i].tfDtype === 'DT_UINT8') {
                    var data = Uint8Array.from(inputs[i].dataSync());
                    var inputTensorId = this.binding.createTensor(inputs[i].shape, this.binding.TF_UINT8, data);
                    tensorIds[i] = inputTensorId;
                }
                else if (inputTensorInfos[i].tfDtype === 'DT_INT64') {
                    var data = encodeInt32ArrayAsInt64(inputs[i].dataSync());
                    var inputTensorId = this.binding.createTensor(inputs[i].shape, this.binding.TF_INT64, data);
                    tensorIds[i] = inputTensorId;
                }
            }
        }
        return tensorIds;
    };
    NodeJSKernelBackend.prototype.runSavedModel = function (id, inputs, inputTensorInfos, outputOpNames) {
        var _this = this;
        var outputMetadata = this.binding.runSavedModel(id, this.getMappedInputTensorIds(inputs, inputTensorInfos), inputTensorInfos.map(function (info) { return info.name; }).join(','), outputOpNames.join(','));
        return outputMetadata.map(function (m) { return _this.createOutputTensor(m); });
    };
    NodeJSKernelBackend.prototype.summaryWriter = function (logdir) {
        var opAttrs = [
            {
                name: 'shared_name',
                type: this.binding.TF_ATTR_STRING,
                value: "logdir:" + logdir
            },
            { name: 'container', type: this.binding.TF_ATTR_STRING, value: '' }
        ];
        var writerResource = this.executeSingleOutput('SummaryWriter', opAttrs, []);
        return writerResource;
    };
    NodeJSKernelBackend.prototype.createSummaryFileWriter = function (resourceHandle, logdir, maxQueue, flushMillis, filenameSuffix) {
        var inputArgs = [
            resourceHandle, scalar(logdir),
            scalar(maxQueue == null ? 10 : maxQueue, 'int32'),
            scalar(flushMillis == null ? 2 * 60 * 1000 : flushMillis, 'int32'),
            scalar(filenameSuffix == null ? '.v2' : filenameSuffix)
        ];
        this.executeMultipleOutputs('CreateSummaryFileWriter', [], inputArgs, 0);
    };
    NodeJSKernelBackend.prototype.writeScalarSummary = function (resourceHandle, step, name, value) {
        var _this = this;
        tidy(function () {
            assert(Number.isInteger(step), function () { return "step is expected to be an integer, but is instead " + step; });
            var inputArgs = [resourceHandle, new Int64Scalar(step), scalar(name, 'string')];
            var typeAttr;
            if (typeof value === 'number') {
                inputArgs.push(scalar(value));
                typeAttr = _this.binding.TF_FLOAT;
            }
            else {
                assert(value.rank === 0, function () {
                    return "A non-scalar tensor (rank " + value.rank + ") is passed to writeScalarSummary()";
                });
                inputArgs.push(value);
                typeAttr = _this.typeAttributeFromTensor(value);
            }
            var opAttrs = [{ name: 'T', type: _this.binding.TF_ATTR_TYPE, value: typeAttr }];
            _this.binding.executeOp('WriteScalarSummary', opAttrs, _this.getInputTensorIds(inputArgs), 0);
        });
    };
    NodeJSKernelBackend.prototype.writeHistogramSummary = function (resourceHandle, step, name, data, bucketCount, description) {
        var _this = this;
        tidy(function () {
            assert(Number.isInteger(step), function () { return "step is expected to be an integer, but is instead " + step; });
            var content = new messages.HistogramPluginData().setVersion(0);
            var pluginData = new messages.SummaryMetadata.PluginData()
                .setPluginName('histograms')
                .setContent(content.serializeBinary());
            var summary = new messages.SummaryMetadata()
                .setPluginData(pluginData)
                .setDisplayName(null)
                .setSummaryDescription(description);
            var summaryTensor = scalar(summary.serializeBinary(), 'string');
            var nameTensor = scalar(name, 'string');
            var stepScalar = new Int64Scalar(step);
            var buckets = _this.buckets(data, bucketCount);
            assert(buckets.rank === 2 && buckets.shape[1] === 3, function () { return "Expected buckets to have shape [k, 3], but they had shape " + buckets.shape; });
            assert(buckets.dtype === 'float32', function () { return "Expected buckets to have dtype float32, but they had dtype " + buckets.dtype; });
            var inputArgs = [resourceHandle, stepScalar, buckets, nameTensor, summaryTensor];
            var typeAttr = _this.typeAttributeFromTensor(buckets);
            var opAttrs = [{ name: 'T', type: _this.binding.TF_ATTR_TYPE, value: typeAttr }];
            _this.binding.executeOp('WriteSummary', opAttrs, _this.getInputTensorIds(inputArgs), 0);
        });
    };
    NodeJSKernelBackend.prototype.flushSummaryWriter = function (resourceHandle) {
        var inputArgs = [resourceHandle];
        this.executeMultipleOutputs('FlushSummaryWriter', [], inputArgs, 0);
    };
    NodeJSKernelBackend.prototype.buckets = function (data, bucketCount) {
        if (data.size === 0) {
            return tensor([], [0, 3], 'float32');
        }
        bucketCount = bucketCount !== undefined ? bucketCount : 30;
        assert(Number.isInteger(bucketCount) && bucketCount > 0, function () {
            return "Expected bucket count to be a strictly positive integer, but it was " + ("" + bucketCount);
        });
        data = data.flatten();
        data = data.cast('float32');
        var min = data.min();
        var max = data.max();
        var range = max.sub(min);
        var isSingular = range.equal(0).arraySync() !== 0;
        if (isSingular) {
            var center = min;
            var bucketStart = center.sub(0.5);
            var bucketEnd = center.add(0.5);
            var bucketCounts_1 = scalar(data.size, 'float32');
            return tf.concat([bucketStart, bucketEnd, bucketCounts_1]).reshape([1, 3]);
        }
        var bucketWidth = range.div(bucketCount);
        var offsets = data.sub(min);
        var bucketIndices = offsets.floorDiv(bucketWidth).cast('int32');
        var clampedIndices = tf.minimum(bucketIndices, bucketCount - 1).cast('int32');
        var oneHots = tf.oneHot(clampedIndices, bucketCount);
        var bucketCounts = oneHots.sum(0).cast('int32');
        var edges = tf.linspace(min.arraySync(), max.arraySync(), bucketCount + 1);
        edges = tf.concat([edges.slice(0, bucketCount), max.reshape([1])], 0);
        var leftEdges = edges.slice(0, bucketCount);
        var rightEdges = edges.slice(1, bucketCount);
        return tf.stack([leftEdges, rightEdges, bucketCounts.cast('float32')])
            .transpose();
    };
    NodeJSKernelBackend.prototype.memory = function () {
        return { unreliable: true };
    };
    NodeJSKernelBackend.prototype.time = function (f) {
        return __awaiter(this, void 0, void 0, function () {
            var start, elapsed;
            return __generator(this, function (_a) {
                start = process.hrtime();
                f();
                elapsed = process.hrtime(start);
                return [2, { kernelMs: elapsed[0] * 1000 + elapsed[1] / 1000000 }];
            });
        });
    };
    NodeJSKernelBackend.prototype.getNumOfSavedModels = function () {
        return this.binding.getNumOfSavedModels();
    };
    return NodeJSKernelBackend;
}());
class Variable extends Tensor {
    constructor(initialValue, trainable, name, tensorId) {
        super(initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
        this.trainable = trainable;
        this.name = name;
    }
    /**
     * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
     * the same shape and dtype as the old `tf.Tensor`.
     *
     * @param newValue New tensor to be assigned to this variable.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    assign(newValue) {
        if (newValue.dtype !== this.dtype) {
            throw new Error(`dtype of the new value (${newValue.dtype}) and ` +
                `previous value (${this.dtype}) must match`);
        }
        if (!util.arraysEqual(newValue.shape, this.shape)) {
            throw new Error(`shape of the new value (${newValue.shape}) and ` +
                `previous value (${this.shape}) must match`);
        }
        trackerFn().disposeTensor(this);
        this.dataId = newValue.dataId;
        trackerFn().incRef(this, null /* backend */);
    }
    dispose() {
        trackerFn().disposeVariable(this);
        this.isDisposedInternal = true;
    }
}
var path = require("path");
var binary = require('@mapbox/node-pre-gyp');
var bindingPath = binary.find(path.resolve(path.join(__dirname, '/config.json')));
if (!fs.existsSync(bindingPath)) {
    throw new Error("The Node.js native addon module (tfjs_binding.node) can not be found at path: " + String(bindingPath) + ". \nPlease run command " +
        "'npm rebuild @tensorflow/tfjs-node" + (String(bindingPath).indexOf('tfjs-node-gpu') > 0 ? "-gpu" : "") + " --build-addon-from-source' to " +
        "rebuild the native addon module. \nIf you have problem with building the addon module, " +
        "please check https://github.com/tensorflow/tfjs/blob/master/tfjs-node/WINDOWS_TROUBLESHOOTING.md or file an issue.");
}
var bindings = require(bindingPath);
/**
 * @type {Engine}
 */
const ENGINE = getOrMakeEngine();
registerBackend('tensorflow', function () {
    return new NodeJSKernelBackend(bindings, '@tensorflow/tfjs-node');
}, 3);
var success = ENGINE.setBackend('tensorflow');
if (!success) {
    throw new Error("Could not initialize Mini TensorFlow backend.");
}
function registerBackend(name, factory, priority = 1) {
    return ENGINE.registerBackend(name, factory, priority);
}
function nodeBackend() {
    return ENGINE.findBackend('tensorflow');
}
function ones(shape) {
    const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
    return ENGINE.makeTensor(values, shape, 'float32');
}
function tidy(nameOrFn, fn) {
    return ENGINE.tidy(nameOrFn, fn);
}
function makeOnesTypedArray(size, dtype) {
    const array = makeZerosTypedArray(size, dtype);
    for (let i = 0; i < array.length; i++) {
        array[i] = 1;
    }
    return array;
}
module.exports = {
    loadSavedModel: loadSavedModel,
    tensor: tensor
};