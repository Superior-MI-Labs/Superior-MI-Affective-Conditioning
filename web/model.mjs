// Read-only inference for weights exported by src/vad/model.py.
// Training is implemented in PyTorch only; the browser never learns weights.

export const AXES = Object.freeze(['Valence', 'Arousal', 'Dominance']);

export function checkVAD(values) {
  if (!Array.isArray(values) || values.length !== 3 ||
      values.some(value => typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1)) {
    throw new RangeError('VAD requires three finite values between -1 and +1.');
  }
  return values;
}

export function checkModel(model) {
  if (model?.schema_version !== 1 || model.activation !== 'tanh' ||
      JSON.stringify(model.architecture) !== '[6,16,3]' || !Array.isArray(model.layers) || model.layers.length !== 2) {
    throw new Error('Unsupported VAD checkpoint format.');
  }
  for (const [i, [out, input]] of [[16, 6], [3, 16]].entries()) {
    const layer = model.layers[i];
    if (!Array.isArray(layer.weights) || layer.weights.length !== out ||
        !Array.isArray(layer.bias) || layer.bias.length !== out ||
        layer.weights.some(row => !Array.isArray(row) || row.length !== input ||
          row.some(v => typeof v !== 'number' || !Number.isFinite(v))) ||
        layer.bias.some(v => typeof v !== 'number' || !Number.isFinite(v))) {
      throw new Error('Invalid exported neural weights.');
    }
  }
  return model;
}

export function infer(model, current, observed) {
  checkVAD(current);
  checkVAD(observed);
  let values = [...current, ...observed];
  for (const layer of checkModel(model).layers) {
    values = layer.weights.map((row, i) => Math.tanh(
      row.reduce((sum, weight, j) => sum + weight * values[j], layer.bias[i])
    ));
  }
  return values;
}

export function stateDelta(predicted, current) {
  checkVAD(predicted);
  checkVAD(current);
  return predicted.map((value, i) => value - current[i]);
}
