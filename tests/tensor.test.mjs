import test from "node:test";
import assert from "node:assert/strict";
import katex from "katex";
import {
  createTensorModel,
  INITIAL_AXES,
  indicesAt,
  offsetOf,
  toggleAxis,
  validSelection,
  updateValue,
  modelFromShape,
} from "../src/model/tensor.ts";
import {
  tensorSpace,
  componentLatex,
  generalComponent,
  sliceLatex,
} from "../src/utils/notation.ts";
import { operations } from "../src/operations/index.ts";
import { reshape, transpose, reduction, dot, matmul, contract1d3d, contract2d3d, doubleContraction, kron, frobeniusNorm, einsumPreset, gradientPreset, matrixVectorDerivative, chainRulePreset } from "../src/operations/core.ts";
import { runOperation } from "../src/operations/run.ts";

const axesFor = (shape) =>
  [0, 1, 2].map((i) => ({
    id: i + 1,
    enabled: i < shape.length,
    length: shape[i] ?? 3,
  }));
test("2 × 3 × 4 model has 24 values with last index varying fastest", () => {
  const model = createTensorModel(INITIAL_AXES);
  assert.equal(model.order, 3);
  assert.deepEqual(model.shape, [2, 3, 4]);
  assert.equal(model.values.length, 24);
  assert.equal(offsetOf([2, 1, 3], model.shape), 14);
  assert.equal(model.values[14], 15);
  assert.deepEqual(indicesAt(14, model.shape), [2, 1, 3]);
});
test("row-major indices round trip at every supported order, including scalar and maximum size", () => {
  for (const shape of [[], [1], [6], [2, 3], [2, 3, 4], [6, 6, 6]]) {
    const model = createTensorModel(axesFor(shape));
    model.values.forEach((_, offset) =>
      assert.equal(offsetOf(indicesAt(offset, shape), shape), offset),
    );
  }
});
test("scalar is one value with no indices, distinct from no selection", () => {
  const model = createTensorModel(axesFor([]));
  assert.equal(model.order, 0);
  assert.deepEqual(model.values, [1]);
  assert.equal(offsetOf([], []), 0);
  assert.deepEqual(validSelection({ component: [], slice: null }, model), {
    component: [],
    slice: null,
  });
});
test("turning off an axis disables its following axes and cannot create gaps", () => {
  const axes = toggleAxis(INITIAL_AXES, 0, false);
  assert.deepEqual(
    axes.map((a) => a.enabled),
    [false, false, false],
  );
  assert.deepEqual(toggleAxis(axes, 2, true), axes);
  const one = toggleAxis(axes, 0, true);
  assert.deepEqual(
    one.map((a) => a.enabled),
    [true, false, false],
  );
  assert.deepEqual(
    toggleAxis(toggleAxis(one, 1, true), 2, true).map((a) => a.enabled),
    [true, true, true],
  );
  assert.deepEqual(
    toggleAxis(INITIAL_AXES, 1, false).map((a) => a.enabled),
    [true, false, false],
  );
});
test("dimensions validate integers 1 through 6 and contiguous axes", () => {
  for (const bad of [0, 7, 2.5, NaN, Infinity])
    assert.throws(() => createTensorModel(axesFor([bad])), RangeError);
  assert.throws(
    () =>
      createTensorModel([
        { id: 1, enabled: false, length: 2 },
        ...INITIAL_AXES.slice(1),
      ]),
    RangeError,
  );
  assert.throws(() => offsetOf([0, 2], [2, 3]), RangeError);
  assert.throws(() => offsetOf([2, 4], [2, 3]), RangeError);
});
test("out-of-range selections are cleared on shrink; valid selections survive", () => {
  const model = createTensorModel(axesFor([2, 2, 2]));
  assert.deepEqual(validSelection({ component: [2, 2, 2], slice: 2 }, model), {
    component: [2, 2, 2],
    slice: 2,
  });
  assert.deepEqual(validSelection({ component: [2, 3, 4], slice: 2 }, model), {
    component: null,
    slice: 2,
  });
  assert.deepEqual(validSelection({ component: [3, 1, 1], slice: 3 }, model), {
    component: null,
    slice: null,
  });
  assert.deepEqual(
    validSelection(
      { component: [2, 1, 1], slice: 2 },
      createTensorModel(axesFor([2, 2])),
    ),
    { component: null, slice: null },
  );
});
test("numeric edits are immutable, finite, and preserved by coordinate during resize", () => {
  const original = createTensorModel(INITIAL_AXES);
  const edited = updateValue(original, [2, 1, 3], -7.5);
  assert.equal(original.values[14], 15);
  assert.equal(edited.values[14], -7.5);
  const resized = createTensorModel(axesFor([2, 4, 5]), edited);
  assert.equal(resized.values[offsetOf([2, 1, 3], resized.shape)], -7.5);
  assert.equal(updateValue(edited, [2, 1, 3], NaN), edited);
});
test("all abstract, concrete, component and slice notation render in KaTeX", () => {
  for (const shape of [[], [3], [2, 3], [2, 3, 4]]) {
    const model = createTensorModel(axesFor(shape));
    for (const latex of [
      tensorSpace(model),
      tensorSpace(model, true),
      generalComponent(model),
      componentLatex(model, shape),
      componentLatex(model, shape, true),
      ...(shape.length === 3 ? [sliceLatex(model, 2)] : []),
    ]) {
      assert.doesNotThrow(() =>
        katex.renderToString(latex, { throwOnError: true, strict: "error" }),
      );
    }
  }
  assert.equal(
    componentLatex(createTensorModel(INITIAL_AXES), [2, 1, 3]),
    "X_{213}",
  );
});
test("all registered operations are executable learning modules", () => {
  assert.ok(operations.every((op) => op.available));
  assert.equal(operations.length, 20);
  assert.equal(new Set(operations.map((op) => op.id)).size, operations.length);
});
test("representative chapter operations compute exact shapes and values", () => {
  const base = createTensorModel(INITIAL_AXES);
  assert.deepEqual(reshape(base, [3, 2, 4]).result.shape, [3, 2, 4]);
  assert.deepEqual(transpose(base, [0, 2, 1]).result.shape, [2, 4, 3]);
  assert.deepEqual(reduction(base, 0).result.shape, [3, 4]);
  assert.equal(dot().result.values[0], 32);
  assert.deepEqual(matmul().result.values, [19, 22, 43, 50]);
  assert.deepEqual(contract1d3d().result.shape, [3, 4]);
  assert.deepEqual(contract2d3d().result.shape, [2, 2, 3]);
  assert.deepEqual(doubleContraction().result.shape, [2, 2]);
  assert.deepEqual(kron().result.shape, [4, 4]);
  assert.equal(frobeniusNorm().result.values[0], Math.sqrt(30));
  assert.deepEqual(einsumPreset("ij,jk->ik").result.values, [19, 22, 43, 50]);
  assert.deepEqual(gradientPreset(modelFromShape([2,2], [1,2,3,4], "X")).result.values, [2,4,6,8]);
  assert.deepEqual(matrixVectorDerivative().result.shape, [3,2,2]);
  assert.deepEqual(chainRulePreset().result.shape, [2,2,2]);
});
test("index notation can represent a future shared Greek index without invalid TeX", () => {
  const model = createTensorModel(INITIAL_AXES);
  model.notation.indices = ["\\alpha", "i", "j"];
  assert.doesNotThrow(() =>
    katex.renderToString(generalComponent(model), { throwOnError: true }),
  );
});
test("every simulator operation produces a finite result and valid KaTeX", () => {
  const source = createTensorModel(INITIAL_AXES);
  for (const operation of operations.filter((item) => item.id !== "explore" && item.id !== "slice")) {
    const output = runOperation(operation.id, source, { targetShape: "3,2,4", axis: 0, fixedIndex: 1, permutation: "1,3,2", einsum: "a,amn->mn" });
    assert.ok(output.result.values.every(Number.isFinite), operation.id);
    assert.ok(output.events.length >= 2, operation.id);
    assert.doesNotThrow(() => katex.renderToString(output.formula, { throwOnError: true, strict: "error" }), operation.id);
  }
  assert.throws(() => reshape(source, [2, 2]), /Reshape needs/);
  assert.throws(() => einsumPreset("bad->notation"), /supported preset/);
});
