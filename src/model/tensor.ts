import type {
  AxisMetadata,
  TensorCell,
  TensorModel,
  TensorOrder,
  TensorSelection,
} from "../types/index.ts";
import { notationFor } from "../utils/notation.ts";

export const INITIAL_AXES: AxisMetadata[] = [
  { id: 1, enabled: true, length: 2 },
  { id: 2, enabled: true, length: 3 },
  { id: 3, enabled: true, length: 4 },
];
export const EMPTY_SELECTION: TensorSelection = {
  component: null,
  slice: null,
};

export function modelFromShape(
  shape: number[],
  values?: number[],
  symbolicName?: string,
): TensorModel {
  if (shape.length > 3 || shape.some((length) => !Number.isInteger(length) || length < 1 || length > 6))
    throw new RangeError("Tensor shape supports one to three dimensions, each 1 through 6");
  const count = shape.reduce((size, length) => size * length, 1);
  if (values && values.length !== count) throw new RangeError("Value count does not match tensor shape");
  const axes: AxisMetadata[] = [0, 1, 2].map((index) => ({ id: index + 1, enabled: index < shape.length, length: shape[index] ?? 1 }));
  const order = shape.length as TensorOrder;
  const notation = notationFor(order);
  return { order, shape: [...shape], axes, values: values ? [...values] : Array.from({ length: count }, (_, i) => i + 1), symbolicName: symbolicName ?? notation.symbolicName, notation: notation.notation };
}

export function indicesAt(offset: number, shape: number[]): number[] {
  const indices = Array<number>(shape.length);
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    indices[axis] = (offset % shape[axis]) + 1;
    offset = Math.floor(offset / shape[axis]);
  }
  return indices;
}
export function isValidComponent(indices: number[], shape: number[]): boolean {
  return (
    indices.length === shape.length &&
    indices.every(
      (index, axis) =>
        Number.isInteger(index) && index >= 1 && index <= shape[axis],
    )
  );
}
export function offsetOf(indices: number[], shape: number[]): number {
  if (!isValidComponent(indices, shape))
    throw new RangeError("Component indices are outside the tensor shape");
  return indices.reduce(
    (offset, index, axis) => offset * shape[axis] + index - 1,
    0,
  );
}
export function createTensorModel(
  axes: AxisMetadata[],
  previous?: TensorModel,
): TensorModel {
  if (axes.length !== 3)
    throw new RangeError("Exactly three axis controls are required");
  let inactive = false;
  for (const axis of axes) {
    if (!Number.isInteger(axis.length) || axis.length < 1 || axis.length > 6)
      throw new RangeError("Axis length must be an integer from 1 to 6");
    if (inactive && axis.enabled)
      throw new RangeError("Axes must be enabled in order");
    if (!axis.enabled) inactive = true;
  }
  const shape = axes.filter((axis) => axis.enabled).map((axis) => axis.length);
  const count = shape.reduce((size, length) => size * length, 1);
  const values = Array.from({ length: count }, (_, offset) => {
    const indices = indicesAt(offset, shape);
    return previous && isValidComponent(indices, previous.shape)
      ? previous.values[offsetOf(indices, previous.shape)]
      : offset + 1;
  });
  const model = modelFromShape(shape, values);
  return { ...model, axes: axes.map((axis) => ({ ...axis })) };
}
export function toggleAxis(
  axes: AxisMetadata[],
  index: number,
  enabled: boolean,
): AxisMetadata[] {
  if (index < 0 || index > 2) return axes;
  if (enabled && index > 0 && !axes[index - 1].enabled) return axes;
  return axes.map((axis, i) => ({
    ...axis,
    enabled:
      i === index ? enabled : !enabled && i > index ? false : axis.enabled,
  }));
}
export function tensorCells(model: TensorModel): TensorCell[] {
  return model.values.map((value, offset) => ({
    indices: indicesAt(offset, model.shape),
    offset,
    value,
  }));
}
export function validSelection(
  selection: TensorSelection,
  model: TensorModel,
): TensorSelection {
  return {
    component:
      selection.component !== null &&
      isValidComponent(selection.component, model.shape)
        ? selection.component
        : null,
    slice:
      model.order === 3 &&
      selection.slice !== null &&
      Number.isInteger(selection.slice) &&
      selection.slice >= 1 &&
      selection.slice <= model.shape[0]
        ? selection.slice
        : null,
  };
}
export function updateValue(
  model: TensorModel,
  indices: number[],
  value: number,
): TensorModel {
  if (!Number.isFinite(value)) return model;
  const values = [...model.values];
  values[offsetOf(indices, model.shape)] = value;
  return { ...model, values };
}
