import type {
  IndexNotation,
  TensorModel,
  TensorOrder,
} from "../types/index.ts";

const NOTATION: Record<TensorOrder, { name: string; notation: IndexNotation }> =
  {
    0: {
      name: "x",
      notation: { indices: [], dimensions: [], componentName: "x" },
    },
    1: {
      name: "x",
      notation: { indices: ["i"], dimensions: ["r"], componentName: "x" },
    },
    2: {
      name: "A",
      notation: {
        indices: ["i", "j"],
        dimensions: ["m", "n"],
        componentName: "a",
      },
    },
    3: {
      name: "X",
      notation: {
        indices: ["i", "j", "k"],
        dimensions: ["r", "m", "n"],
        componentName: "x",
      },
    },
  };

export function notationFor(order: TensorOrder) {
  const entry = NOTATION[order];
  return {
    symbolicName: entry.name,
    notation: {
      ...entry.notation,
      indices: [...entry.notation.indices],
      dimensions: [...entry.notation.dimensions],
    },
  };
}
export function tensorSpace(model: TensorModel, abstract = false): string {
  const dimensions = abstract ? model.notation.dimensions : model.shape;
  return `${model.symbolicName}\\in\\mathbb{R}${dimensions.length ? `^{${dimensions.join("\\times ")}}` : ""}`;
}
export function generalComponent(model: TensorModel): string {
  return (
    model.symbolicName +
    (model.order ? `_{${model.notation.indices.join(" ")}}` : "")
  );
}
export function componentLatex(
  model: TensorModel,
  indices: number[],
  cellLabel = false,
): string {
  const name = cellLabel ? model.notation.componentName : model.symbolicName;
  return name + (indices.length ? `_{${indices.join("")}}` : "");
}
export function sliceLatex(model: TensorModel, slice: number): string {
  return `${model.symbolicName}_{${slice},:,:}`;
}
export function shapeText(model: TensorModel): string {
  return `Order ${model.order} · Shape (${model.shape.join(", ")})`;
}
export function tensorKind(order: TensorOrder): string {
  return ["Scalar", "Vector", "Matrix", "3D tensor"][order];
}
