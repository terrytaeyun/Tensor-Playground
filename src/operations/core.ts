import { modelFromShape, indicesAt, offsetOf } from "../model/tensor.ts";
import type { AnimationEvent, OperationResult, TensorModel } from "../types/index.ts";

const events = (formula: string, extra: AnimationEvent[] = []): AnimationEvent[] => [
  { kind: "highlight", label: "Choose compatible components" },
  { kind: "extract", label: "Extract indexed entries" },
  { kind: "moveToFormula", label: "Assemble symbolic formula", latex: formula },
  ...extra,
  { kind: "createResult", label: "Create result tensor" },
];
const result = (name: string, model: TensorModel, inputs: TensorModel[], formula: string, detail: string, numericFormula?: string, extra?: AnimationEvent[]): OperationResult => ({ name, result: model, inputs, formula, detail, numericFormula: numericFormula ?? `\\mathrm{result}=(${model.values.join(",")})`, events: events(formula, extra) });
const count = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
const asMatrix = (shape: number[], values: number[], name: string) => modelFromShape(shape, values, name);
const sample = (shape: number[], name: string, start = 1) => modelFromShape(shape, Array.from({ length: count(shape) }, (_, i) => start + i), name);

export function reshape(source: TensorModel, target: number[]): OperationResult {
  if (!target.length || target.length > 3 || target.some(x => !Number.isInteger(x) || x < 1 || x > 6)) throw new RangeError("Target shape uses one to three lengths from 1 to 6");
  if (count(target) !== source.values.length) throw new RangeError(`Reshape needs ${source.values.length} entries; target has ${count(target)}`);
  return result("Reshape", modelFromShape(target, source.values, source.symbolicName), [source], `${source.symbolicName}_{\\mathrm{flat}}\\rightarrow ${source.symbolicName}_{\\mathrm{reshape}}`, `Row-major flatten: (${source.shape.join(",")}) → (${target.join(",")})`, undefined, [{ kind: "extract", label: "Flatten values in row-major order" }, { kind: "moveToFormula", label: "Place values into the target shape" }]);
}

export function transpose(source: TensorModel, permutation: number[]): OperationResult {
  if (permutation.length !== source.order || new Set(permutation).size !== source.order || permutation.some(x => !Number.isInteger(x) || x < 0 || x >= source.order)) throw new RangeError("Axis permutation must contain each active axis once");
  const shape = permutation.map(axis => source.shape[axis]);
  const values = Array.from({ length: source.values.length }, (_, output) => {
    const outputIndex = indicesAt(output, shape);
    const inputIndex = Array(source.order);
    permutation.forEach((axis, i) => { inputIndex[axis] = outputIndex[i]; });
    return source.values[offsetOf(inputIndex, source.shape)];
  });
  const oldIndices = source.notation.indices.join(" ");
  const newIndices = permutation.map(axis => source.notation.indices[axis]).join(" ");
  return result("Transpose", modelFromShape(shape, values, source.symbolicName), [source], `${source.symbolicName}_{${oldIndices}}\rightarrow ${source.symbolicName}_{${newIndices}}`, `Axis order ${permutation.map(x => x + 1).join(" → ")}`, undefined, [{ kind: "highlight", label: "Highlight axis labels" }, { kind: "moveToFormula", label: "Permute index labels" }]);
}

export function slice(source: TensorModel, axis: number, index: number): OperationResult {
  if (axis < 0 || axis >= source.order || !Number.isInteger(index) || index < 1 || index > source.shape[axis]) throw new RangeError("Slice axis or fixed index is outside the tensor shape");
  const shape = source.shape.filter((_, i) => i !== axis);
  const values = Array.from({ length: count(shape) }, (_, output) => {
    const outputIndex = indicesAt(output, shape); outputIndex.splice(axis, 0, index);
    return source.values[offsetOf(outputIndex, source.shape)];
  });
  const indices = source.notation.indices.map((name, i) => i === axis ? index : name).join(",");
  return result("Slice", modelFromShape(shape, values, source.symbolicName), [source], `${source.symbolicName}_{${indices}}`, `Fixed Axis ${axis + 1} at index ${index}`, undefined, [{ kind: "highlight", label: `Highlight Axis ${axis + 1} = ${index}` }, { kind: "extract", label: "Extract the selected plane" }]);
}

export function reduction(source: TensorModel, axis: number): OperationResult {
  if (axis < 0 || axis >= source.order) throw new RangeError("Reduction axis is outside the tensor shape");
  const shape = source.shape.filter((_, i) => i !== axis);
  const values = Array.from({ length: count(shape) }, (_, output) => {
    const kept = indicesAt(output, shape); let total = 0;
    for (let i = 1; i <= source.shape[axis]; i++) { const index = [...kept]; index.splice(axis, 0, i); total += source.values[offsetOf(index, source.shape)]; }
    return total;
  });
  const retained = source.notation.indices.filter((_, i) => i !== axis).join(" ");
  const summed = source.notation.indices[axis];
  return result("Reduction", modelFromShape(shape, values, source.symbolicName), [source], `${source.symbolicName}_{${retained}}=\\sum_{${summed}}${source.symbolicName}_{${source.notation.indices.join(" ")}}`, `Sum over Axis ${axis + 1}`, undefined, [{ kind: "sum", label: `Sum over ${summed}` }, { kind: "contractIndex", label: `Remove ${summed} from the result` }]);
}

export function hadamard(a: TensorModel, b = sample(a.shape, "B", 2)): OperationResult {
  if (a.shape.join(",") !== b.shape.join(",")) throw new RangeError("Hadamard product requires equal shapes");
  return result("Hadamard Product", modelFromShape(a.shape, a.values.map((value, i) => value * b.values[i]), "C"), [a, b], `C_{${a.notation.indices.join(" ")}}=A_{${a.notation.indices.join(" ")}}B_{${a.notation.indices.join(" ")}}`, "Elementwise multiplication", undefined, [{ kind: "pair", label: "Pair matching positions" }, { kind: "multiply", label: "Multiply each pair" }]);
}

export function broadcast(source: TensorModel, target: number[]): OperationResult {
  if (!target.length || target.length > 3 || target.length < source.order) throw new RangeError("Broadcast target must retain all source dimensions");
  const padded = Array(target.length - source.order).fill(1).concat(source.shape);
  if (padded.some((length, i) => length !== 1 && length !== target[i])) throw new RangeError("A source axis must equal its target axis or be length 1");
  const values = Array.from({ length: count(target) }, (_, output) => {
    const outputIndex = indicesAt(output, target);
    const sourceIndex = outputIndex.slice(target.length - source.order).map((value, i) => source.shape[i] === 1 ? 1 : value);
    return source.values[offsetOf(sourceIndex, source.shape)];
  });
  return result("Broadcast", modelFromShape(target, values, source.symbolicName), [source], `${source.symbolicName}^{${source.shape.join("\\times")}}\rightsquigarrow ${source.symbolicName}^{${target.join("\\times")}}`, `Repeat singleton axes: ${padded.map((x, i) => x === 1 && target[i] > 1 ? i + 1 : null).filter(Boolean).join(", ") || "none"}`, undefined, [{ kind: "stackSlice", label: "Repeat singleton axes" }]);
}

export function dot(a = sample([3], "a"), b = sample([3], "b", 4)): OperationResult {
  if (a.order !== 1 || b.order !== 1 || a.shape[0] !== b.shape[0]) throw new RangeError("Dot product requires equal-length vectors");
  const value = a.values.reduce((total, current, i) => total + current * b.values[i], 0);
  return result("Dot Product", modelFromShape([], [value], "c"), [a, b], `c=\\sum_i a_i b_i`, "Vector dot product", `${a.values.map((x, i) => `${x}\\cdot${b.values[i]}`).join("+")}=${value}`, [{ kind: "pair", label: "Pair aᵢ with bᵢ" }, { kind: "multiply", label: "Multiply each pair" }, { kind: "sum", label: "Sum products" }, { kind: "contractIndex", label: "Contract i" }]);
}

export function matmul(a = sample([2, 2], "A"), b = sample([2, 2], "B", 5)): OperationResult {
  if (a.order !== 2 || b.order !== 2 || a.shape[1] !== b.shape[0]) throw new RangeError("Matrix multiplication needs A(m,r) and B(r,n)");
  const [m, r] = a.shape; const n = b.shape[1]; const values: number[] = [];
  for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) { let sum = 0; for (let alpha = 1; alpha <= r; alpha++) sum += a.values[offsetOf([i, alpha], a.shape)] * b.values[offsetOf([alpha, j], b.shape)]; values.push(sum); }
  return result("Matrix Multiplication", asMatrix([m, n], values, "C"), [a, b], `C_{ij}=\\sum_{\\alpha}A_{i\\alpha}B_{\\alpha j}`, "Shared index α contracts between row i and column j", undefined, [{ kind: "highlight", label: "Highlight A row and B column" }, { kind: "pair", label: "Pair shared α" }, { kind: "multiply", label: "Multiply AᵢαBαⱼ" }, { kind: "sum", label: "Sum over α" }, { kind: "contractIndex", label: "Remove α" }]);
}

export function contract1d3d(vector = sample([2], "v", 2), tensor = sample([2, 3, 4], "X", 3)): OperationResult {
  if (vector.order !== 1 || tensor.order !== 3 || vector.shape[0] !== tensor.shape[0]) throw new RangeError("1D × 3D contraction needs v(r) and X(r,m,n)");
  const [, m, n] = tensor.shape; const values: number[] = [];
  for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) { let sum = 0; for (let alpha = 1; alpha <= vector.shape[0]; alpha++) sum += vector.values[alpha - 1] * tensor.values[offsetOf([alpha, i, j], tensor.shape)]; values.push(sum); }
  const first = vector.values.map((value, alpha) => `${value}\\cdot${tensor.values[offsetOf([alpha + 1, 1, 1], tensor.shape)]}`).join("+");
  return result("1D × 3D Contraction", modelFromShape([m, n], values, "C"), [vector, tensor], `C_{ij}=\\sum_{\\alpha}v_{\\alpha}X_{\\alpha ij}`, "Component view: matching (i,j) entries across every matrix slice", `C_{11}=${first}=${values[0]}`, [{ kind: "stackSlice", label: "Keep the same (i,j) across slices" }, { kind: "pair", label: "Pair vα with Xαij" }, { kind: "multiply", label: "Weight each matrix slice" }, { kind: "sum", label: "Sum weighted terms" }, { kind: "contractIndex", label: "Contract α" }]);
}

export function contract2d3d(matrix = sample([2, 2], "A", 1), tensor = sample([2, 2, 3], "X", 3)): OperationResult {
  if (matrix.order !== 2 || tensor.order !== 3 || matrix.shape[1] !== tensor.shape[0]) throw new RangeError("2D × 3D contraction needs A(p,r) and X(r,m,n)");
  const [p, r] = matrix.shape; const [, m, n] = tensor.shape; const values: number[] = [];
  for (let i = 1; i <= p; i++) for (let j = 1; j <= m; j++) for (let k = 1; k <= n; k++) { let sum = 0; for (let alpha = 1; alpha <= r; alpha++) sum += matrix.values[offsetOf([i, alpha], matrix.shape)] * tensor.values[offsetOf([alpha, j, k], tensor.shape)]; values.push(sum); }
  return result("2D × 3D Contraction", modelFromShape([p, m, n], values, "C"), [matrix, tensor], `C_{ijk}=\\sum_{\\alpha}A_{i\\alpha}X_{\\alpha jk}`, "Each output slice i reuses the 1D × 3D contraction engine", undefined, [{ kind: "highlight", label: "Highlight Aᵢ,: and tensor slices" }, { kind: "pair", label: "Pair shared α" }, { kind: "sum", label: "Sum weighted slices" }, { kind: "contractIndex", label: "Contract α" }, { kind: "stackSlice", label: "Stack output slices i" }]);
}

export function doubleContraction(x = sample([2, 2, 2], "X", 1), y = sample([2, 2, 2], "Y", 2)): OperationResult {
  if (x.order !== 3 || y.order !== 3 || x.shape[1] !== y.shape[0] || x.shape[2] !== y.shape[1]) throw new RangeError("Double contraction needs X(r,m,n) and Y(m,n,q)");
  const [r, m, n] = x.shape; const q = y.shape[2]; const values: number[] = [];
  for (let i = 1; i <= r; i++) for (let l = 1; l <= q; l++) { let sum = 0; for (let j = 1; j <= m; j++) for (let k = 1; k <= n; k++) sum += x.values[offsetOf([i, j, k], x.shape)] * y.values[offsetOf([j, k, l], y.shape)]; values.push(sum); }
  return result("Double Contraction", modelFromShape([r, q], values, "C"), [x, y], `C_{il}=\\sum_j\\sum_kX_{ijk}Y_{jkl}`, "Shared indices j and k both disappear from the result", undefined, [{ kind: "pair", label: "Pair components sharing j,k" }, { kind: "multiply", label: "Multiply paired terms" }, { kind: "sum", label: "Double sum over j,k" }, { kind: "contractIndex", label: "Remove j and k" }]);
}

export function kron(a = sample([2, 2], "A"), b = sample([2, 2], "B", 5)): OperationResult {
  if (a.order !== 2 || b.order !== 2) throw new RangeError("Kronecker product currently uses two matrices");
  const [m, n] = a.shape; const [p, q] = b.shape; const values: number[] = [];
  for (let i = 1; i <= m; i++) for (let u = 1; u <= p; u++) for (let j = 1; j <= n; j++) for (let v = 1; v <= q; v++) values.push(a.values[offsetOf([i, j], a.shape)] * b.values[offsetOf([u, v], b.shape)]);
  return result("Kronecker Product", modelFromShape([m * p, n * q], values, "K"), [a, b], `K=A\\otimes B,\\qquad K_{(i,u),(j,v)}=A_{ij}B_{uv}`, "Each Aᵢⱼ expands into a block AᵢⱼB", undefined, [{ kind: "highlight", label: "Choose a matrix block" }, { kind: "multiply", label: "Scale B by Aᵢⱼ" }, { kind: "stackSlice", label: "Place blocks into K" }]);
}

export function frobeniusInner(a = sample([2, 2], "A"), b = sample([2, 2], "B", 5)): OperationResult {
  if (a.shape.join(",") !== b.shape.join(",")) throw new RangeError("Frobenius inner product requires equal shapes");
  const value = a.values.reduce((sum, x, i) => sum + x * b.values[i], 0);
  return result("Frobenius Inner Product", modelFromShape([], [value], "f"), [a, b], `\\langle A,B\\rangle_F=\\sum_{ij}a_{ij}b_{ij}=\\operatorname{tr}(A^TB)`, "Elementwise product then sum", `${a.values.map((x, i) => `${x}\\cdot${b.values[i]}`).join("+")}=${value}`, [{ kind: "pair", label: "Pair matching cells" }, { kind: "multiply", label: "Form elementwise product" }, { kind: "sum", label: "Sum all cells" }]);
}

export function frobeniusNorm(a = sample([2, 2], "A")): OperationResult {
  if (a.order !== 2) throw new RangeError("Frobenius norm currently uses a matrix");
  const squared = a.values.reduce((sum, x) => sum + x * x, 0); const value = Math.sqrt(squared);
  return result("Frobenius Norm", modelFromShape([], [value], "n"), [a], `\\lVert A\\rVert_F=\\sqrt{\\sum_{ij}a_{ij}^2}`, "Square each cell, sum, then take the square root", `\\sqrt{${a.values.map(x => `${x}^2`).join("+")}}=${value}`, [{ kind: "multiply", label: "Square each cell" }, { kind: "sum", label: "Sum squares" }, { kind: "createResult", label: "Take square root" }]);
}

export function gradientPreset(x = sample([2, 2], "X")): OperationResult {
  if (x.order !== 2) throw new RangeError("Gradient preset requires a matrix X");
  return result("Matrix → Scalar Gradient", modelFromShape(x.shape, x.values.map(value => 2 * value), "∇f"), [x], `f(X)=\\lVert X\\rVert_F^2,\\qquad \\nabla_Xf=2X`, "Every ∂f/∂xᵢⱼ enters the matching gradient cell", undefined, [{ kind: "highlight", label: "Choose Xᵢⱼ" }, { kind: "extract", label: "Differentiate  xᵢⱼ²" }, { kind: "multiply", label: "Apply factor 2" }]);
}

export function directionalDerivative(x = sample([2, 2], "X"), direction = sample([2, 2], "V", 2)): OperationResult {
  if (x.shape.join(",") !== direction.shape.join(",") || x.order !== 2) throw new RangeError("Directional derivative preset uses equal-size matrices");
  const gradient = x.values.map(value => 2 * value); const value = gradient.reduce((sum, current, i) => sum + current * direction.values[i], 0);
  return result("Directional Derivative", modelFromShape([], [value], "D"), [x, direction], `D_Vf(X)=\\langle\\nabla_Xf(X),V\\rangle_F=\\langle2X,V\\rangle_F`, "Gradient followed by the Frobenius inner product", undefined, [{ kind: "extract", label: "Create ∇ₓf = 2X" }, { kind: "pair", label: "Pair gradient with V" }, { kind: "multiply", label: "Multiply pairs" }, { kind: "sum", label: "Frobenius sum" }]);
}

export function matrixVectorDerivative(x = sample([2, 2], "X")): OperationResult {
  if (x.shape.join(",") !== "2,2") throw new RangeError("Matrix → vector derivative preset currently requires a 2 × 2 matrix");
  const [a, b, c, d] = x.values;
  // ∇tr(X), ∇||X||², and ∇det(X), stacked along the output axis.
  const values = [1, 0, 0, 1, 2*a, 2*b, 2*c, 2*d, d, -c, -b, a];
  return result("Matrix → Vector Derivative", modelFromShape([3, 2, 2], values, "∂f/∂X"), [x], `f(X)=[\\operatorname{tr}(X),\\lVert X\\rVert_F^2,\\det(X)]^T,\\quad\\frac{\\partial f}{\\partial X}\\in\\mathbb R^{3\\times2\\times2}`, "Three gradient matrices are stacked as output slices", undefined, [{ kind: "extract", label: "Differentiate f₁, f₂, f₃" }, { kind: "stackSlice", label: "Stack gradient matrices" }]);
}

export function chainRulePreset(): OperationResult {
  const c = sample([2, 2], "C", 1); const dyDx = sample([2, 2, 2], "∂y/∂X", 1);
  const output = contract2d3d(c, dyDx);
  return { ...output, name: "Tensor Chain Rule", formula: `\\frac{\\partial z}{\\partial X}=\\frac{\\partial z}{\\partial y}\\cdot\\frac{\\partial y}{\\partial X},\\qquad(\\partial z/\\partial X)_{kij}=\\sum_{\\alpha}(\\partial z_k/\\partial y_\\alpha)(\\partial y_\\alpha/\\partial x_{ij})`, detail: "Preset y = Xa + b, z = Cy; reuse the 2D × 3D contraction", events: [{ kind: "highlight", label: "Trace X → y → z" }, ...output.events] };
}

export function einsumPreset(expression: string): OperationResult {
  const normalized = expression.replace(/\\s/g, "");
  if (normalized === "i,i->") return dot();
  if (normalized === "ij,jk->ik") return matmul();
  if (normalized === "a,amn->mn") return contract1d3d();
  if (normalized === "pr,rmn->pmn") return contract2d3d();
  if (normalized === "ijk,jkl->il") return doubleContraction();
  throw new RangeError("Use a supported preset: i,i-> · ij,jk->ik · a,amn->mn · pr,rmn->pmn · ijk,jkl->il");
}
