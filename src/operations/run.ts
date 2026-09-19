import { modelFromShape } from "../model/tensor.ts";
import type { OperationResult, TensorModel } from "../types/index.ts";
import { broadcast, chainRulePreset, contract1d3d, contract2d3d, directionalDerivative, dot, doubleContraction, einsumPreset, frobeniusInner, frobeniusNorm, gradientPreset, hadamard, kron, matmul, matrixVectorDerivative, reduction, reshape, slice, transpose } from "./core.ts";

const defaultTarget = (model: TensorModel) => model.order === 3 ? [3, 2, 4] : model.shape;
export function runOperation(id: string, source: TensorModel, options: { targetShape?: string; axis?: number; fixedIndex?: number; permutation?: string; einsum?: string }): OperationResult {
  const target = (options.targetShape ?? "").split(",").map(value => Number(value.trim())).filter(Boolean);
  const axis = options.axis ?? 0;
  switch (id) {
    case "reshape": return reshape(source, target.length ? target : defaultTarget(source));
    case "transpose": { const permutation = (options.permutation ?? "").split(",").map(value => Number(value.trim()) - 1).filter(value => value >= 0); return transpose(source, permutation.length ? permutation : Array.from({ length: source.order }, (_, i) => source.order - 1 - i)); }
    case "slice": return slice(source, axis, options.fixedIndex ?? 1);
    case "reduction": return reduction(source, axis);
    case "hadamard": return hadamard(source);
    case "broadcast": return broadcast(modelFromShape([1], [source.values[0]], "s"), target.length ? target : source.shape);
    case "dot": return dot();
    case "matmul": return matmul();
    case "contraction-1d3d": return contract1d3d();
    case "contraction-2d3d": return contract2d3d();
    case "double-contraction": return doubleContraction();
    case "einsum": return einsumPreset(options.einsum ?? "a,amn->mn");
    case "kronecker": return kron();
    case "frobenius-inner": return frobeniusInner();
    case "frobenius-norm": return frobeniusNorm();
    case "gradient": return gradientPreset(source.order === 2 ? source : modelFromShape([2, 2], [1, 2, 3, 4], "X"));
    case "directional-derivative": return directionalDerivative();
    case "matrix-vector-derivative": return matrixVectorDerivative();
    case "tensor-chain-rule": return chainRulePreset();
    default: throw new RangeError("Choose an operation to run");
  }
}
