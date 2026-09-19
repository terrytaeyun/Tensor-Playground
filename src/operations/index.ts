import type { OperationDefinition } from "../types/index.ts";
import { exploreOperation } from "./explore.ts";
import { sliceOperation } from "./slice.ts";
const available = (id: string, label: string, minimumOrder?: 0 | 1 | 2 | 3): OperationDefinition => ({ id, label, available: true, interaction: "component", minimumOrder });
export const operations: OperationDefinition[] = [
  exploreOperation,
  available("reshape", "Reshape", 1),
  available("transpose", "Transpose", 2),
  sliceOperation,
  available("reduction", "Reduction", 1),
  available("hadamard", "Hadamard Product", 1),
  available("broadcast", "Broadcasting", 0),
  available("dot", "Dot Product"),
  available("matmul", "Matrix Multiplication"),
  available("contraction-1d3d", "1D × 3D Contraction"),
  available("contraction-2d3d", "2D × 3D Contraction"),
  available("double-contraction", "Double Contraction"),
  available("einsum", "Einstein Summation"),
  available("kronecker", "Kronecker Product"),
  available("frobenius-inner", "Frobenius Inner Product"),
  available("frobenius-norm", "Frobenius Norm"),
  available("gradient", "Matrix → Scalar Gradient"),
  available("directional-derivative", "Directional Derivative"),
  available("matrix-vector-derivative", "Matrix → Vector Derivative"),
  available("tensor-chain-rule", "Tensor Chain Rule"),
];
export function getOperation(id: string): OperationDefinition {
  return (
    operations.find((operation) => operation.id === id) ?? exploreOperation
  );
}
