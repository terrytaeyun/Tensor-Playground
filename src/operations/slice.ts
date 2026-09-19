import type { OperationDefinition } from "../types/index.ts";
export const sliceOperation: OperationDefinition = {
  id: "slice",
  label: "Slice",
  available: true,
  interaction: "slice",
  minimumOrder: 3,
};
