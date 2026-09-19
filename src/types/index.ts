export type TensorOrder = 0 | 1 | 2 | 3;
export type DisplayMode = "symbolic" | "numeric";
export interface AxisMetadata {
  id: number;
  enabled: boolean;
  length: number;
}
export interface IndexNotation {
  indices: string[];
  dimensions: string[];
  componentName: string;
}
export interface TensorModel {
  order: TensorOrder;
  shape: number[];
  axes: AxisMetadata[];
  values: number[];
  symbolicName: string;
  notation: IndexNotation;
}
// Indices are one-based throughout the model and UI.
export interface TensorSelection {
  component: number[] | null;
  slice: number | null;
}
export interface TensorCell {
  indices: number[];
  offset: number;
  value: number;
}
export type InteractionMode = "component" | "slice";
export interface OperationDefinition {
  id: string;
  label: string;
  available: boolean;
  interaction?: InteractionMode;
  minimumOrder?: TensorOrder;
}
export type AnimationEventKind =
  | "highlight"
  | "extract"
  | "moveToFormula"
  | "pair"
  | "multiply"
  | "sum"
  | "contractIndex"
  | "createResult"
  | "stackSlice";
export interface AnimationEvent {
  kind: AnimationEventKind;
  label: string;
  latex?: string;
}
export interface OperationResult {
  name: string;
  result: TensorModel;
  inputs: TensorModel[];
  formula: string;
  numericFormula?: string;
  detail: string;
  events: AnimationEvent[];
}
