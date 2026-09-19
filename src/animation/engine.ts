import type { AnimationEvent } from "../types/index.ts";

export const EVENT_NAMES: Record<AnimationEvent["kind"], string> = {
  highlight: "highlight selected components",
  extract: "extract components",
  moveToFormula: "move to formula",
  pair: "pair matching indices",
  multiply: "multiply terms",
  sum: "sum retained terms",
  contractIndex: "remove contracted index",
  createResult: "create result tensor",
  stackSlice: "stack result slices",
};

export function clampAnimationStep(step: number, events: AnimationEvent[]) {
  return Math.max(0, Math.min(step, Math.max(0, events.length - 1)));
}
