import type { DisplayMode, OperationResult, TensorModel, TensorSelection, VisualAnimationState } from "../types";
import { offsetOf } from "../model/tensor";
import {
  componentLatex,
  generalComponent,
  sliceLatex,
  tensorSpace,
} from "../utils/notation";
import { MathText } from "./MathText";
export function FormulaPanel({
  model,
  mode,
  selection,
  operationResult,
  animation,
}: {
  model: TensorModel;
  mode: DisplayMode;
  selection: TensorSelection;
  operationResult: OperationResult | null;
  animation: VisualAnimationState;
}) {
  const selected = selection.component;
  return (
    <section
      className="formula-area"
      aria-label="Tensor formula"
      aria-live="polite"
    >
      <div className="tensor-formula">
        <MathText latex={tensorSpace(model)} />
        <MathText latex={generalComponent(model)} />
      </div>
      <div className="selection-formula">
        {operationResult && <><div className="operation-formula"><MathText latex={animation.result?.events[animation.step]?.latex ?? operationResult.formula} /></div>{animation.result?.events[animation.step] ? <p className="formula-step">Current step · {animation.result.events[animation.step].label}</p> : null}{mode === "numeric" && operationResult.numericFormula ? <MathText className="numeric-formula" latex={operationResult.numericFormula} /> : null}<ResultTensor result={operationResult} mode={mode} /></>}
        {selection.slice !== null && (
          <p className="selected-slice">
            Selected slice:{" "}
            <MathText latex={sliceLatex(model, selection.slice)} />
          </p>
        )}
        {selected !== null && (
          <>
            <div className="selected-component">
              <MathText
                latex={
                  componentLatex(model, selected) +
                  (mode === "numeric"
                    ? ` = ${model.values[offsetOf(selected, model.shape)]}`
                    : "")
                }
              />
            </div>
            <div className="index-readout">
              {selected.length ? (
                selected.map((index, axis) => (
                  <span key={axis}>
                    Axis {axis + 1} index = {index}
                  </span>
                ))
              ) : (
                <span>Scalar · no axes</span>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function ResultTensor({ result, mode }: { result: OperationResult; mode: DisplayMode }) {
  const values = result.result.values;
  const cells = values.slice(0, 36);
  return <div className="result-tensor" aria-label={`${result.name} result tensor`}>
    <p>Result tensor · Shape ({result.result.shape.join(", ") || "scalar"})</p>
    <div className="result-grid" style={{ gridTemplateColumns: `repeat(${Math.min(result.result.shape.at(-1) ?? 1, 6)}, minmax(28px, 1fr))` }}>
      {cells.map((value, index) => <span key={index}>{mode === "numeric" ? value : `${result.result.symbolicName}${index + 1}`}</span>)}
    </div>
    {values.length > cells.length ? <small>Showing first {cells.length} of {values.length} components</small> : null}
  </div>;
}
