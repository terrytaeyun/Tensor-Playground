import { useState } from "react";
import type {
  AxisMetadata,
  DisplayMode,
  TensorModel,
  TensorSelection,
} from "../types";
import { toggleAxis, offsetOf } from "../model/tensor";
import { operations } from "../operations";
import { shapeText, sliceLatex, tensorKind } from "../utils/notation";
import { MathText } from "./MathText";

function LengthInput({
  value,
  disabled,
  label,
  onCommit,
}: {
  value: number;
  disabled: boolean;
  label: string;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const valid =
    Number.isInteger(Number(draft)) && Number(draft) >= 1 && Number(draft) <= 6;
  return (
    <input
      aria-label={label}
      type="number"
      min={1}
      max={6}
      step={1}
      disabled={disabled}
      value={draft}
      aria-invalid={!disabled && !valid}
      onChange={(event) => {
        const text = event.target.value;
        setDraft(text);
        const length = Number(text);
        if (Number.isInteger(length) && length >= 1 && length <= 6)
          onCommit(length);
      }}
      onBlur={() => setDraft(String(value))}
    />
  );
}
interface Props {
  model: TensorModel;
  mode: DisplayMode;
  selection: TensorSelection;
  operationId: string;
  onAxesChange: (axes: AxisMetadata[]) => void;
  onModeChange: (mode: DisplayMode) => void;
  onOperationChange: (id: string) => void;
  onSliceSelect: (slice: number) => void;
  onClear: () => void;
  onValueChange: (value: number) => void;
}
export function ControlPanel({
  model,
  mode,
  selection,
  operationId,
  onAxesChange,
  onModeChange,
  onOperationChange,
  onSliceSelect,
  onClear,
  onValueChange,
}: Props) {
  return (
    <aside className="controls" aria-label="Tensor controls">
      <h2>Tensor</h2>
      <p className="shape-summary" aria-live="polite">
        {shapeText(model)}
      </p>
      <p className="tensor-kind">{tensorKind(model.order)}</p>
      <fieldset className="axis-fields">
        <legend className="sr-only">Axes</legend>
        {model.axes.map((axis, index) => (
          <div className="axis-row" key={axis.id}>
            <span id={`axis-${axis.id}`}>Axis {axis.id}</span>
            <button
              className="axis-toggle"
              role="switch"
              aria-label={`Axis ${axis.id}`}
              aria-checked={axis.enabled}
              disabled={
                !axis.enabled && index > 0 && !model.axes[index - 1].enabled
              }
              onClick={() =>
                onAxesChange(toggleAxis(model.axes, index, !axis.enabled))
              }
            >
              {axis.enabled ? "ON" : "OFF"}
            </button>
            <span className="length-caption">length</span>
            <LengthInput
              value={axis.length}
              disabled={!axis.enabled}
              label={`Axis ${axis.id} length`}
              onCommit={(length) =>
                onAxesChange(
                  model.axes.map((item, i) =>
                    i === index ? { ...item, length } : item,
                  ),
                )
              }
            />
          </div>
        ))}
        <p className="quiet">길이 1–6 · 앞선 축을 끄면 뒤의 축도 꺼집니다.</p>
      </fieldset>
      <fieldset className="mode-controls">
        <legend>Values</legend>
        <div className="text-toggle">
          {(["symbolic", "numeric"] as DisplayMode[]).map((value) => (
            <button
              key={value}
              aria-pressed={mode === value}
              onClick={() => onModeChange(value)}
            >
              {value === "symbolic" ? "Symbolic" : "Numeric"}
            </button>
          ))}
        </div>
        {mode === "numeric" && (
          <p className="quiet">
            초기값은 1, 2, …입니다. 성분을 선택해 편집하세요.
          </p>
        )}
      </fieldset>
      {model.order === 3 && (
        <fieldset className="slice-controls">
          <legend>Matrix slices</legend>
          <div className="slice-options">
            {Array.from({ length: model.shape[0] }, (_, i) => (
              <button
                key={i}
                aria-label={`Select slice ${i + 1}`}
                aria-pressed={selection.slice === i + 1}
                onClick={() => onSliceSelect(i + 1)}
              >
                <MathText latex={sliceLatex(model, i + 1)} />
              </button>
            ))}
          </div>
          <p className="quiet">
            {model.shape[1]} × {model.shape[2]} matrices · {model.shape[0]}{" "}
            slices
          </p>
        </fieldset>
      )}
      {(selection.slice !== null || selection.component !== null) && (
        <button className="text-button clear-selection" onClick={onClear}>
          전체 텐서 보기 / Clear selection
        </button>
      )}
      {mode === "numeric" && selection.component !== null && (
        <label className="value-editor">
          Selected value
          <input
            key={`${model.shape.join(",")}:${selection.component.join(",")}`}
            aria-label="Selected value"
            type="number"
            step="any"
            defaultValue={
              model.values[offsetOf(selection.component, model.shape)]
            }
            onChange={(event) => {
              if (event.target.value !== "")
                onValueChange(event.target.valueAsNumber);
            }}
            onBlur={(event) => {
              event.target.value = String(
                model.values[offsetOf(selection.component!, model.shape)],
              );
            }}
          />
        </label>
      )}
      <div className="operation-control">
        <label htmlFor="operation">Operation</label>
        <select
          id="operation"
          value={operationId}
          onChange={(event) => onOperationChange(event.target.value)}
        >
          {operations.map((operation) => (
            <option
              key={operation.id}
              value={operation.id}
              disabled={model.order < (operation.minimumOrder ?? 0)}
            >
              {operation.label}
              {model.order < (operation.minimumOrder ?? 0)
                  ? " — requires 3 axes"
                  : ""}
            </option>
          ))}
        </select>
        <p className="quiet">
          {operationId === "explore" ? "셀을 클릭하여 성분을 살펴보세요." : "설정을 정한 뒤 Run operation을 누르면 수식과 결과가 단계적으로 나타납니다."}
        </p>
      </div>
    </aside>
  );
}
