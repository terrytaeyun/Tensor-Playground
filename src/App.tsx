import { useCallback, useState } from "react";
import { TensorVisualizer } from "./components/TensorVisualizer";
import { ControlPanel } from "./components/ControlPanel";
import { FormulaPanel } from "./components/FormulaPanel";
import {
  createTensorModel,
  EMPTY_SELECTION,
  INITIAL_AXES,
  validSelection,
  updateValue,
} from "./model/tensor";
import { getOperation } from "./operations";
import { OperationStudio } from "./components/OperationStudio";
import type { AxisMetadata, DisplayMode, OperationResult, TensorSelection, VisualAnimationState } from "./types";

export default function App() {
  const [model, setModel] = useState(() => createTensorModel(INITIAL_AXES));
  const [mode, setMode] = useState<DisplayMode>("symbolic");
  const [selection, setSelection] = useState<TensorSelection>(EMPTY_SELECTION);
  const [operationId, setOperationId] = useState("explore");
  const [operationResult, setOperationResult] = useState<OperationResult | null>(null);
  const [animation, setAnimation] = useState<VisualAnimationState>({ operationId: "explore", step: 0, view: "component", outputIndices: [1, 1], active: false, result: null, durationMs: 1100 });
  const operation = getOperation(operationId);
  function changeAxes(axes: AxisMetadata[]) {
    const next = createTensorModel(axes, model);
    setModel(next);
    setOperationResult(null);
    setAnimation(current => ({ ...current, active: false, result: null, step: 0 }));
    setSelection(validSelection(selection, next));
    if (next.order < (operation.minimumOrder ?? 0)) setOperationId("explore");
  }
  function selectComponent(indices: number[]) {
    setSelection({
      component: indices,
      slice: model.order === 3 ? indices[0] : null,
    });
  }
  function selectSlice(slice: number) {
    setSelection({ component: null, slice });
  }
  const receiveResult = useCallback((next: OperationResult | null) => setOperationResult(next), []);
  const receiveAnimation = useCallback((frame: Partial<VisualAnimationState>) => setAnimation(current => ({ ...current, ...frame })), []);
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Tensor Playground</h1>
        <span>Tensor builder</span>
      </header>
      <main className="workspace">
        <div className="canvas-column">
          <TensorVisualizer
            model={model}
            mode={mode}
            selection={selection}
            interaction={operation.interaction ?? "component"}
            animation={animation}
            onComponentSelect={selectComponent}
            onSliceSelect={selectSlice}
            onClear={() => setSelection(EMPTY_SELECTION)}
          />
          <FormulaPanel model={model} mode={mode} selection={selection} operationResult={operationResult} />
          <OperationStudio key={`${operationId}:${model.shape.join(",")}:${model.values.join(",")}`} operationId={operationId} source={model} onResult={receiveResult} onAnimation={receiveAnimation} />
        </div>
        <ControlPanel
          model={model}
          mode={mode}
          selection={selection}
          operationId={operationId}
          onAxesChange={changeAxes}
          onModeChange={setMode}
          onOperationChange={(id) => { setOperationId(id); setOperationResult(null); }}
          onSliceSelect={selectSlice}
          onClear={() => setSelection(EMPTY_SELECTION)}
          onValueChange={(value) => {
            if (selection.component !== null)
              setModel(updateValue(model, selection.component, value));
          }}
        />
      </main>
    </div>
  );
}
