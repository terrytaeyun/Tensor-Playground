import { useEffect, useRef, useState } from "react";
import type {
  DisplayMode,
  InteractionMode,
  TensorModel,
  TensorSelection,
} from "../types";
import { createTensorScene } from "../visualization/createTensorScene";
import { generalComponent, tensorSpace } from "../utils/notation";
import { MathText } from "./MathText";
interface Props {
  model: TensorModel;
  mode: DisplayMode;
  selection: TensorSelection;
  interaction: InteractionMode;
  onComponentSelect: (indices: number[]) => void;
  onSliceSelect: (slice: number) => void;
  onClear: () => void;
}
export function TensorVisualizer({
  model,
  mode,
  selection,
  interaction,
  onComponentSelect,
  onSliceSelect,
  onClear,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<ReturnType<typeof createTensorScene> | null>(null);
  const callbacks = useRef({ onComponentSelect, onSliceSelect });
  const [error, setError] = useState(false);
  useEffect(() => {
    callbacks.current = { onComponentSelect, onSliceSelect };
  }, [onComponentSelect, onSliceSelect]);
  useEffect(() => {
    try {
      scene.current = createTensorScene(host.current!, {
        onComponentSelect: (indices) =>
          callbacks.current.onComponentSelect(indices),
        onSliceSelect: (slice) => callbacks.current.onSliceSelect(slice),
      });
    } catch (reason) {
      console.error("Unable to initialize tensor scene", reason);
      queueMicrotask(() => setError(true));
    }
    return () => {
      scene.current?.dispose();
      scene.current = null;
    };
  }, []);
  useEffect(() => {
    scene.current?.update(model, mode, selection, interaction);
  }, [model, mode, selection, interaction]);
  return (
    <section className="visualizer" aria-label="Tensor visualization">
      <div className="abstract-notation">
        <MathText
          latex={`${tensorSpace(model, true)},\\qquad ${generalComponent(model)}`}
        />
      </div>
      <div className="scene-wrap">
        <div ref={host} className="scene" />
        {error && (
          <p className="scene-error" role="alert">
            3D 화면을 시작할 수 없습니다. 브라우저의 WebGL 지원과 하드웨어
            가속을 확인하세요.
          </p>
        )}
      </div>
      <div className="scene-toolbar">
        <span>Drag to rotate · Scroll to zoom · Right-drag to pan</span>
        <div>
          <button
            className="text-button"
            onClick={() => scene.current?.reset()}
          >
            Reset View
          </button>
          {(selection.component !== null || selection.slice !== null) && (
            <button className="text-button" onClick={onClear}>
              Clear selection
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
