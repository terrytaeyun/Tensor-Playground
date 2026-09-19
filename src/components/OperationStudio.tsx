import { useEffect, useState } from "react";
import { clampAnimationStep, EVENT_NAMES } from "../animation/engine";
import { runOperation } from "../operations/run";
import type { OperationResult, TensorModel, VisualAnimationState } from "../types";
import { MathText } from "./MathText";

const speedMs = { Slow: 1900, Normal: 1100, Fast: 700 } as const;
type Speed = keyof typeof speedMs;
function progressiveLatex(result: OperationResult, step: number, source: TensorModel) {
  if (!result.name.includes("1D × 3D")) return result.events[step]?.latex ?? result.formula;
  const rank = source.shape[0] ?? 1;
  if (step <= 0) return "C_{ij}=";
  if (step >= result.events.length - 1) return result.formula;
  const count = Math.min(rank, Math.max(1, step));
  const terms = Array.from({ length: count }, (_, index) => `v_{${index + 1}}X_{${index + 1}ij}`);
  return `C_{ij}=${terms.join("+")}`;
}
export function OperationStudio({ operationId, source, onResult, onAnimation }: { operationId: string; source: TensorModel; onResult: (result: OperationResult | null) => void; onAnimation: (frame: Partial<VisualAnimationState>) => void }) {
  const [targetShape, setTargetShape] = useState("3, 2, 4");
  const [permutation, setPermutation] = useState("1, 3, 2");
  const [axis, setAxis] = useState(1);
  const [fixedIndex, setFixedIndex] = useState(1);
  const [einsum, setEinsum] = useState("a,amn->mn");
  const [result, setResult] = useState<OperationResult | null>(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>("Normal");
  const [error, setError] = useState("");
  const [view] = useState<"component" | "slice">("component");
  const [outputIndices] = useState([1, 1]);
  useEffect(() => { onAnimation({ operationId, step, view, outputIndices, active: Boolean(result), result }); }, [operationId, step, view, outputIndices, result, onAnimation]);
  useEffect(() => {
    if (!playing || !result) return;
    const id = window.setTimeout(() => setStep(current => {
      const next = current + 1;
      if (next >= result.events.length) { setPlaying(false); return result.events.length - 1; }
      return next;
    }), speedMs[speed]);
    return () => window.clearTimeout(id);
  }, [playing, result, step, speed]);
  if (operationId === "explore") return null;
  const run = () => {
    try { const next = runOperation(operationId, source, { targetShape, axis: axis - 1, fixedIndex, permutation, einsum }); setResult(next); onResult(next); setStep(0); setPlaying(true); setError(""); }
    catch (reason) { setResult(null); onResult(null); setError(reason instanceof Error ? reason.message : "Operation could not run"); }
  };
  const hasAxis = ["slice", "reduction"].includes(operationId);
  return <section className="operation-studio" aria-label="Operation simulator">
    <p className="studio-title">Operation setup</p>
    {operationId === "reshape" || operationId === "broadcast" ? <label>Target shape<input aria-label="Target shape" value={targetShape} onChange={e => setTargetShape(e.target.value)} placeholder="3, 2, 4" /></label> : null}
    {operationId === "transpose" ? <label>Axis order<input aria-label="Axis order" value={permutation} onChange={e => setPermutation(e.target.value)} placeholder="1, 3, 2" /></label> : null}
    {hasAxis ? <div className="studio-grid"><label>Axis<select aria-label="Operation axis" value={axis} onChange={e => setAxis(Number(e.target.value))}>{source.shape.map((_, i) => <option key={i} value={i + 1}>Axis {i + 1}</option>)}</select></label>{operationId === "slice" ? <label>Fixed index<input aria-label="Fixed index" type="number" min="1" max={source.shape[axis - 1]} value={fixedIndex} onChange={e => setFixedIndex(Number(e.target.value))} /></label> : null}</div> : null}
    {operationId === "einsum" ? <label>Einstein notation<input aria-label="Einstein notation" value={einsum} onChange={e => setEinsum(e.target.value)} /></label> : null}
    <div className="studio-actions"><button className="run-button" onClick={run}>Run operation</button><label>Speed<select aria-label="Animation speed" value={speed} onChange={e => setSpeed(e.target.value as Speed)}>{Object.keys(speedMs).map(value => <option key={value}>{value}</option>)}</select></label></div>
    {error ? <p className="operation-error" role="alert">{error}</p> : null}
    {result ? <div className="operation-result"><p className="result-name">{result.name} <span>→ Shape ({result.result.shape.join(", ")})</span></p><p className="operation-detail">{result.detail}</p><div className="event-track">{result.events.map((event, index) => <button key={`${event.kind}-${index}`} aria-pressed={step === index} className={index <= step ? "is-complete" : ""} onClick={() => { setPlaying(false); setStep(index); }}><span>{index + 1}</span>{EVENT_NAMES[event.kind]}</button>)}</div><div className="event-current"><MathText latex={progressiveLatex(result, step, source)} /><p>{result.events[step]?.label}</p>{result.numericFormula && step >= 2 ? <MathText className="numeric-formula" latex={result.numericFormula} /> : null}</div><div className="player-controls"><button onClick={() => { setPlaying(false); setStep(value => clampAnimationStep(value - 1, result.events)); }}>Previous</button><button onClick={() => setPlaying(value => !value)}>{playing ? "Pause" : "Play"}</button><button onClick={() => { setPlaying(false); setStep(value => clampAnimationStep(value + 1, result.events)); }}>Next</button><button onClick={() => { setStep(0); setPlaying(true); }}>Replay</button></div></div> : null}
  </section>;
}
