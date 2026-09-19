import { useEffect, useState } from "react";
import { clampAnimationStep, EVENT_NAMES } from "../animation/engine";
import { runOperation } from "../operations/run";
import type { OperationResult, TensorModel, VisualAnimationState } from "../types";
import { MathText } from "./MathText";

const speedMs = { Slow: 1900, Normal: 1100, Fast: 700 } as const;
type Speed = keyof typeof speedMs;

interface Props {
  operationId: string;
  source: TensorModel;
  onResult: (result: OperationResult | null) => void;
  onAnimation: (frame: Partial<VisualAnimationState>) => void;
}

export function OperationStudio({ operationId, source, onResult, onAnimation }: Props) {
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
  const [view, setView] = useState<VisualAnimationState["view"]>(operationId === "tensor-chain-rule" ? "tensor" : "component");
  const [outputIndices] = useState([1, 1]);

  useEffect(() => {
    onAnimation({ operationId, step, view, outputIndices, active: Boolean(result), result, durationMs: speedMs[speed] });
  }, [operationId, step, view, outputIndices, result, speed, onAnimation]);

  useEffect(() => {
    if (!playing || !result) return;
    const id = window.setTimeout(() => setStep(current => {
      const next = current + 1;
      if (next >= result.events.length) {
        setPlaying(false);
        return result.events.length - 1;
      }
      return next;
    }), speedMs[speed]);
    return () => window.clearTimeout(id);
  }, [playing, result, step, speed]);

  if (operationId === "explore") return null;
  const run = () => {
    try {
      const next = runOperation(operationId, source, { targetShape, axis: axis - 1, fixedIndex, permutation, einsum });
      setResult(next);
      onResult(next);
      setStep(0);
      setPlaying(true);
      setError("");
    } catch (reason) {
      setResult(null);
      onResult(null);
      setError(reason instanceof Error ? reason.message : "Operation could not run");
    }
  };
  const hasAxis = ["slice", "reduction"].includes(operationId);
  const currentEvent = result?.events[step]?.kind;
  const chainFlow = currentEvent === "highlight" ? "X\\longrightarrow y=Xa+b\\longrightarrow z=Cy" : "\\frac{\\partial z}{\\partial y}\\frac{\\partial y}{\\partial X}\\longrightarrow\\frac{\\partial z}{\\partial X}";

  return <section className="operation-studio" aria-label="Operation simulator">
    <p className="studio-title">Operation setup</p>
    {operationId === "reshape" || operationId === "broadcast" ? <label>Target shape<input aria-label="Target shape" value={targetShape} onChange={event => setTargetShape(event.target.value)} /></label> : null}
    {operationId === "transpose" ? <label>Axis order<input aria-label="Axis order" value={permutation} onChange={event => setPermutation(event.target.value)} /></label> : null}
    {hasAxis ? <div className="studio-grid"><label>Axis<select aria-label="Operation axis" value={axis} onChange={event => setAxis(Number(event.target.value))}>{source.shape.map((_, index) => <option key={index} value={index + 1}>Axis {index + 1}</option>)}</select></label>{operationId === "slice" ? <label>Fixed index<input aria-label="Fixed index" type="number" min="1" max={source.shape[axis - 1]} value={fixedIndex} onChange={event => setFixedIndex(Number(event.target.value))} /></label> : null}</div> : null}
    {operationId === "einsum" ? <label>Einstein notation<input aria-label="Einstein notation" value={einsum} onChange={event => setEinsum(event.target.value)} /></label> : null}
    <div className="studio-actions"><button className="run-button" onClick={run}>Run operation</button><label>Speed<select aria-label="Animation speed" value={speed} onChange={event => setSpeed(event.target.value as Speed)}>{Object.keys(speedMs).map(value => <option key={value}>{value}</option>)}</select></label></div>
    {error ? <p className="operation-error" role="alert">{error}</p> : null}
    {result ? <div className="operation-result">
      <p className="result-name">{result.name} <span>→ Shape ({result.result.shape.join(", ")})</span></p>
      {operationId === "contraction-1d3d" ? <div className="view-toggle"><button aria-pressed={view === "component"} onClick={() => setView("component")}>Component View</button><button aria-pressed={view === "slice"} onClick={() => setView("slice")}>Slice View</button></div> : null}
      {operationId === "tensor-chain-rule" ? <><div className="view-toggle"><button aria-pressed={view === "tensor"} onClick={() => setView("tensor")}>Tensor View</button><button aria-pressed={view === "vectorized"} onClick={() => setView("vectorized")}>Vectorized View</button></div><div className="chain-flow"><MathText latex={view === "vectorized" ? "\\frac{\\partial z}{\\partial\\operatorname{vec}(X)}=\\frac{\\partial z}{\\partial y}\\frac{\\partial y}{\\partial\\operatorname{vec}(X)}" : chainFlow} /></div></> : null}
      <p className="operation-detail">{result.detail}</p>
      <div className="event-track">{result.events.map((event, index) => <button key={`${event.kind}-${index}`} aria-pressed={step === index} className={index <= step ? "is-complete" : ""} onClick={() => { setPlaying(false); setStep(index); }}><span>{index + 1}</span>{EVENT_NAMES[event.kind]}</button>)}</div>
      <div className="player-controls"><button onClick={() => { setPlaying(false); setStep(value => clampAnimationStep(value - 1, result.events)); }}>Previous</button><button onClick={() => setPlaying(value => !value)}>{playing ? "Pause" : "Play"}</button><button onClick={() => { setPlaying(false); setStep(value => clampAnimationStep(value + 1, result.events)); }}>Next</button><button onClick={() => { setStep(0); setPlaying(true); }}>Replay</button></div>
    </div> : null}
  </section>;
}
