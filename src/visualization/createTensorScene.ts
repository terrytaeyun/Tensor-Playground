import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  CSS2DObject,
  CSS2DRenderer,
} from "three/addons/renderers/CSS2DRenderer.js";
import katex from "katex";
import type {
  DisplayMode,
  InteractionMode,
  TensorModel,
  TensorSelection,
  VisualAnimationState,
} from "../types";
import { tensorCells } from "../model/tensor";
import { componentLatex, sliceLatex } from "../utils/notation";

interface Callbacks {
  onComponentSelect: (indices: number[]) => void;
  onSliceSelect: (slice: number) => void;
}
interface CellView {
  indices: number[];
  offset: number;
  slice: number | null;
  mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial>;
  outline: THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>;
  label: CSS2DObject;
  latex: string;
  ghost: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial>;
}
interface SliceView {
  index: number;
  plane: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  outline: THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>;
  label: CSS2DObject;
}
interface VectorView {
  index: number;
  mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial>;
  label: CSS2DObject;
  basePosition: THREE.Vector3;
}
interface OperandView {
  input: number;
  indices: number[];
  mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial>;
  label: CSS2DObject;
  basePosition: THREE.Vector3;
}
interface Hit {
  cell?: CellView;
  slice?: number;
}
interface Motion {
  startPosition: THREE.Vector3;
  targetPosition: THREE.Vector3;
  startScale: number;
  targetScale: number;
  startOpacity: number;
  targetOpacity: number;
}

// This module owns drawing, hit testing, and camera state only. Tensor values,
// index semantics, and operation behavior are supplied by the application.
export function createTensorScene(
  container: HTMLDivElement,
  callbacks: Callbacks,
) {
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.domElement.setAttribute(
    "aria-label",
    "Interactive tensor. Drag to rotate; scroll to zoom; right-drag to pan.",
  );
  const labels = new CSS2DRenderer();
  labels.domElement.className = "scene-label-layer";
  container.append(renderer.domElement, labels.domElement);
  const scene = new THREE.Scene();
  const content = new THREE.Group();
  scene.add(content);
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 500);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = true;
  controls.maxDistance = 120;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let model: TensorModel | null = null;
  let mode: DisplayMode = "symbolic";
  let selection: TensorSelection = { component: null, slice: null };
  let interaction: InteractionMode = "component";
  let animation: VisualAnimationState | null = null;
  let cells: CellView[] = [];
  let slices: SliceView[] = [];
  let targets: THREE.Object3D[] = [];
  let hovered: Hit | null = null;
  let down: { x: number; y: number; id: number; dragged: boolean } | null =
    null;
  let shapeKey = "";
  let disposed = false;
  let sceneRadius = 3;
  let stepStartedAt = performance.now();
  let previousAnimationKey = "";
  let resultGroup: THREE.Group | null = null;
  let resultCells: THREE.Mesh[] = [];
  let vectorCells: VectorView[] = [];
  let motions = new Map<CellView, Motion>();
  let vectorMotions = new Map<VectorView, Motion>();
  let operandGroup: THREE.Group | null = null;
  let operandCells: OperandView[] = [];
  let operandMotions = new Map<OperandView, Motion>();
  let operandKey = "";
  let genericResultGroup: THREE.Group | null = null;

  function drawMath(element: HTMLElement, latex: string) {
    katex.render(latex, element, {
      throwOnError: false,
      trust: false,
      output: "html",
    });
  }
  function reset() {
    if (!model) return;
    const bounds = new THREE.Box3().setFromObject(content);
    const sphere = bounds.getBoundingSphere(new THREE.Sphere());
    sceneRadius = Math.max(sphere.radius, 1.1);
    const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
    const distance =
      (sceneRadius / Math.sin(Math.min(vertical, horizontal))) * 1.13;
    const direction =
      model.order === 3
        ? new THREE.Vector3(0.63, 0.28, 1)
        : new THREE.Vector3(0.1, 0.08, 1);
    camera.position.copy(
      direction.normalize().multiplyScalar(distance).add(sphere.center),
    );
    controls.target.copy(sphere.center);
    controls.minDistance = Math.max(1.7, sceneRadius * 0.65);
    controls.maxDistance = Math.max(80, distance * 4);
    controls.update();
    controls.saveState();
  }
  function clearContent() {
    content.traverse((object) => {
      if (
        object instanceof THREE.Mesh ||
        object instanceof THREE.LineSegments
      ) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((material) => material.dispose());
      }
      if (object instanceof CSS2DObject) object.element.remove();
    });
    content.clear();
    cells = [];
    slices = [];
    targets = [];
    resultGroup = null;
    resultCells = [];
    vectorCells = [];
    motions.clear();
    vectorMotions.clear();
    operandGroup = null;
    operandCells = [];
    operandMotions.clear();
    operandKey = "";
    genericResultGroup = null;
    hovered = null;
  }
  function rebuild() {
    if (!model) return;
    clearContent();
    const rows =
      model.order === 3
        ? model.shape[1]
        : model.order === 2
          ? model.shape[0]
          : 1;
    const columns =
      model.order === 3
        ? model.shape[2]
        : model.order === 2
          ? model.shape[1]
          : model.order === 1
            ? model.shape[0]
            : 1;
    const count = model.order === 3 ? model.shape[0] : 1;
    const spacing = 1.08;
    const sliceSpacing = 1.85;
    const planeWidth = columns * spacing + 0.18;
    const planeHeight = rows * spacing + 0.18;
    for (let slice = 0; slice < count; slice++) {
      const z = ((count - 1) / 2 - slice) * sliceSpacing;
      if (model.order >= 2) {
        const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
        const plane = new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({
            color: "#7b92ab",
            transparent: true,
            opacity: 0.045,
            side: THREE.DoubleSide,
            depthWrite: false,
          }),
        );
        plane.position.z = z - 0.085;
        plane.userData.baseX = plane.position.x;
        const outline = new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry),
          new THREE.LineBasicMaterial({
            color: "#7288a5",
            transparent: true,
            opacity: 0.28,
          }),
        );
        plane.add(outline);
        content.add(plane);
        if (model.order === 3) {
          plane.userData.slice = slice + 1;
          targets.push(plane);
          const button = document.createElement("button");
          button.className = "slice-label";
          button.setAttribute("aria-label", `Canvas slice ${slice + 1}`);
          drawMath(button, sliceLatex(model, slice + 1));
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            callbacks.onSliceSelect(slice + 1);
          });
          const label = new CSS2DObject(button);
          label.position.set(-planeWidth / 2 - 0.48, planeHeight / 2 + 0.16, z);
          content.add(label);
          slices.push({ index: slice + 1, plane, outline, label });
        }
      }
    }
    for (const cell of tensorCells(model)) {
      const sliceIndex = model.order === 3 ? cell.indices[0] - 1 : 0;
      const row =
        model.order === 3
          ? cell.indices[1] - 1
          : model.order === 2
            ? cell.indices[0] - 1
            : 0;
      const column =
        model.order === 3
          ? cell.indices[2] - 1
          : model.order === 2
            ? cell.indices[1] - 1
            : model.order === 1
              ? cell.indices[0] - 1
              : 0;
      const geometry = new THREE.BoxGeometry(0.96, 0.96, 0.12);
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({
          color: "#b9c9d9",
          transparent: true,
          opacity: 0.23,
          depthWrite: false,
        }),
      );
      mesh.position.set(
        (column - (columns - 1) / 2) * spacing,
        ((rows - 1) / 2 - row) * spacing,
        ((count - 1) / 2 - sliceIndex) * sliceSpacing,
      );
      mesh.userData.baseX = mesh.position.x;
      mesh.userData.baseZ = mesh.position.z;
      const ghost = new THREE.Mesh(geometry.clone(), new THREE.MeshBasicMaterial({ color: "#90a5bb", transparent: true, opacity: 0, depthWrite: false }));
      ghost.position.copy(mesh.position);
      ghost.renderOrder = -1;
      content.add(ghost);
      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({
          color: "#8197af",
          transparent: true,
          opacity: 0.38,
        }),
      );
      mesh.add(outline);
      const element = document.createElement("span");
      element.className = "cell-label";
      // Hit testing stays on WebGL so labels never block orbit gestures or cells.
      element.setAttribute("aria-hidden", "true");
      const label = new CSS2DObject(element);
      label.position.z = 0.09;
      mesh.add(label);
      const view: CellView = {
        ...cell,
        slice: model.order === 3 ? sliceIndex + 1 : null,
        mesh,
        outline,
        label,
        latex: "",
        ghost,
      };
      mesh.userData.cell = view;
      content.add(mesh);
      cells.push(view);
      targets.push(mesh);
    }
    if (model.order === 3) {
      const vectorX = -planeWidth / 2 - 1.7;
      for (let index = 0; index < model.shape[0]; index += 1) {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.22), new THREE.MeshBasicMaterial({ color: "#d2a66f", transparent: true, opacity: 0, depthWrite: false }));
        mesh.position.set(vectorX, ((model.shape[0] - 1) / 2 - index) * 1.08, 0.35);
        const element = document.createElement("span");
        element.className = "cell-label vector-label";
        element.setAttribute("aria-hidden", "true");
        const label = new CSS2DObject(element);
        label.position.z = 0.14;
        mesh.add(label);
        content.add(mesh);
        vectorCells.push({ index: index + 1, mesh, label, basePosition: mesh.position.clone() });
      }
    }
    if (model.order === 3) {
      resultGroup = new THREE.Group();
      resultGroup.userData.width = model.shape[2];
      resultGroup.position.x = planeWidth / 2 + 2.2;
      resultGroup.visible = false;
      const rw = model.shape[2];
      const rh = model.shape[1];
      for (let row = 0; row < rh; row += 1) for (let col = 0; col < rw; col += 1) {
        const cell = new THREE.Mesh(new THREE.BoxGeometry(0.96, 0.96, 0.16), new THREE.MeshBasicMaterial({ color: "#5f86b5", transparent: true, opacity: 0.7, depthWrite: false }));
        cell.position.set((col - (rw - 1) / 2) * spacing, ((rh - 1) / 2 - row) * spacing, 0);
        resultGroup.add(cell); resultCells.push(cell);
      }
      content.add(resultGroup);
    }
    reset();
  }
  function activeEventKind() {
    return animation?.result?.events[animation.step]?.kind ?? null;
  }
  function clearOperandGroup() {
    if (!operandGroup) return;
    operandGroup.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose();
        (object.material as THREE.Material).dispose();
      }
      if (object instanceof CSS2DObject) object.element.remove();
    });
    content.remove(operandGroup);
    operandGroup = null;
    genericResultGroup = null;
    operandCells = [];
    operandMotions.clear();
  }
  function buildOperationOperands() {
    const result = animation?.result;
    const operationId = animation?.operationId;
    const key = result && operationId ? `${operationId}:${result.inputs.map(input => input.shape.join("x")).join("|")}` : "";
    if (key === operandKey) return;
    clearOperandGroup();
    operandKey = key;
    if (!result || !operationId || operationId === "contraction-1d3d" || operationId === "explore") return;
    operandGroup = new THREE.Group();
    const gap = 5.4;
    result.inputs.forEach((input, inputIndex) => {
      const rows = input.order >= 2 ? input.shape[input.order - 2] : 1;
      const columns = input.order >= 1 ? input.shape[input.order - 1] : 1;
      const slicesCount = input.order === 3 ? input.shape[0] : 1;
      for (const cell of tensorCells(input)) {
        const slice = input.order === 3 ? cell.indices[0] - 1 : 0;
        const row = input.order >= 2 ? cell.indices[input.order - 2] - 1 : 0;
        const column = input.order >= 1 ? cell.indices[input.order - 1] - 1 : 0;
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.76, 0.14), new THREE.MeshBasicMaterial({ color: inputIndex === 0 ? "#83a8ca" : "#d2a66f", transparent: true, opacity: 0.5, depthWrite: false }));
        mesh.position.set((inputIndex - (result.inputs.length - 1) / 2) * gap + (column - (columns - 1) / 2) * 0.88, ((rows - 1) / 2 - row) * 0.88, (slicesCount - 1 - slice) * 0.9 + 1.1);
        const element = document.createElement("span");
        element.className = "cell-label operation-operand-label";
        element.setAttribute("aria-hidden", "true");
        drawMath(element, mode === "numeric" ? String(cell.value) : `${input.symbolicName}_{${cell.indices.join("")}}`);
        const label = new CSS2DObject(element);
        label.position.z = 0.1;
        mesh.add(label);
        operandGroup?.add(mesh);
        operandCells.push({ input: inputIndex, indices: cell.indices, mesh, label, basePosition: mesh.position.clone() });
      }
    });
    const output = result.result;
    genericResultGroup = new THREE.Group();
    genericResultGroup.position.set(0, -2.45, 1.2);
    genericResultGroup.visible = false;
    const outputRows = output.order >= 2 ? output.shape[output.order - 2] : 1;
    const outputColumns = output.order >= 1 ? output.shape[output.order - 1] : 1;
    const outputCells = tensorCells(output).slice(0, 36);
    for (const cell of outputCells) {
      const row = output.order >= 2 ? cell.indices[output.order - 2] - 1 : 0;
      const column = output.order >= 1 ? cell.indices[output.order - 1] - 1 : 0;
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.72, 0.16), new THREE.MeshBasicMaterial({ color: "#527aaf", transparent: true, opacity: 0.78, depthWrite: false }));
      mesh.position.set((column - (outputColumns - 1) / 2) * 0.8, ((outputRows - 1) / 2 - row) * 0.8, 0);
      const element = document.createElement("span");
      element.className = "cell-label operation-result-label";
      element.setAttribute("aria-hidden", "true");
      drawMath(element, mode === "numeric" ? String(cell.value) : `${output.symbolicName}_{${cell.indices.join("")}}`);
      const label = new CSS2DObject(element);
      label.position.z = 0.12;
      mesh.add(label);
      genericResultGroup.add(mesh);
    }
    operandGroup.add(genericResultGroup);
    content.add(operandGroup);
  }
  function isContractedComponent(cell: CellView) {
    const output = animation?.outputIndices ?? [1, 1];
    return Boolean(animation?.active && animation.operationId === "contraction-1d3d" && cell.indices.length === 3 && cell.indices[1] === output[0] && cell.indices[2] === output[1]);
  }
  function easeInOutCubic(value: number) {
    return value < 0.5 ? 4 * value * value * value : 1 - Math.pow(-2 * value + 2, 3) / 2;
  }
  function isActiveOperand(cell: OperandView) {
    const operationId = animation?.operationId;
    if (operationId === "matmul" || operationId === "einsum") return cell.input === 0 ? cell.indices[0] === 1 : cell.indices.at(-1) === 1;
    if (operationId === "contraction-2d3d" || operationId === "tensor-chain-rule") return cell.input === 0 ? cell.indices[0] === 1 : cell.indices.length === 3 && cell.indices[1] === 1 && cell.indices[2] === 1;
    if (operationId === "double-contraction") return cell.input === 0 ? cell.indices[0] === 1 : cell.indices.at(-1) === 1;
    return true;
  }
  function prepareStepMotion() {
    const event = activeEventKind();
    motions.clear();
    vectorMotions.clear();
    for (const cell of cells) {
      if (!isContractedComponent(cell)) continue;
      const base = new THREE.Vector3(cell.mesh.userData.baseX as number, cell.mesh.position.y, cell.mesh.userData.baseZ as number);
      const extracted = base.clone().add(new THREE.Vector3(0, 0, 1.15));
      const formula = base.clone().add(new THREE.Vector3(3.4, 0.15, 1.35));
      const summed = base.clone().add(new THREE.Vector3(4.25, -0.15, 0.7));
      const stacked = formula.clone().add(new THREE.Vector3(0, 0.55, 0));
      const paired = formula.clone().add(new THREE.Vector3(0.35, 0.15, 0));
      const weighted = formula.clone().add(new THREE.Vector3(0.7, -0.1, 0));
      const collapsed = summed.clone().add(new THREE.Vector3(0.45, 0, -0.2));
      const target = event === "highlight" ? base : event === "extract" ? extracted : event === "moveToFormula" ? formula : event === "stackSlice" ? stacked : event === "pair" ? paired : event === "multiply" ? weighted : event === "sum" ? summed : event === "contractIndex" || event === "createResult" ? collapsed : base;
      const targetScale = event === "multiply" ? 0.78 : event === "sum" || event === "contractIndex" || event === "createResult" ? 0.65 : 1;
      const targetOpacity = event === "contractIndex" || event === "createResult" ? 0.12 : 0.82;
      motions.set(cell, { startPosition: cell.mesh.position.clone(), targetPosition: target, startScale: cell.mesh.scale.x, targetScale, startOpacity: cell.mesh.material.opacity, targetOpacity });
    }
    for (const vectorCell of vectorCells) {
      const extracted = vectorCell.basePosition.clone().add(new THREE.Vector3(0, 0, 0.8));
      const formula = vectorCell.basePosition.clone().add(new THREE.Vector3(2.1, 0, 0.9));
      const target = event === "extract" ? extracted : event === "moveToFormula" || event === "pair" ? formula : event === "multiply" ? formula.clone().add(new THREE.Vector3(0.55, 0, 0)) : event === "sum" || event === "contractIndex" || event === "createResult" ? formula.clone().add(new THREE.Vector3(1.3, -0.1, -0.2)) : vectorCell.basePosition.clone();
      vectorMotions.set(vectorCell, { startPosition: vectorCell.mesh.position.clone(), targetPosition: target, startScale: vectorCell.mesh.scale.x, targetScale: event === "multiply" ? 0.78 : event === "contractIndex" || event === "createResult" ? 0.64 : 1, startOpacity: vectorCell.mesh.material.opacity, targetOpacity: event === "contractIndex" || event === "createResult" ? 0.16 : 0.9 });
    }
    operandMotions.clear();
    for (const operand of operandCells) {
      if (!isActiveOperand(operand)) continue;
      const direction = operand.input === 0 ? 1 : -1;
      const extracted = operand.basePosition.clone().add(new THREE.Vector3(0, 0, 0.85));
      const paired = operand.basePosition.clone().add(new THREE.Vector3(direction * 1.15, 0, 1.1));
      const formula = new THREE.Vector3(direction * 0.45, -2.1, 2.1);
      const summed = new THREE.Vector3(0, -2.35, 1.65);
      const target = event === "extract" ? extracted : event === "moveToFormula" || event === "pair" ? paired : event === "multiply" ? formula : event === "sum" || event === "contractIndex" || event === "createResult" ? summed : operand.basePosition.clone();
      operandMotions.set(operand, { startPosition: operand.mesh.position.clone(), targetPosition: target, startScale: operand.mesh.scale.x, targetScale: event === "multiply" ? 0.76 : event === "sum" || event === "contractIndex" ? 0.62 : 1, startOpacity: operand.mesh.material.opacity, targetOpacity: event === "contractIndex" || event === "createResult" ? 0.15 : 0.9 });
    }
  }
  function refreshAppearance() {
    if (!model) return;
    const anyAnimationActive = Boolean(animation?.active);
    const visualAnimationActive = Boolean(animation?.active && animation.operationId === "contraction-1d3d");
    const phase = anyAnimationActive ? Math.min(1, Math.max(0, (performance.now() - stepStartedAt) / (animation?.durationMs ?? 1100))) : 0;
    const eased = easeInOutCubic(phase);
    const event = activeEventKind();
    const selectedKey = selection.component?.join(",");
    const dense = cells.length > 64;
    for (const cell of cells) {
      const selected =
        selectedKey !== undefined && cell.indices.join(",") === selectedKey;
      const hover = hovered?.cell === cell;
      const sliceSelected =
        selection.slice !== null && selection.slice === cell.slice;
      const muted = selection.slice !== null && selection.slice !== cell.slice;
      const visualActive = animation?.active && animation.operationId === "contraction-1d3d";
      const output = animation?.outputIndices ?? [1, 1];
      const contractionSelected = visualActive && cell.indices.length === 3 && cell.indices[1] === output[0] && cell.indices[2] === output[1];
      const visualMuted = visualActive && !contractionSelected;
      const overlayActive = anyAnimationActive && animation?.operationId !== "contraction-1d3d";
      const motion = motions.get(cell);
      cell.ghost.visible = Boolean(visualActive && contractionSelected && event !== "highlight");
      cell.ghost.material.opacity = cell.ghost.visible ? 0.16 : 0;
      cell.ghost.position.set(cell.mesh.userData.baseX as number, cell.mesh.position.y, cell.mesh.userData.baseZ as number);
      cell.mesh.material.color.set(
        selected ? "#527aaf" : hover ? "#8baacc" : "#b9c9d9",
      );
      cell.mesh.material.opacity = contractionSelected
        ? motion ? THREE.MathUtils.lerp(motion.startOpacity, motion.targetOpacity, eased) : 0.82
        : visualMuted
          ? 0.045
          : selected
        ? 0.65
        : hover
          ? 0.48
          : muted
            ? 0.055
            : sliceSelected
              ? 0.32
              : overlayActive ? 0.025 : 0.2;
      cell.outline.material.color.set(
        selected || hover ? "#3b6395" : "#8197af",
      );
      cell.outline.material.opacity =
        contractionSelected ? 1 : selected || hover ? 0.95 : muted ? 0.14 : sliceSelected ? 0.65 : 0.34;
      if (motion) {
        cell.mesh.position.lerpVectors(motion.startPosition, motion.targetPosition, eased);
        cell.mesh.scale.setScalar(THREE.MathUtils.lerp(motion.startScale, motion.targetScale, eased));
      }
      const element = cell.label.element;
      element.className = [
        "cell-label",
        selected ? "is-selected" : "",
        hover ? "is-hovered" : "",
        muted ? "is-muted" : "",
        dense ? "is-dense" : "",
        sliceSelected ? "in-selected-slice" : "",
      ].join(" ");
      const latex =
        mode === "symbolic"
          ? componentLatex(model, cell.indices, true)
          : String(model.values[cell.offset]);
      if (cell.latex !== latex) {
        drawMath(element, latex);
        cell.latex = latex;
      }
    }
    const vector = animation?.result?.inputs[0];
    for (const vectorCell of vectorCells) {
      const active = visualAnimationActive;
      const paired = event === "pair" || event === "multiply" || event === "sum" || event === "contractIndex";
      const vectorMotion = vectorMotions.get(vectorCell);
      if (vectorMotion) {
        vectorCell.mesh.position.lerpVectors(vectorMotion.startPosition, vectorMotion.targetPosition, eased);
        vectorCell.mesh.scale.setScalar(THREE.MathUtils.lerp(vectorMotion.startScale, vectorMotion.targetScale, eased));
      }
      vectorCell.mesh.material.opacity = active ? (vectorMotion ? THREE.MathUtils.lerp(vectorMotion.startOpacity, vectorMotion.targetOpacity, eased) : 0.9) : 0;
      vectorCell.mesh.material.color.set(paired ? "#bf7d32" : "#d2a66f");
      const latex = mode === "numeric" && vector ? String(vector.values[vectorCell.index - 1]) : `v_{${vectorCell.index}}`;
      drawMath(vectorCell.label.element, latex);
    }
    for (const operand of operandCells) {
      const active = isActiveOperand(operand);
      const motion = operandMotions.get(operand);
      if (motion) {
        operand.mesh.position.lerpVectors(motion.startPosition, motion.targetPosition, eased);
        operand.mesh.scale.setScalar(THREE.MathUtils.lerp(motion.startScale, motion.targetScale, eased));
        operand.mesh.material.opacity = THREE.MathUtils.lerp(motion.startOpacity, motion.targetOpacity, eased);
      } else {
        operand.mesh.material.opacity = anyAnimationActive ? (active ? 0.82 : 0.12) : 0;
      }
      operand.mesh.material.color.set(active ? (operand.input === 0 ? "#4f82b2" : "#bf7d32") : "#c8d0d8");
      operand.label.element.classList.toggle("is-muted", !active);
    }
    if (genericResultGroup) genericResultGroup.visible = event === "createResult";
    for (const slice of slices) {
      const selected = selection.slice === slice.index;
      const hover = hovered?.slice === slice.index && !hovered.cell;
      const sliceAnimation = visualAnimationActive && animation?.view === "slice";
      const weighted = sliceAnimation && (animation?.step ?? 0) >= 2;
      const sliceBaseX = slice.plane.userData.baseX as number;
      slice.plane.position.x += (sliceBaseX + (weighted ? 0.65 : 0) - slice.plane.position.x) * 0.12;
      slice.plane.userData.moveX = weighted ? 0.25 : 0;
      slice.plane.material.opacity = weighted ? 0.16 : selected ? 0.12 : hover ? 0.09 : 0.035;
      slice.outline.material.color.set(selected ? "#3b6395" : "#7288a5");
      slice.outline.material.opacity = selected ? 0.95 : hover ? 0.7 : 0.3;
      slice.label.element.classList.toggle("is-selected", selected);
      slice.label.element.setAttribute("aria-pressed", String(selected));
      if (visualAnimationActive && animation?.view === "slice") {
        const vectorLatex = mode === "numeric" && vector ? String(vector.values[slice.index - 1]) : `v_{${slice.index}}`;
        drawMath(slice.label.element, `${vectorLatex}\\,X_{${slice.index},:,:}`);
      }
    }
    if (resultGroup) {
      const show = Boolean(visualAnimationActive && event === "createResult");
      resultGroup.visible = show;
      resultCells.forEach((cell, index) => {
        const active = animation?.outputIndices && index === ((animation.outputIndices[0] - 1) * ((resultGroup?.userData.width as number) ?? 1) + animation.outputIndices[1] - 1);
        (cell.material as THREE.MeshBasicMaterial).opacity = active ? 1 : 0.45;
      });
    }
  }
  function hitAt(event: PointerEvent): Hit | null {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      (-(event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(targets, false);
    // A selected plane remains inspectable through the now-muted front slices.
    const activeSliceHit =
      interaction === "component" && selection.slice !== null
        ? hits.find(
            (hit) =>
              hit.object.userData.cell?.slice === selection.slice ||
              hit.object.userData.slice === selection.slice,
          )
        : undefined;
    const hit = activeSliceHit ?? hits[0];
    if (!hit) return null;
    const cell = hit.object.userData.cell as CellView | undefined;
    return cell
      ? { cell, slice: cell.slice ?? undefined }
      : { slice: hit.object.userData.slice as number };
  }
  function pointerDown(event: PointerEvent) {
    if (event.button === 0)
      down = {
        x: event.clientX,
        y: event.clientY,
        id: event.pointerId,
        dragged: false,
      };
  }
  function pointerMove(event: PointerEvent) {
    if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5)
      down.dragged = true;
    hovered = event.buttons ? null : hitAt(event);
    renderer.domElement.style.cursor = event.buttons
      ? "grabbing"
      : hovered
        ? "pointer"
        : "grab";
    refreshAppearance();
  }
  function pointerUp(event: PointerEvent) {
    if (!down || down.id !== event.pointerId) return;
    const click =
      !down.dragged &&
      Math.hypot(event.clientX - down.x, event.clientY - down.y) <= 5;
    down = null;
    if (!click) return;
    const hit = hitAt(event);
    if (hit?.cell && interaction === "component")
      callbacks.onComponentSelect([...hit.cell.indices]);
    else if (hit?.slice !== undefined) callbacks.onSliceSelect(hit.slice);
  }
  function pointerLeave() {
    hovered = null;
    refreshAppearance();
  }
  function pointerCancel() {
    down = null;
    pointerLeave();
  }
  renderer.domElement.addEventListener("pointerdown", pointerDown);
  renderer.domElement.addEventListener("pointermove", pointerMove);
  renderer.domElement.addEventListener("pointerup", pointerUp);
  renderer.domElement.addEventListener("pointerleave", pointerLeave);
  renderer.domElement.addEventListener("pointercancel", pointerCancel);
  function resize() {
    const { width, height } = container.getBoundingClientRect();
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    labels.setSize(width, height);
  }
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();
  renderer.setAnimationLoop(() => {
    if (disposed) return;
    controls.update();
    // Keep visual interpolation alive independently from React playback updates.
    // React changes the target once per step; Three.js advances toward that target
    // on every rendered frame so a step cannot appear to stall after one tick.
    refreshAppearance();
    renderer.render(scene, camera);
    labels.render(scene, camera);
    // CSS2D depth sorting is overridden only for the active cell's legibility.
    for (const cell of cells) {
      if (
        cell.label.element.classList.contains("is-selected") ||
        cell.label.element.classList.contains("is-hovered")
      )
        cell.label.element.style.zIndex = "1000";
    }
  });
  return {
    update(
      nextModel: TensorModel,
      nextMode: DisplayMode,
      nextSelection: TensorSelection,
      nextInteraction: InteractionMode,
      nextAnimation?: VisualAnimationState,
    ) {
      model = nextModel;
      mode = nextMode;
      selection = nextSelection;
      interaction = nextInteraction;
      animation = nextAnimation ?? null;
      buildOperationOperands();
      const animationKey = `${nextAnimation?.operationId ?? "none"}:${nextAnimation?.step ?? -1}:${Boolean(nextAnimation?.active)}`;
      const stepChanged = animationKey !== previousAnimationKey;
      if (stepChanged) {
        previousAnimationKey = animationKey;
        stepStartedAt = performance.now();
      }
      const key = model.shape.join(",") + `:${model.order}`;
      if (key !== shapeKey) {
        shapeKey = key;
        rebuild();
      }
      if (stepChanged) prepareStepMotion();
      refreshAppearance();
    },
    reset,
    dispose() {
      disposed = true;
      observer.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", pointerDown);
      renderer.domElement.removeEventListener("pointermove", pointerMove);
      renderer.domElement.removeEventListener("pointerup", pointerUp);
      renderer.domElement.removeEventListener("pointerleave", pointerLeave);
      renderer.domElement.removeEventListener("pointercancel", pointerCancel);
      clearContent();
      renderer.dispose();
      renderer.domElement.remove();
      labels.domElement.remove();
    },
  };
}
