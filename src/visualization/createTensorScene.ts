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
}
interface SliceView {
  index: number;
  plane: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  outline: THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>;
  label: CSS2DObject;
}
interface Hit {
  cell?: CellView;
  slice?: number;
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
  let cells: CellView[] = [];
  let slices: SliceView[] = [];
  let targets: THREE.Object3D[] = [];
  let hovered: Hit | null = null;
  let down: { x: number; y: number; id: number; dragged: boolean } | null =
    null;
  let shapeKey = "";
  let disposed = false;
  let sceneRadius = 3;

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
      };
      mesh.userData.cell = view;
      content.add(mesh);
      cells.push(view);
      targets.push(mesh);
    }
    reset();
  }
  function refreshAppearance() {
    if (!model) return;
    const selectedKey = selection.component?.join(",");
    const dense = cells.length > 64;
    for (const cell of cells) {
      const selected =
        selectedKey !== undefined && cell.indices.join(",") === selectedKey;
      const hover = hovered?.cell === cell;
      const sliceSelected =
        selection.slice !== null && selection.slice === cell.slice;
      const muted = selection.slice !== null && selection.slice !== cell.slice;
      cell.mesh.material.color.set(
        selected ? "#527aaf" : hover ? "#8baacc" : "#b9c9d9",
      );
      cell.mesh.material.opacity = selected
        ? 0.65
        : hover
          ? 0.48
          : muted
            ? 0.055
            : sliceSelected
              ? 0.32
              : 0.2;
      cell.outline.material.color.set(
        selected || hover ? "#3b6395" : "#8197af",
      );
      cell.outline.material.opacity =
        selected || hover ? 0.95 : muted ? 0.14 : sliceSelected ? 0.65 : 0.34;
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
    for (const slice of slices) {
      const selected = selection.slice === slice.index;
      const hover = hovered?.slice === slice.index && !hovered.cell;
      slice.plane.material.opacity = selected ? 0.12 : hover ? 0.09 : 0.035;
      slice.outline.material.color.set(selected ? "#3b6395" : "#7288a5");
      slice.outline.material.opacity = selected ? 0.95 : hover ? 0.7 : 0.3;
      slice.label.element.classList.toggle("is-selected", selected);
      slice.label.element.setAttribute("aria-pressed", String(selected));
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
    ) {
      model = nextModel;
      mode = nextMode;
      selection = nextSelection;
      interaction = nextInteraction;
      const key = model.shape.join(",") + `:${model.order}`;
      if (key !== shapeKey) {
        shapeKey = key;
        rebuild();
      }
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
