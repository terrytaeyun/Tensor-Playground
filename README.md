# Tensor Playground

기존 Vite + React + TypeScript + Three.js + KaTeX 프로젝트 위에 만든 Tensor Builder입니다.

## 실행

Node.js 24 LTS 권장 (검증: 24.19.0).

```sh
npm install
npm run dev -- --host 127.0.0.1
```

터미널의 로컬 주소를 엽니다. 기본 주소: http://127.0.0.1:5173/

```sh
npm run build
npm run lint
npm test
```

## 현재 가능한 동작

- Axis 1 → Axis 2 → Axis 3 순서로 활성화. 앞 축을 끄면 뒤 축도 꺼집니다.
- 활성 축이 없으면 scalar, 1개면 vector, 2개면 matrix, 3개면 matrix-slice stack입니다.
- 각 축 길이는 1–6이며 입력 즉시 3D와 수식이 바뀝니다. 빈 값/범위 밖 값은 모델에 적용하지 않고 포커스를 옮기면 마지막 유효 값으로 돌아갑니다.
- 기본 shape (2, 3, 4)는 3 × 4 행렬 두 장입니다. Axis 1 = slice, Axis 2 = row, Axis 3 = column. 모든 화면 인덱스는 1부터 시작합니다.
- Symbolic: 각 셀에 성분 기호. Numeric: 실제 model values 표시. 초기값 1, 2, …, 선택한 성분 값 편집 가능.
- Explore: 셀 클릭으로 성분과 축별 인덱스 선택. 슬라이스 라벨/면/오른쪽 목록으로 행렬 선택.
- Slice: 셀 또는 행렬 면을 클릭하면 해당 슬라이스 선택.
- 선택한 슬라이스는 윤곽/투명도로 강조됩니다. 선택한 슬라이스의 셀은 앞쪽의 흐린 슬라이스 너머로도 선택할 수 있습니다.
- 드래그 회전, 휠 확대/축소, 오른쪽 드래그 이동, Reset View. 카메라 조작과 표시 모드 변경은 선택을 유지합니다.
- Clear selection으로 전체 텐서 보기. shape 변경 시 범위를 벗어난 선택만 해제합니다.
- 같은 order에서 shape를 바꾸면 겹치는 좌표의 숫자 값은 보존합니다. 새로고침 시 기본 모델로 돌아갑니다.
- 그 외 15개 operation은 Coming next로 비활성 표시하며 계산은 구현하지 않았습니다.

## 구조

- `src/model/tensor.ts`: TensorModel 생성, 순차 축 설정, row-major 값/인덱스 매핑, 값 수정과 선택 검증. React/Three.js와 독립.
- `src/types/index.ts`: 모델, 축 metadata, notation, selection, operation 타입.
- `src/utils/notation.ts`: 일반/구체 차원, 성분, 슬라이스의 LaTeX 표기. 향후 공유 인덱스를 위한 별도 notation model.
- `src/operations/`: Explore, Slice 모듈과 확장용 operation registry.
- `src/visualization/createTensorScene.ts`: Three.js 렌더링, CSS2D/KaTeX 라벨, raycasting, 카메라, 리소스 정리. 텐서 계산 없음.
- `src/components/TensorVisualizer.tsx`: 모델과 렌더러 연결, 일반 notation, 카메라 제어.
- `src/components/ControlPanel.tsx`: 축 토글/길이, 표시 모드, 슬라이스 목록, 값 편집, operation 선택.
- `src/components/FormulaPanel.tsx`: 구체 shape와 선택한 슬라이스/성분 수식.
- `src/components/MathText.tsx`: React용 KaTeX 렌더링.
- `src/App.tsx`: 모델·선택·모드·operation 상태 연결.
- `src/index.css`: 카드 없는 흰 캔버스, 데스크톱 좌우 배치 및 모바일 세로 배치.
- `tests/tensor.test.mjs`: 모델/인덱싱/수식/operation 회귀 테스트. Node 기본 test runner 사용.

새 operation은 `src/operations/`에 모듈을 추가하고 registry에 등록합니다. 계산은 model/독립 operation 모듈에서 수행하고 시각화에는 결과 모델을 전달하는 구조를 유지합니다.

WebGL 지원 브라우저가 필요합니다. Three.js와 KaTeX가 포함되어 build 시 번들 크기 경고가 나올 수 있으나 build 오류는 아닙니다.
