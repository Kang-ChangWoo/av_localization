# F3Loc Code Inventory for AV-FPLoc Design

> ChatGPT 전달용 baseline inventory입니다. F3Loc의 파일·함수·호출 경로를 정리하고, AV-FPLoc `track1_core`에서 재사용할 알고리즘과 새 wrapper가 필요한 경계를 구분합니다.

## 0. Scope and baseline provenance

- External baseline: `https://github.com/felix-ch/f3loc`
- Reviewed revision: `9e8027d9219ca505078283ebfd925580f476ab97`
- Relevant baseline root: original Track1 workspace의 `repos/f3loc/`
- AV-FPLoc target package: `track1_core/`
- DESDF는 코드상 “directional ESDF”이며, 구현상 각 위치·방향의 first-hit range field입니다. 일반적인 scalar signed Euclidean distance field와 동일한 개념으로 취급하면 안 됩니다.

전체 흐름:

~~~text
floorplan map.png
  → offline raycast_desdf()
  → scene/desdf.npy: (H, W, 36)
  → runtime localize(desdf, rays)
       → likelihood (H, W, 36)
       → likelihood_2d (H, W)
       → pose (x_grid, y_grid, yaw)
~~~

---

## 1. Pose grid 생성 함수와 호출 위치

### 핵심 사실

F3Loc에는 별도 `PoseGrid` 클래스가 없습니다. `desdf[grid_y, grid_x, yaw_bin]` 자체가 pose grid이자 floorplan geometry descriptor입니다.

| 역할 | 파일 | 함수 |
|---|---|---|
| 핵심 generator | `repos/f3loc/utils/generate_desdf.py:15` | `raycast_desdf(occ, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1)` |
| 단일 ray primitive | `repos/f3loc/utils/utils.py:58` | `ray_cast(occ, pos, ang, dist_max)` |
| Structured3D wrapper | `scripts/preprocess_structured3d_for_f3loc.py:286` | `write_desdf(scene_dir, desdf_root, overwrite=False)` |
| legacy S3D script | `repos/f3loc/s3d/process/get_desdf_s3d.py:44` | `raycast_desdf(..., orn_slice=36, max_dist=20, original_resolution=0.02)` |

### 생성 호출 경로

~~~text
preprocess_structured3d_for_f3loc.py:main()
  → floorplan_geometry()
  → write_map()
  → write_poses_map()
  → write_depth()
  → write_desdf()
       → raycast_desdf()
            → ray_cast()
       → np.save(scene/desdf.npy)
~~~

평가 시에는 이미 생성된 cache를 로드합니다.

- Single observation: `repos/f3loc/eval_observation.py:170–177`
- Filtering: `repos/f3loc/eval_filtering.py:311–318`
- Structured3D observation: `repos/f3loc/eval_s3d_observation.py:95–106`

---

## 2. Pose volume shape와 axis order

~~~text
desdf: (H, W, O)
O = 36
~~~

- `H`: grid row, array/map y
- `W`: grid column, array/map x
- `O`: yaw bin

표현별 순서는 다음과 같습니다.

| 표현 | 순서 |
|---|---|
| DESDF tensor | `[grid_y, grid_x, yaw_bin]` |
| `localize()` 반환 pose | `[x_grid, y_grid, yaw]` |
| `ray_cast()` position | `[row, col] = [y, x]` |
| yaw axis | 마지막 tensor axis |

`localize()`는 `torch.where`의 결과를 y,x로 받은 후 `[x,y,theta]`로 반환합니다. 따라서 array axis와 pose vector 순서를 그대로 동일시하면 안 됩니다.

---

## 3. x/y 및 yaw resolution

### Spatial resolution

기본 DESDF output resolution은 `0.1 m/cell`입니다.

| Dataset/path | 입력 map resolution | 환산 |
|---|---:|---:|
| Gibson baseline assumption | 0.01 m/pixel | 10 map pixels/cell |
| Structured3D adapter | 0.02 m/pixel | 5 map pixels/cell |

generic generator는 다음 ratio를 사용합니다.

~~~python
ratio = resolution / original_resolution
desdf_shape = floorplan_shape // ratio
~~~

### Yaw resolution

~~~text
O = 36
yaw(o) = o / 36 * 2π
1 bin = 10°
~~~

DESDF orientation axis는 counter-clockwise입니다. observation ray API는 left-to-right/clockwise 규약으로 설명되며, `localize()`가 입력 ray를 flip하여 정렬합니다.

### AV-FPLoc에서 새로 만들 변환 API

현재 F3Loc은 dataset별 수식을 evaluator 안에 inline으로 둡니다. `track1_core/floorplan/`에서 아래 API를 단일화해야 합니다.

~~~text
metric_pose_to_grid(pose_metric, map_metadata)
grid_pose_to_metric(pose_grid, map_metadata)
yaw_to_bin(yaw, O=36)
bin_to_yaw(yaw_bin, O=36)
~~~

metadata에는 source resolution, crop origin, handedness, yaw-zero convention이 포함되어야 합니다.

---

## 4. Valid pose mask 생성·적용 위치

### Baseline 상태

F3Loc에는 pose candidate용 valid mask가 없습니다.

`s3d/data_utils.py:111–119`의 `mask`는 roll/pitch gravity alignment용 image attention mask이며, pose mask가 아닙니다.

현재 동작:

- `raycast_desdf()`는 crop의 모든 row/column에서 ray를 생성합니다.
- 시작점이 wall/free cell인지 검사하지 않습니다.
- `localize()`는 전체 `(H,W,O)`를 score하고 global maximum을 선택합니다.
- free-space/traversability/valid-pose mask를 likelihood에 곱하지 않습니다.

### AV-FPLoc 권장 contract

~~~text
valid_pose_mask: bool, shape (H, W)
~~~

동일 mask를 다음에 모두 적용해야 합니다.

1. visual likelihood
2. acoustic likelihood
3. fused posterior
4. normalization
5. argmax/Top-K
6. GT rank 및 entropy diagnostics

invalid cell은 normalization 전에 zero probability 또는 negative-infinity log score로 처리해야 합니다. mask는 floorplan rasterization/downsampling 후 생성하며, wall clearance/erosion 정책을 문서화해야 합니다.

---

## 5. Floorplan ray-scan 생성 함수

### A. 단일 ray

`repos/f3loc/utils/utils.py:58`

~~~python
ray_cast(occ, pos, ang, dist_max=500) -> scalar pixel distance
~~~

- position: `[row, col]`
- angle: radians
- output: source-map pixel distance
- wall hit: first occupied pixel
- map boundary/no hit: `dist_max`
- invalid flag/NaN 없음

### B. Dense floorplan geometry volume

`repos/f3loc/utils/generate_desdf.py:15`

~~~python
raycast_desdf(occ, ...) -> ndarray(H, W, O)
~~~

모든 candidate spatial cell과 yaw bin에서 한 방향 ray를 쏘고 first-hit 거리를 저장합니다. 반환 array에 `original_resolution`을 곱하므로 stored value는 meter입니다.

### C. Per-camera offline depth files

`scripts/preprocess_structured3d_for_f3loc.py:253–283`

~~~python
ray_angles(ray_n) -> angle vector
write_depth(scene_dir, ray_n) -> depth{ray_n}.txt
~~~

camera pose마다 raycast radial range를 계산한 후 `radial_range * cos(center_angle)`로 camera-forward depth를 저장합니다. legacy 경로는 `repos/f3loc/s3d/process/get_ray_s3d.py`입니다.

---

## 6. Ray cache 저장 위치와 파일 형식

### DESDF cache

| 항목 | 형식 |
|---|---|
| 경로 | `<dataset_root>/desdf/<scene>/desdf.npy` |
| 파일 | NumPy `.npy`, pickled Python dict |
| 핵심 key | `payload["desdf"]`, shape `(H,W,36)` |
| S3D metadata | `payload["l"]`, `payload["t"]`, coordinate comment |
| 생성 방식 | offline 생성, evaluation에서 scene dict로 load |

로드 코드는 다음 형태입니다.

~~~python
np.load(path, allow_pickle=True).item()
~~~

평가 script는 보통 DESDF value가 10 m를 넘으면 10으로 clip합니다.

### Per-camera depth cache

| 파일 | 한 줄 | 값 |
|---|---|---|
| `depth40.txt` | camera pose 하나 | 40 camera-forward depth samples |
| `depth160.txt` | camera pose 하나 | 160 camera-forward depth samples |

각 줄은 space-separated meter 값입니다. 이것은 dense candidate-pose volume이 아니라 observation training target입니다.

Runtime에는 floorplan을 매 query마다 raycast하지 않고 DESDF cache를 load한 뒤 observation ray와 match합니다.

---

## 7. Ray angle, FoV, ordering, units, invalid handling

### Runtime visual ray

`repos/f3loc/utils/localization_utils.py:74`

~~~python
get_ray_from_depth(d, V=11, dv=10, a0=None, F_W=3/8)
~~~

기본 설정:

- `V=11`
- 간격 `10°`
- angle `[-50°, -40°, ..., 0°, ..., +50°]`
- image left-to-right ordering
- API 설명상 clockwise
- output은 `depth / cos(angle)`로 복원한 radial range meter

`F_W=3/8`은 horizontal FoV 약 106.26°에 해당하며, 11개 endpoint 간 span은 100°입니다.

### Structured3D offline convention

`scripts/preprocess_structured3d_for_f3loc.py:253–255` 및 legacy script는:

~~~python
F_W = 1 / tan(0.698132) / 2 ≈ 0.596
~~~

horizontal FoV 약 80°입니다. 따라서 S3D `depth40.txt`와 runtime `get_ray_from_depth(..., F_W=3/8)`를 같은 angular convention으로 가정하면 안 됩니다.

### Ordering and invalid rays

`localize()`는 입력 ray를 다음처럼 flip합니다.

~~~python
rays = torch.flip(rays, [0])
~~~

`ray_cast()`는 map boundary/no hit를 `dist_max`로 반환하며 validity bit를 만들지 않습니다. `get_ray_from_depth()`의 `griddata`도 custom sampling에서 NaN guard가 없습니다.

AV-FPLoc observation adapter는 최소한 다음을 명시해야 합니다.

~~~text
ray_values: float32 (V,)
ray_angles_rad: float32 (V,)
ray_order: explicit
ray_units: radial_meter or camera_forward_depth_meter
invalid_ray_mask: bool (V,)
field_of_view: radians
~~~

camera-forward depth는 DESDF와 직접 비교하기 전에 radial range로 변환해야 합니다.

---

## 8. Monocular output에서 visual likelihood까지의 호출 경로

### Network

- `repos/f3loc/modules/mono/depth_net.py:15`: `depth_net`
- `repos/f3loc/modules/mono/depth_net_pl.py:7`: `depth_net_pl`

입력 image width 640 기준 feature width는 약 40입니다. output은 batch 기준 `(N,fW)` depth profile입니다. depth hypothesis는:

~~~text
d_min = 0.1 m
d_max = 15.0 m
D = 128
d_hyp = -0.2
~~~

### Single observation path

~~~text
eval_observation.py
  → GridSeqDataset.__getitem__()
       → ref_img (C,H,W)
  → d_net.encoder(ref_img_torch, ref_mask_torch)
       → pred_depths
  → get_ray_from_depth(pred_depths)
       → pred_rays (11,)
  → localize(torch.tensor(desdf["desdf"]), pred_rays)
       → prob_vol_pred (H,W,36)
       → prob_dist_pred (H,W)
       → orientations_pred (H,W)
       → pose_pred (3,)
~~~

주요 위치:

- model setup: `eval_observation.py:116–164`
- depth inference: `294–305`
- ray conversion: `306–307`
- likelihood matching: `309–312`
- metric: `314–319`

### Matching equation

~~~text
score[y,x,o] = -L1(DESDF circular window - flip(ray_vector))
likelihood[y,x,o] = exp(score[y,x,o] / lambda)
V = 11
lambda = 40
~~~

`prob_vol`은 likelihood score volume이며 normalized posterior가 아닙니다. `prob_dist`는 yaw max-pool 결과입니다.

Filtering path:

~~~text
eval_filtering.py:490–509
  → get_ray_from_depth()
  → localize(..., return_np=False)
  → posterior = prior * likelihood
~~~

---

## 9. GT pose와 grid-index 변환

F3Loc에는 공통 metric/grid conversion module이 없습니다.

### Gibson

`eval_observation.py:191–204`:

~~~python
x_map = x_world / 0.01 + map_width / 2
y_map = y_world / 0.01 + map_height / 2
theta_map = theta_world
~~~

`eval_observation.py:253–256`:

~~~python
x_grid = (x_map - desdf["l"]) / 10
y_grid = (y_map - desdf["t"]) / 10
theta_grid = theta_map
~~~

prediction → map inverse는 `eval_filtering.py:529–534`에 inline입니다.

~~~python
x_map = x_grid * 10 + desdf["l"]
y_map = y_grid * 10 + desdf["t"]
~~~

### Structured3D

Reusable forward helper:

~~~python
repos/f3loc/eval_s3d_observation.py:18
desdf_pose_from_s3d_map_pose(pose_map, desdf)
~~~

S3D map resolution 0.02 m/px와 DESDF 0.1 m/cell을 사용하므로 scale은 5입니다.

~~~text
x_grid = (x_map - l) / 5
y_grid = (y_map - t) / 5
theta_grid = theta_map
~~~

Metric helper는 `eval_s3d_observation.py:30`의 `metric_from_s3d_prediction()`입니다. grid translation error에 0.1 m를 곱하고 wrapped yaw error를 degree로 계산합니다.

### GT oracle locations

- Gibson GT load: `eval_observation.py:179–204`
- Gibson single score: `eval_observation.py:314–319`
- Filtering GT conversion: `eval_filtering.py:320–405`
- Filtering GT transition: `eval_filtering.py:629–650`
- S3D GT: `S3DDataset.gt_poses` 및 `eval_s3d_observation.py:136–140`

`mvd`/`comp`는 `ref_pose/src_pose`를 model input으로 받으며, filtering transition도 GT pose에서 계산됩니다. AV-FPLoc 실험에서는 이 oracle 조건을 명시해야 합니다.

---

## 10. 재사용 함수와 새 wrapper 구분

### 알고리즘을 재사용할 후보

| Capability | F3Loc function | AV-FPLoc 처리 |
|---|---|---|
| 단일 occupancy ray | `utils.utils.ray_cast()` | 알고리즘 재사용, 단위/invalid metadata wrapper 필요 |
| dense directional geometry | `utils.generate_desdf.raycast_desdf()` | 알고리즘 재사용 또는 재구현, float32/metadata 추가 |
| likelihood matching | `utils.localization_utils.localize()` | matching 아이디어 재사용, mask/invalid ray 추가 |
| visual depth→ray | `get_ray_from_depth()` | visual-only adapter로 사용 |
| transition | `transit()`, `get_filters()` | sequential baseline에서 선택적 재사용 |
| relative pose | `get_rel_pose()` | dataset-neutral pose utility로 감싸기 |
| visualization | `floorplan_panel_from_map()`, `save_observation_figure()` | visual/acoustic/fused panel로 확장 |

### 새 wrapper 또는 새 구현이 필요한 항목

| Need | 이유 | Target |
|---|---|---|
| metric/grid conversion | baseline 수식이 evaluator에 inline | `track1_core/floorplan/` |
| valid pose mask | baseline에 없음 | `track1_core/floorplan/` |
| acoustic observation adapter | F3Loc에는 acoustic path 없음 | `track1_core/datasets/` 또는 `likelihood/` |
| acoustic likelihood | fixed visual ray convention을 가정 | `track1_core/likelihood/` |
| likelihood calibration | raw exp/L1 scale이 modality weight가 될 수 있음 | `track1_core/fusion/` |
| dataset annotation adapter | `floorplan_geometry()`는 Structured3D-specific | `track1_core/datasets/` |
| cache metadata/version | 기존 .npy dict의 implicit assumptions | `track1_core/floorplan/` |
| oracle protocol | GT pose가 model/transition path에도 사용됨 | `track1_core/evaluation/` |

### 권장 첫 구현

~~~text
floorplan + explicit metadata
  → pose grid + valid_pose_mask
GT pose + acoustic beam specification
  → oracle acoustic ray vector
pose grid + oracle rays + mask
  → acoustic likelihood (H,W,36)
~~~

그 다음 같은 candidate grid/support에서 다음을 비교합니다.

~~~text
visual-only likelihood
acoustic-only likelihood
calibrated visual × acoustic posterior
~~~

---

## ChatGPT 전달용 prompt

~~~text
첨부한 f3loc_code_inventory.md를 baseline source of truth로 사용하세요.
이번 대화에서는 AV-FPLoc acoustic geometry oracle 설계만 논의합니다.

반드시:
1. coordinate/ray/pose-volume contract를 먼저 요약하고,
2. baseline 재사용 알고리즘과 새 wrapper 경계를 구분하고,
3. valid-pose mask와 GT oracle leakage를 명시하고,
4. 2~3개 설계안의 trade-off를 제시하세요.

아직 구현 코드는 작성하지 말고 입력/출력 tensor contract와 검증 실험 설계까지만 제안하세요.
~~~

