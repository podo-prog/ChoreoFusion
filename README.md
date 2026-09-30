# ChoreoFusion

BVH·FBX 댄스 모션에서 동작 구간의 경계를 찾고, annotation JSON으로 내보내는 Python 프로젝트입니다. 이 프로젝트의 자동 분할 모델은 **동작 이름이나 의미를 붙이지 않습니다.** 사람이 표시한 시작·끝 시각을 학습해 구간을 생성하며, 결과 JSON의 각 `label`은 빈 문자열로 둡니다.

> 이 저장소는 공개입니다. 프로젝트 코드와 문서만 포함합니다. 학습 데이터, annotation 원본, 예측 JSON, 모델 체크포인트와 가상환경은 용량·데이터 관리 목적으로 포함하지 않습니다. 독립 실행에는 별도로 보관한 모델 가중치와 사용 권한이 있는 BVH/FBX·annotation 파일이 필요합니다.

## 프로젝트 구성

- `choreofusion/annotations.py` — annotation JSON에서 `전체 동작 설명` 레이어를 찾고 시작·끝 시간만 읽습니다. 자연어 설명·태그·S/H/R/E 값은 경계 목표로 사용하지 않습니다.
- `choreofusion/motion.py` — BVH 모션을 읽고 이미 나눈 구간을 고정 길이 특징으로 바꿉니다. 기존 군집 기능에서 사용합니다.
- `choreofusion/cluster.py` — 이미 분리된 구간을 특징화하고 PCA·K-means로 묶는 기존 기능입니다. 자동 경계 탐지 모델과는 별도입니다.
- `choreofusion/boundary.py` — 단일 Dilated TCN 학습·fine-tuning·추론, BVH/FBX 특징 계산, 경계 디코딩과 JSON 생성을 담당합니다.
- `choreofusion/actionformer_boundary.py` — ActionFormer 스타일의 시간 구간 제안 모델 실험입니다.
- `choreofusion/mstcnpp_stgcn_boundary.py` — 최종 선택 모델인 120Hz MS-TCN++ 스타일 경계 모델과 ST-GCN 경계 점수 보조 모델의 학습·추론을 담당합니다.
- `choreofusion/__main__.py` — `python -m choreofusion` CLI 진입점입니다.
- `requirements.txt` — Python 실행 의존성입니다.
- `PROJECT_EXPERIMENTS.md` — 데이터 결정, 모델 실험, 검증 수치와 알려진 한계의 상세 기록입니다.

## 데이터와 학습 목표

모델 준비에는 사람이 구간을 나눈 약 29곡의 BVH·FBX와 시간 범위를 담은 JSON을 사용했습니다. 원본 영상은 사용할 수 없는 조건이어서 영상 픽셀 모델은 학습하지 않았습니다. WAV와 PKL도 구간 경계 모델의 입력에 포함하지 않았습니다.

JSON에서는 `전체 동작 설명` 레이어의 `start`·`end` 값만 읽습니다. 설명 문장과 태그는 무시합니다. 입력 관절 골격은 학습 자료와 호환되는 공통 51관절 구조여야 합니다. 임의의 다른 골격을 자동으로 재매핑하는 기능은 없습니다.

BVH와 FBX가 같은 곡에서 나온 짝이면 항상 같은 데이터 분할에 둡니다. 최종 모델 실험의 분할은 학습 18곡, 검증 5곡, 보류 6곡이었습니다. 검증에서 모델·디코더 설정을 선택한 뒤 29곡 전체를 사용해 최종 체크포인트를 다시 맞췄습니다.

## 모델 실험과 현재 선택

프로젝트에 기록된 다섯 버전은 다음과 같습니다.

1. Dilated TCN — 30Hz, 48채널·7개 dilation 층. ChillKill 수동 검토 출력은 28구간.
2. Dilated TCN — 60Hz, 48채널·8개 층. ChillKill 출력은 26구간.
3. Dilated TCN — 120Hz, 48채널·9개 층.
4. ActionFormer 스타일 temporal localization 모델 — 120Hz. ChillKill 출력은 25구간.
5. **최종 선택: MS-TCN++ 스타일 경계 모델 + ST-GCN 경계 점수 보조 모델 — 120Hz.** Crazy 출력은 27구간, ChillKill 출력은 31구간.

다섯 번째 모델은 이름이 알려진 동작을 맞히지 않습니다. MS-TCN++ 스타일 네 단계 temporal convolution 모델이 프레임별 경계 점수를 예측하고, ST-GCN이 51관절의 공간·시간 변화를 반영해 별도의 경계 점수를 냅니다. 두 점수를 결합한 뒤 threshold와 최소 간격으로 경계 후보를 정리합니다. 입력 특징은 학습 보고서 기준 414차원입니다. MS-TCN++ 부분은 48채널, 단계마다 7개 temporal convolution 층으로 구성됩니다.

세부 수치와 실험의 주의점은 [PROJECT_EXPERIMENTS.md](PROJECT_EXPERIMENTS.md)에 정리했습니다.

## 최종 모델 설정

- 시간축: 120Hz
- 경계 목표 폭: 0.075초
- 기본 경계 점수 threshold: 0.30
- 인접 경계 최소 간격: 0.375초
- 기본 일괄 시간 오프셋: 0초
- 평가 허용오차: 정답과 예측을 가장 가까운 120Hz 프레임으로 매핑한 뒤 같은 프레임인 경우에만 일치 처리

0.375초 간격은 과도하게 촘촘한 후보를 줄이지만, 그보다 짧은 실제 동작도 있을 수 있어 놓칠 위험이 있습니다. 사용자 기준에서 경계 정확도는 엄격합니다.

## 설치

Python 3.12 환경을 권장합니다. 프로젝트 루트에서:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m choreofusion --help
```

FBX 읽기에는 `pufbx`가 사용됩니다. 학습 자료와 새 입력은 BVH·FBX에서 같은 순서로 추출되는 호환 관절 이름을 가져야 합니다.

## 새 모션 분할

먼저 MS-TCN++와 ST-GCN 체크포인트를 프로젝트 바깥의 안전한 저장 위치에서 준비합니다. 체크포인트는 이 저장소에 업로드하지 않았습니다. 아래 경로는 로컬 출력 폴더 구조 예시이며, 실제 가중치 위치에 맞게 바꿉니다.

```bash
python -m choreofusion segment \
  --model outputs/boundary/05_MS-TCNpp_ST-GCN_120Hz_29songs/models/ms_tcnpp_boundary_120hz_29songs.pt \
  --st-model outputs/boundary/05_MS-TCNpp_ST-GCN_120Hz_29songs/models/st_gcn_boundary_verifier_120hz_29songs.pt \
  --input /path/to/new_motion.bvh \
  --input /path/to/new_motion.fbx \
  --template /path/to/annotation_template.json \
  --output outputs/predicted.annotations.json
```

모션 형식 하나만 있으면 `--input`을 한 번 지정하면 됩니다. BVH와 FBX 짝을 모두 주면 두 모달리티의 점수를 시간축에 맞춰 평균 결합합니다. `--template`은 원래 annotation JSON 구조와 메타데이터를 바탕으로 결과를 만들 때 지정합니다. 출력은 `전체 동작 설명` 레이어에 빈 라벨 구간을 씁니다. `--threshold`, `--min-gap`, `--boundary-offset`으로 디코딩 값을 조정할 수 있습니다.

## 학습

비공개 데이터가 별도 폴더에 준비되어 있어야 합니다. 최종 조합 학습 CLI는 다음과 같습니다.

```bash
python -m choreofusion.mstcnpp_stgcn_boundary train \
  --data-dir /path/to/private_training_data \
  --output-dir outputs/boundary/05_MS-TCNpp_ST-GCN_120Hz_29songs
```

학습 자료 폴더에는 곡별 BVH·FBX와 annotation JSON이 있어야 합니다. JSON `videoTitle`과 파일명으로 짝을 찾고, 시간 단위는 초입니다. 데이터 경로는 명령행에서 지정하며 개인 컴퓨터 경로는 코드에 고정하지 않습니다. 실행 결과에는 학습 로그와 체크포인트가 만들어집니다.

옛 단일 TCN의 학습과 이어 학습은 `python -m choreofusion train-boundary` 및 `python -m choreofusion fine-tune-boundary`로 제공합니다. ActionFormer 실험은 `python -m choreofusion.actionformer_boundary`의 `train`·`predict` 명령을 사용합니다.

## 검증 결과와 해석

최종 모델의 정확 프레임 기준 내부 macro F1은 검증 5곡 3.1%, 보류 6곡 2.9%였습니다. 0 허용오차를 120Hz 격자에서 적용한 엄격한 기준이며, 현재 경계 예측이 원하는 재현 수준에 도달하지 못했음을 보여줍니다. Crazy·ChillKill 출력 개수는 수동 검토용 결과 수일 뿐 정확도 점수가 아닙니다.

ChillKill의 사용자가 고친 정답은 최초 31구간 예측에서 27구간으로 수정됐습니다. 20개 경계를 유지하고 6개를 이동, 4개를 삭제했습니다. 예측을 수정해 만든 정답이므로 독립적인 테스트 자료가 아닙니다. 수정본은 현재 체크포인트에 재학습 반영하지 않았습니다.

## 저장소에 포함하지 않은 파일

`data/`의 원본 BVH·FBX·annotation JSON, WAV·PKL, `outputs/`의 예측 JSON·학습 보고서·체크포인트, `.venv/` 가상환경은 저장소에서 제외했습니다. 이 파일들은 크기와 데이터 관리가 코드와 다르며, annotation에는 사용자가 작성한 데이터가 들어갈 수 있습니다. 실행 전에는 해당 데이터를 사용할 권한과 별도 보관 상태를 확인해야 합니다.
