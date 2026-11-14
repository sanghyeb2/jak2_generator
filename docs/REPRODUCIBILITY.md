# 재현 범위와 보존 정책

## 이미 완료된 원본 실행에 남아 있는 기록

고정5만 후보 → 고유39,167개 → 새 fullprior → neural을 수행한 원본은 `local_artifacts/runs/fresh_seed42`에 있습니다. fullprior300step/neural best75step이며 최종CSV SHA-256은 `23f6608f7952b496cbc3ddfd9064940e6e9e92e69deea0feb78d5c28ddce8268`입니다.

원본 complete/result에는 저장 fullprior logit→CSV 일치, neural 저장 모델의 실제 forward→CSV 일치, RDKit2025.09.6 유효2,000/고유2,000 검사가 PASS로 기록되어 있습니다. 원본 기록을 읽어 문서에 옮겼고 이번 정리에서 재실행하지 않았습니다.

## 이번 정리의 변경

- 폴더명 fresh_fullprior_neural_20261114 → 최종 제출 예정작.
- 완료 실행·원본 코드 스냅샷·수정 이력은 local_artifacts에 보관.
- 5개 생성모델 코드, training·transfer503·공개 초기 자산·최종 가중치·runtime을 내부에 복사.
- 과거 HPO/폐기 폴더를 읽는 경로 제거. 최종 선택된 하이퍼파라미터를 고정.
- GP의 Hugging Face 모델ID 로딩을 같은 revision의 로컬 자산 디렉터리 로딩으로 변경.
- 선별용 ChemNet과 공식 RDKit 경로를 이 폴더 안으로 변경.
- 균일3시드 생성 루프를 기존 실험에서 추출하고 모델별 실행 설정·해시 저장을 추가.
- 실행 진입점 main.py와 비공개 자료 제외용 GitHub 소스 내보내기 추가.

계산 목적의 loss·모델 크기·최종 학습epoch를 바꾸려는 정리는 아닙니다. 다만 코드·경로·원격 모듈 로딩 위치가 달라졌으므로 과거 실행의 성공을 새 코드의 검증 성공으로 간주하면 안 됩니다.

## 미검증

1. 새 패키지를 별도 컴퓨터로 옮겨 설치하고 실행하기.
2. 공개 초기 자산부터 다섯 모델을 재학습해 저장 weight 텐서가 일치하는지 확인하기.
3. 각 모델의 후보10,000개 내용·순서·seed quota가 원본과 일치하는지 확인하기.
4. 새 패키지로 fullprior·neural을 독립 재학습해 최종CSV가 일치하는지 확인하기.

이번에는 사용자 요청대로 학습·생성·재현 검증을 실행하지 않았습니다. 포장용 파일 존재·구문·복사본 해시 확인은 학습 재현성 검증과 다릅니다.

## 보관과 재개

최종 제출CSV·원래 weight·후보·로그는 덮어쓰지 않습니다. 원본 config와 log의 이전 절대경로는 이력으로 보존하며, 새 코드에서 원본 run을 resume하지 않습니다.

새 전체 학습은 --stage all을 명시할 때만 시작합니다. 입력·코드·runtime 변경 시 새 run-name을 사용합니다. GRU와 GP/transfer의 중단 학습 재개 한계는 README에 명시했습니다.

데이터를 포함한 로컬 폴더가 실행 패키지이고 GitHub 소스 ZIP은 그 일부입니다. 전체 학습 재현과 저장 weight 재추론을 보고서에서도 구분합니다.
