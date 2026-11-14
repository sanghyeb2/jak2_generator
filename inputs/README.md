# 고정 선별 입력 (CSV는 GitHub 제외)

- `train.csv`: 대회 training6,029개 원본 순서.
- `candidates/gp.csv`, `gru.csv`, `transfer.csv`, `molgpt.csv`, `reinvent.csv`: 모델당10,000개.
- 각 후보 CSV는 smiles/key/seed 열을 포함합니다. seed123/124/125 quota3,334/3,333/3,333입니다.
- 이 고정50,000개는 중복 제거 후39,167개가 되었고, 실제60.52285점 선별 실험에 사용되었습니다.
- `provenance.json`은 원래 파일 위치를 보존한 로컬 이력입니다. 코드가 그 원본 경로를 읽지는 않습니다.

CSV와 전체 provenance는 비공개 로컬 패키지로만 전달합니다. 공개 해시는 results/assets_manifest.json에 있습니다.
새로 생성한 후보는 inputs를 덮어쓰지 않고 runs/<run-name>/generation/candidates에 저장합니다.
