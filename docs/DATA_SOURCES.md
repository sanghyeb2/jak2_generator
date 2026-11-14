# 데이터 출처와 전달 범위

## 대회 training

ChEMBL37 human JAK2 CHEMBL2971, 대회 제공6,029개. 원본 순서와 SMILES를 유지합니다. 데이터 출처의 CC BY-SA3.0 조건과 별도로 **대회 참가 목적 외 사용·외부 재배포 금지** 규칙을 따릅니다.

동일 파일을 `inputs/train.csv`와 `generators/data/train.csv`에 보관합니다. 전자는 선별용, 후자는 모델 학습용입니다. 실행 진입점은 두 파일의 해시가 같아야 진행합니다.

## Transfer용 외부503개

고정 파일: `generators/data/jak_transfer_candidates_503.csv`.

기존 수집 기록에 따르면 ChEMBL API의 human JAK1(CHEMBL2835), JAK3(CHEMBL2148), TYK2(CHEMBL3553)에서 정확한IC50≤1,000nM binding assay를 수집했습니다. 오류 주석·중복 구조·train 일치 구조를 제외하고, 당시 학습5,029개 기준 scaffold fold별 최근접Morgan–Tanimoto≥0.50 및 분자량 중앙95% 조건을 적용했습니다. 별도로 수집한 human JAK2 유효 고유16,025구조와 일치하는 후보를 제외해503개를 얻었다는 기록입니다.

API 응답 버전 기록은 ChEMBL_37입니다. 릴리스 고정 아카이브와 별도 대조하거나 이번 정리에서 재수집·누수 검사를 수행하지 않았습니다. 재학습에는 포함된 고정503개CSV를 사용합니다. 이 설명은 과거 기록이며 새로운 검증 결과가 아닙니다.

- 데이터베이스: https://www.ebi.ac.uk/chembl/
- 릴리스: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/
- 라이선스: https://creativecommons.org/licenses/by-sa/3.0/
- 고정 CSV SHA-256: `40e1dd47b7ceb2ba77b44bbceef16749961fe9e8b556ffbbcebf4a278849e143`.

## 후보와 제출물

후보는 다섯 생성모델의 저장 가중치에서 생성한 파일입니다. 기존 선별기나 최고점 분자의 점수를 가져온 후보가 아닙니다. source는 첫 등장 모델을 뜻하며, 중복된 분자의 모델 간 생성 기여도를 전부 대표하지는 않습니다.

데이터·분자 구조가 담긴 CSV·전체 입력 provenance는 공개 GitHub ZIP에 포함하지 않습니다. 교수님께 대회 참가·검증 목적의 비공개 패키지로 전달합니다.
