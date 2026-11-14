# 초기 자산과 환경

모든 자산은 이 패키지 안에 실파일로 복사했습니다. 상위 프로젝트 캐시나 심볼릭 링크를 필요로 하지 않습니다. 버전과 SHA-256은 `results/assets_manifest.json`에 기록합니다. 모델 ID와 revision은 원래 저장 기록에서 옮겼으며 이번에 온라인 조회를 다시 하지 않았습니다.

## GP-MoLFormer와 tokenizer

- 모델: https://huggingface.co/ibm-research/GP-MoLFormer-Uniq
- revision: `6eca879581e2302b4e1ab07bb02908636bddb4a2`
- 로컬: `assets/GP-MoLFormer-Uniq/`
- tokenizer: https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct
- revision: `361063d0ad524ef77cf39b08469f6be770dc550f`
- 로컬: `assets/MoLFormer-XL-both-10pct/`

모델의 공개 configuration/modeling Python 코드도 함께 보관했습니다. `trust_remote_code=True`이지만 실행은 로컬 자산과 offline 환경을 사용합니다. 학습된 최종 GP 가중치와 transfer 가중치는 별도 `generators/weights`에 있습니다.

## MolGPT

- https://huggingface.co/jonghyunlee1993/MolGPT_long_context_pretrained-by-ZINC15
- revision: `a52c3df13d9d67cf68f1977aca5dd7777cf88935`
- 초기 모델·tokenizer: `generators/weights/molgpt/pretrained/`
- 저장 학습 모델: `generators/weights/molgpt/checkpoint.pt`
- 라이브러리: `runtime/molgpt-runtime`의 Transformers4.46.3/tokenizers0.20.3.

## REINVENT

- 코드: https://github.com/MolecularAI/REINVENT4
- commit: `0f075b349ed22015e0406350c6f426701bb0815d`
- 초기 prior: https://zenodo.org/records/20701824
- 다운로드 원래 출처: https://zenodo.org/api/records/20701824/files/reinvent_pubchem.prior/content
- prior SHA-256: `fe8cd1678452ad292a8f93e97cb19a85959b729e17113e157180d5e69ae89ef3`
- 로컬: `generators/weights/reinvent/pretrained/reinvent_pubchem.prior`
- 필요한 공식 rnn.py/vocabulary.py와 LICENSE: 같은 폴더의 `source/`.

## ChemNet

fcd-torch1.0.7의 `ChemNet_v0.13_pretrained.pt`를 `assets/`에 복사했고 selector가 이 파일을 명시적으로 읽습니다. 기존 실험에서 사용한 SHA-256은 `a36658cc7c94d3c728eacd7ed2d2f104753bb2fde742c2817daefe9533a28080`입니다.

## 설치 환경과 runtime

기본 환경은 requirements.txt를 따릅니다. `runtime/rdkit-2025.9.6`에는 공식 제출 조건 검사에 쓸2025.09.6을 보관했습니다. main.py가 검사용 별도 프로세스에만 이 경로를 넣습니다. MolGPT도 별도 프로세스에서만 해당 Transformers overlay를 사용합니다.

runtime은 Windows/Python3.11 바이너리입니다. Python·CUDA·드라이버·기본 PyTorch 등은 설치가 필요합니다. 새 환경의 pip 설치 성공과 같은 결과 생성은 미검증입니다.

공개 소스 ZIP은 대형 바이너리를 포함하지 않습니다. 교수님께는 로컬 자산 묶음을 별도로 전달해야 합니다. 제3자 파일의 기존 저작권·LICENSE를 유지하며, 프로젝트 전체에 임의의 단일 라이선스를 부여하지 않았습니다.
