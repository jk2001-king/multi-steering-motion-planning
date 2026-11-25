# Workspace layout and original evidence

루트의 활성 구조는 `src`, `build`, `install`, `log`다. `archive`는 원본 결과와 정리 이전 자료를 보관하는 별도 영역이며 빌드 대상이 아니다.

| 이전 위치 | 새 위치 |
|---|---|
| `studies/IJCAS/scripts` | `src/ijcas_experiments/runtime/study` |
| `isaac/scripts` | `src/ijcas_experiments/runtime/sdk` |
| `isaac/assets`, `isaac/scenes` | `src/ijcas_experiments/assets/isaac` |
| `studies/IJCAS/config` | `src/ijcas_experiments/config/study` |
| `config/platforms`, `config/common` | `src/ijcas_experiments/config` |
| `docs`, `tools` | `src/ijcas_experiments/docs`, `tools` |
| 원본 `studies`, `data`, `migration`, `results` | `archive/original` 아래 원래 상대경로 |
| 새 실행/분석 출력 | `log/ijcas/experiments`, `log/ijcas/reproduced` |

옛 실행 로그·JSON·source ZIP·실패 기록의 bytes, run ID, 분모는 수정하지 않았다. 과거의 절대경로는 명시적인 `ijcas_paths` 읽기 뷰에서만 새 위치로 대응시킨다. 옛 workspace나 루트 호환 symlink를 필요로 하지 않는다. 원본 사본과 전체 파일 목록/해시는 `archive/original`과 `archive/layout_20261126`에 있다.

기존 제어 함수의 경로 결합 표현을 명시적인 경로 해석 함수로 기계적으로 옮겼다. 제어·계획 수식, 수치 파라미터, 임계값, 고정 실험 조건, 안전 판정과 논문 통계는 변경하지 않는다. 예전 source ZIP과 현재 소스의 호환성 확인에는 정리 전/후 전체 파일 hash로 검증한 원본 소스 뷰를 사용한다. 소스나 원본이 달라지면 검사는 실패하며 자동 허용하지 않는다.

```bash
cd /home/jk/Research/IJCAS_ws
source install/setup.bash
ros2 run ijcas_experiments check --full
ros2 launch ijcas_experiments experiment.launch.py scenario:=H2 method:=P condition:=3 stage:=check
```

기본 확인은 Isaac·물리·주행을 시작하지 않는다. `stage:=plan`은 SDK Python의 오프라인 계획/고정 경로 감사만 수행한다. 새 주행에는 기존과 같이 `stage:=run allow_motion:=true`를 명시한다. 이번 구조 변경 검증에서 새 주행이나 기존 결과 교체는 하지 않는다.

공개 저장소도 이 구조로 갱신한다. 기존 커밋과 원본 사용자 수정은 보존하며 강제 push나 이력 재작성은 하지 않는다. 보존 검사는 정리 전 Git 커밋이 이후 커밋의 조상으로 남아 있는지를 확인한다. 로컬 보존 검사는 원래 Git 디렉터리의 유지 여부도 확인하는 PC별 검사이며, 새 clone의 보편적 재현성 검사가 아니다. 원본 데이터 없는 코드 복제본에서는 자료 의존 확인·실행·결과 재계산을 수행할 수 없다.

공개 이후 정상 커밋을 허용하기 위한 검증기 자체의 변경은 `config/publication_source_revisions.json`에 이전/현재 SHA-256과 이유를 별도 기록했다. 불변 `archive/layout_20261126/source_manifest.json`은 고치지 않는다. 이 명시적 예외는 검증기 한 파일에만 적용되고 계획·제어·안전·지표 코드에는 적용되지 않으며, 임의 수정이나 이전 해시 불일치는 계속 거부한다.
