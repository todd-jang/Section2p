# Section2pSection 2에서 머신러닝 파이프라인을 구축하는 방법에 대해 공부하셨습니다.
Section 2에서 다양한 데이터셋을 통해 머신러닝 방법론을 공부했습니다. 데이터셋은 익숙한 데이터부터 전혀 도메인 지식이 없는 데이터까지 모두 다뤄봤습니다.
이번 프로젝트에선 장기훈님이 직접 선택한 데이터셋을 사용해서 만든 머신러닝 예측 모델을 통한 성능 및 인사이트를 도출/공유하는 것이 목표 입니다.

프로젝트 목표🔥
Section 2 프로젝트의 목표는 개요에서 알려드린대로

데이터셋을 사용하여 머신러닝 모델을 만든 후 성능 및 인사이트를 도출 / 공유하는 것

을 데이터셋 선정부터 모델 해석까지의 결과로 보여주는 것이 목표입니다.

이러한 데이터셋 전처리/EDA부터 모델을 해석하는 과정을 colab을 사용하여 작성하고, 해당 내용을 기반으로 설명하는 영상을 작성하는 것이 이번 프로젝트 기간 동안 장기훈님이 수행하셔야 하는 태스크입니다. 당연히 어떠한 코드, 분석, 라이브러리 혹은 목표를 가지는지는 온전히 장기훈님의 자유 입니다. 그러나, Section 2 Project인 만큼 해당 기간 동안 배운 내용을 위주로 설정하는 것이 권장됩니다.

추가로, 장기훈님의 발표를 듣는 사람은 비데이터 직군이라 가정합니다.
즉 장기훈님의 생각이나 가정들을 설명하는 과정에서 최대한 배경지식이 없는 사람들도 이해할 수 있도록 노력하시기 바랍니다!

프로젝트 절차 ⏰
프로젝트에 포함되어야 할 하위 태스크와 권장 기한을 안내드립니다. 프로젝트 발표 흐름은 자유이지만 아래 내용이 반드시 포함되어야 합니다!

1. 데이터 선정 이유 및 문제 정의 (1 DAY)
프로젝트에 사용할 데이터셋을 선정하고, 데이터셋으로부터 해결하고자 하는 문제를 정의합니다.
데이터 기반의 사고방식, data-driven의 마음가짐을 section1과 2에서 배웠습니다. 이번에는 그것들을 심화시켜서 문제 해결을 시도 해봅니다.

이 과정을 통해 장기훈님,

해결하고자 하는 문제가 적용되는 시나리오를 제시합니다.
문제 해결의 필요성에 대해 명확히 제시합니다.
데이터셋 선정이 끝났다면 어떤 유형의 문제로 (분류 / 회귀) 접근할지도 결정합니다.
데이터셋에서 해결하고자 하는 문제 정의에 적절한 타겟을 선정합니다.
타겟을 잘 설명할 수 있을 만한 특성에 대한 가설을 설정합니다.
❗ 현실의 문제를 머신러닝의 언어로 정의하는 능력은 머신러닝 개발자의 필수 역량입니다. 부트캠프 내 기존에 사용된 데이터셋이나 캐글 등에서 이미 정의된 문제를 그대로 사용하는 것은 지양해 주세요.

태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다.
내가 세운 가설이 문제 해결에 의미가 있나요?
해당 특성을 target으로 지정한 이유를 설명하세요
2. 데이터 전처리, EDA, 시각화(1.5 DAY)
데이터셋을 머신러닝 모델링에 적합하게 전처리하고, 피쳐의 분포 / 피쳐와 타겟 간의 상관관계 등에 대한 EDA를 진행합니다.
진행 후 얻은 데이터에 대한 인사이트를 시각화를 통해 제시합니다.

이 과정을 통해 장기훈님,

머신러닝에 적합한 형태로 데이터 타입을 변경합니다.
여러 테크닉을 사용해 결측치를 적절히 처리합니다.
본인의 전처리 과정에 대해 근거를 들어 설명합니다.
피쳐의 분포를 확인하고 이상치를 적절히 제거합니다.
1에서 세운 가설과 관련한 시각화 결과를 제시합니다.
❗ 데이터의 전처리 및 cleaning 과정은 모델링 성능에 큰 영향을 줍니다. 장기훈님이 선정한 Task에 맞게 다양한 처리 방법을 조사 후 적용해 보세요.

태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다
Data Leakage가 있었나요? 없었다면 어떻게 방지했나요?
특성과 타겟의 관계 및 가설을 충분히 설명할 수 있는 시각화 결과를 제시했나요?
3. 모델링 및 모델 해석(1 DAY)
문제의 유형(분류 / 회귀)에 맞게 후보 모델군을 선정하고, 해당 모델군에서 최적 모델을 학습 후 성능을 확인합니다.
모델 해석을 위한 여러 수치적 지표(Feature Importances) 및 시각화 기법을 사용하여 모델의 작동을 해석합니다.

이 과정을 통해 장기훈님,
baseline 모델을 선정하고 이에 기반해 모델링 성능을 평가합니다.
반복적인 Feature Engineering 과정을 통해 모델 성능을 최적화합니다.
문제 정의에 맞는 평가 지표를 선택하여 모델의 성능을 설명합니다.
교차검증이나 hold-out 기법을 사용해 학습 성능과 일반화 성능을 구분하여 제시합니다.
여러 수치적 지표 및 permutation importance, pdp, shap 등을 활용하여 최종모델을 설명합니다. 시각화는 설명이 가장 중요합니다.
1에서 제시한 가설을 해결합니다.
태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다.
모델을 학습한 후에 베이스라인보다 잘 나왔나요? 그렇지 않다면 그 이유는 무엇일까요?
모델 성능 개선을 위해 어떤 방법을 적용했나요? 그 방법을 선택한 이유는 무엇인가요?
최종 모델과 일반화 성능에 관해 설명하세요.
모델이 관측치를 예측하기 위해서 어떤 특성을 활용했나요?
4. 발표자료 준비 및 제출(0.5 DAY)
프로젝트의 결과를 정리하여 발표자료를 제작합니다.
청자가 비데이터직군이라 가정하고, 본인의 프로젝트 논리 전개 과정을 꼼꼼히 담아 보세요.

이 과정을 통해 장기훈님,은
설정한 가설의 해소 과정을 논리적으로 제시합니다.
본인의 프로젝트를 회고하고 보완할 점을 제시합니다
Section 2에서 머신러닝 파이프라인을 구축하는 방법에 대해 공부하셨습니다.
Section 2에서 다양한 데이터셋을 통해 머신러닝 방법론을 공부했습니다. 데이터셋은 익숙한 데이터부터 전혀 도메인 지식이 없는 데이터까지 모두 다뤄봤습니다.
이번 프로젝트에선 장기훈님이 직접 선택한 데이터셋을 사용해서 만든 머신러닝 예측 모델을 통한 성능 및 인사이트를 도출/공유하는 것이 목표 입니다.

프로젝트 목표🔥
Section 2 프로젝트의 목표는 개요에서 알려드린대로

데이터셋을 사용하여 머신러닝 모델을 만든 후 성능 및 인사이트를 도출 / 공유하는 것

을 데이터셋 선정부터 모델 해석까지의 결과로 보여주는 것이 목표입니다.

이러한 데이터셋 전처리/EDA부터 모델을 해석하는 과정을 colab을 사용하여 작성하고, 해당 내용을 기반으로 설명하는 영상을 작성하는 것이 이번 프로젝트 기간 동안 장기훈님이 수행하셔야 하는 태스크입니다. 당연히 어떠한 코드, 분석, 라이브러리 혹은 목표를 가지는지는 온전히 장기훈님의 자유 입니다. 그러나, Section 2 Project인 만큼 해당 기간 동안 배운 내용을 위주로 설정하는 것이 권장됩니다.

추가로, 장기훈님의 발표를 듣는 사람은 비데이터 직군이라 가정합니다.
즉 장기훈님의 생각이나 가정들을 설명하는 과정에서 최대한 배경지식이 없는 사람들도 이해할 수 있도록 노력하시기 바랍니다!

프로젝트 절차 ⏰
프로젝트에 포함되어야 할 하위 태스크와 권장 기한을 안내드립니다. 프로젝트 발표 흐름은 자유이지만 아래 내용이 반드시 포함되어야 합니다!

1. 데이터 선정 이유 및 문제 정의 (1 DAY)
프로젝트에 사용할 데이터셋을 선정하고, 데이터셋으로부터 해결하고자 하는 문제를 정의합니다.
데이터 기반의 사고방식, data-driven의 마음가짐을 section1과 2에서 배웠습니다. 이번에는 그것들을 심화시켜서 문제 해결을 시도 해봅니다.

이 과정을 통해 장기훈님,

해결하고자 하는 문제가 적용되는 시나리오를 제시합니다.
문제 해결의 필요성에 대해 명확히 제시합니다.
데이터셋 선정이 끝났다면 어떤 유형의 문제로 (분류 / 회귀) 접근할지도 결정합니다.
데이터셋에서 해결하고자 하는 문제 정의에 적절한 타겟을 선정합니다.
타겟을 잘 설명할 수 있을 만한 특성에 대한 가설을 설정합니다.
❗ 현실의 문제를 머신러닝의 언어로 정의하는 능력은 머신러닝 개발자의 필수 역량입니다. 부트캠프 내 기존에 사용된 데이터셋이나 캐글 등에서 이미 정의된 문제를 그대로 사용하는 것은 지양해 주세요.

태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다.
내가 세운 가설이 문제 해결에 의미가 있나요?
해당 특성을 target으로 지정한 이유를 설명하세요
2. 데이터 전처리, EDA, 시각화(1.5 DAY)
데이터셋을 머신러닝 모델링에 적합하게 전처리하고, 피쳐의 분포 / 피쳐와 타겟 간의 상관관계 등에 대한 EDA를 진행합니다.
진행 후 얻은 데이터에 대한 인사이트를 시각화를 통해 제시합니다.

이 과정을 통해 장기훈님,

머신러닝에 적합한 형태로 데이터 타입을 변경합니다.
여러 테크닉을 사용해 결측치를 적절히 처리합니다.
본인의 전처리 과정에 대해 근거를 들어 설명합니다.
피쳐의 분포를 확인하고 이상치를 적절히 제거합니다.
1에서 세운 가설과 관련한 시각화 결과를 제시합니다.
❗ 데이터의 전처리 및 cleaning 과정은 모델링 성능에 큰 영향을 줍니다. 장기훈님이 선정한 Task에 맞게 다양한 처리 방법을 조사 후 적용해 보세요.

태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다
Data Leakage가 있었나요? 없었다면 어떻게 방지했나요?
특성과 타겟의 관계 및 가설을 충분히 설명할 수 있는 시각화 결과를 제시했나요?
3. 모델링 및 모델 해석(1 DAY)
문제의 유형(분류 / 회귀)에 맞게 후보 모델군을 선정하고, 해당 모델군에서 최적 모델을 학습 후 성능을 확인합니다.
모델 해석을 위한 여러 수치적 지표(Feature Importances) 및 시각화 기법을 사용하여 모델의 작동을 해석합니다.

이 과정을 통해 장기훈님,
baseline 모델을 선정하고 이에 기반해 모델링 성능을 평가합니다.
반복적인 Feature Engineering 과정을 통해 모델 성능을 최적화합니다.
문제 정의에 맞는 평가 지표를 선택하여 모델의 성능을 설명합니다.
교차검증이나 hold-out 기법을 사용해 학습 성능과 일반화 성능을 구분하여 제시합니다.
여러 수치적 지표 및 permutation importance, pdp, shap 등을 활용하여 최종모델을 설명합니다. 시각화는 설명이 가장 중요합니다.
1에서 제시한 가설을 해결합니다.
태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다.
모델을 학습한 후에 베이스라인보다 잘 나왔나요? 그렇지 않다면 그 이유는 무엇일까요?
모델 성능 개선을 위해 어떤 방법을 적용했나요? 그 방법을 선택한 이유는 무엇인가요?
최종 모델과 일반화 성능에 관해 설명하세요.
모델이 관측치를 예측하기 위해서 어떤 특성을 활용했나요?
4. 발표자료 준비 및 제출(0.5 DAY)
프로젝트의 결과를 정리하여 발표자료를 제작합니다.
청자가 비데이터직군이라 가정하고, 본인의 프로젝트 논리 전개 과정을 꼼꼼히 담아 보세요.

이 과정을 통해 장기훈님,은
설정한 가설의 해소 과정을 논리적으로 제시합니다.
본인의 프로젝트를 회고하고 보완할 점을 제시합니다

프로젝트 체크리스트
완성도 높은 프로젝트를 위해 몇 가지 필수 체크리스트를 제공해 드립니다. 본인의 프로젝트 완성도를 점검해 보세요! 😎

Part1

데이터 셋 선정
 부트캠프 내 데이터셋이 아닌 다른 데이터셋을 활용하였다.
문제 정의
 부트캠프 내 문제가 아닌 다른 문제를 정의하고 데이터셋으로부터 문제 정의 과정을 제시하였다.
 문제 정의에 따른 적절한 타겟을 설정하였다.
EDA&전처리
 데이터 분석 과정과 결과를 설명하였다.
 데이터를 살펴보고 전처리를 한 과정이 드러나 있다.
Part 2

모델 학습 및 검증
 모델링 이전에 미리 baseline 모델을 선정하였다.
 문제 정의에서 언급한 회귀 / 분류 문제에 맞는 모델을 선택하였다.
 회귀/분류 문제에 따른 적절한 평가지표를 선택하였다.
 교차검증이나 hold-out을 사용해 데이터셋을 분리하고 모델의 일반화 성능을 검증하였다.
 모델 최적화를 위한 하이퍼파라미터 튜닝을 진행하였다.
모델 해석
 모델 결과로서 Test Score를 제시하여 모델을 평가하였다.
 PDP/SHAP/FeatureImportance 등을 활용하여 모델 작동을 설명하였다.
 모델 학습 결과에 대한 해석이 드러나 있다. (가설 해소 혹은 문제 해결 등)
