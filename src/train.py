import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import yaml
import os


def load_params():
    params = {}
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print("경고: params.yaml 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
        params = {
            "train": {
                "alpha": 0.1,
                "max_iter": 1000,
                "random_state": 42,
                "test_size": 0.2,
                "model_type": "ridge",
            }
        }
    return params.get("train", {})


def get_model(model_type, alpha, max_iter, random_state):
    """모델 타입에 따라 적절한 모델을 반환합니다."""
    if model_type.lower() == "linear":
        return LinearRegression()
    elif model_type.lower() == "ridge":
        return Ridge(alpha=alpha, max_iter=max_iter, random_state=random_state)
    elif model_type.lower() == "lasso":
        return Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
    else:
        print(f"알 수 없는 모델 타입: {model_type}. LinearRegression을 사용합니다.")
        return LinearRegression()


if __name__ == "__main__":
    params = load_params()
    alpha = params.get("alpha", 0.1)
    max_iter = params.get("max_iter", 1000)
    random_state = params.get("random_state", 42)
    test_size = params.get("test_size", 0.2)
    model_type = params.get("model_type", "ridge")

    print(f"📈 모델 학습 시작")
    print(f"   - 모델 타입: {model_type}")
    print(f"   - Alpha: {alpha}")
    print(f"   - Max Iterations: {max_iter}")
    print(f"   - Random State: {random_state}")
    print(f"   - Test Size: {test_size}")

    with mlflow.start_run(run_name=f"{model_type.capitalize()}_Training_alpha_{alpha}"):
        # 파라미터 로깅
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("model_type", model_type)

        # 데이터 로드
        df = pd.read_csv("data/raw_data.csv")
        print(f"📊 데이터 로드 완료: {len(df)} 행, {len(df.columns)} 열")

        # 특성과 타겟 분리
        feature_columns = [col for col in df.columns if col.startswith("feature")]
        X = df[feature_columns]
        y = df["target"]

        print(f"🔍 사용된 특성: {feature_columns}")
        print(f"📈 특성 개수: {len(feature_columns)}")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 특성 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"📊 훈련 데이터: {X_train.shape[0]} 샘플")
        print(f"📊 테스트 데이터: {X_test.shape[0]} 샘플")

        # 모델 생성 및 훈련
        model = get_model(model_type, alpha, max_iter, random_state)
        model.fit(X_train_scaled, y_train)

        # 훈련 성능 평가
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        print(f"✅ 훈련 R² 점수: {train_score:.4f}")
        print(f"✅ 테스트 R² 점수: {test_score:.4f}")

        # MLflow에 메트릭 로깅
        mlflow.log_metric("train_r2_score", train_score)
        mlflow.log_metric("test_r2_score", test_score)

        # 모델과 스케일러 저장
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"✅ 모델이 {model_path}에 저장되었습니다.")
        print(f"✅ 스케일러가 {scaler_path}에 저장되었습니다.")

        # MLflow에 아티팩트 로깅
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        print("✅ MLflow: 모델과 스케일러 아티팩트가 기록되었습니다.")

        # 특성 중요도 (계수) 로깅
        if hasattr(model, "coef_"):
            feature_importance = dict(zip(feature_columns, model.coef_))
            print("🔍 특성 중요도 (계수):")
            for feature, coef in feature_importance.items():
                print(f"   {feature}: {coef:.4f}")
                mlflow.log_param(f"coef_{feature}", coef)
