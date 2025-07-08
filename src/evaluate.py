import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
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
            "train": {"random_state": 42, "test_size": 0.2},
            "evaluate": {"metrics": ["mse", "r2_score", "mae", "rmse"]},
        }
    return params


if __name__ == "__main__":
    params = load_params()
    train_params = params.get("train", {})
    eval_params = params.get("evaluate", {})

    random_state = train_params.get("random_state", 42)
    test_size = train_params.get("test_size", 0.2)
    metrics = eval_params.get("metrics", ["mse", "r2_score", "mae", "rmse"])

    print("📊 모델 평가 시작")
    print(f"   - 평가 메트릭: {metrics}")

    with mlflow.start_run(nested=True, run_name="Model_Evaluation"):
        # 데이터 로드
        df = pd.read_csv("data/raw_data.csv")
        feature_columns = [col for col in df.columns if col.startswith("feature")]
        X = df[feature_columns]
        y_true = df["target"]

        # 데이터 분할 (훈련과 동일한 방식)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=test_size, random_state=random_state
        )

        # 스케일러 로드
        scaler_path = os.path.join("model", "scaler.pkl")
        if not os.path.exists(scaler_path):
            print(f"오류: 스케일러 파일 {scaler_path}을(를) 찾을 수 없습니다.")
            exit(1)

        scaler = joblib.load(scaler_path)
        X_test_scaled = scaler.transform(X_test)

        # 모델 로드 (여러 모델 타입 시도)
        model_types = ["ridge", "lasso", "linear"]
        model = None
        model_type = None

        for mt in model_types:
            model_path = os.path.join("model", f"{mt}_model.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                model_type = mt
                print(f"✅ {model_type} 모델을 로드했습니다.")
                break

        if model is None:
            print("오류: 사용 가능한 모델 파일을 찾을 수 없습니다.")
            exit(1)

        # 예측 수행
        y_pred = model.predict(X_test_scaled)

        # 메트릭 계산
        results = {}

        if "mse" in metrics:
            mse = mean_squared_error(y_test, y_pred)
            results["mse"] = mse
            mlflow.log_metric("mse", mse)
            print(f"✅ MSE: {mse:.4f}")

        if "r2_score" in metrics:
            r2 = r2_score(y_test, y_pred)
            results["r2_score"] = r2
            mlflow.log_metric("r2_score", r2)
            print(f"✅ R² Score: {r2:.4f}")

        if "mae" in metrics:
            mae = mean_absolute_error(y_test, y_pred)
            results["mae"] = mae
            mlflow.log_metric("mae", mae)
            print(f"✅ MAE: {mae:.4f}")

        if "rmse" in metrics:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results["rmse"] = rmse
            mlflow.log_metric("rmse", rmse)
            print(f"✅ RMSE: {rmse:.4f}")

        # 추가 통계 정보
        print(f"📊 테스트 데이터 크기: {len(y_test)}")
        print(f"📊 예측값 범위: {y_pred.min():.2f} ~ {y_pred.max():.2f}")
        print(f"📊 실제값 범위: {y_test.min():.2f} ~ {y_test.max():.2f}")

        # 특성 중요도 출력
        if hasattr(model, "coef_"):
            feature_importance = dict(zip(feature_columns, model.coef_))
            print("🔍 특성 중요도 (계수):")
            for feature, coef in sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                print(f"   {feature}: {coef:.4f}")

        # MLflow에 파라미터 로깅
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        print("✅ MLflow: 모든 메트릭과 파라미터가 기록되었습니다.")

        # 결과 요약
        print("\n📋 평가 결과 요약:")
        for metric, value in results.items():
            print(f"   {metric.upper()}: {value:.4f}")
