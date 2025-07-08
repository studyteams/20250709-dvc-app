import subprocess
import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
import yaml

app = Flask(__name__)
app.secret_key = "your_super_secret_key_for_flash_messages"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_command(command, cwd=PROJECT_ROOT):
    print(f"Executing command: {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True, cwd=cwd
        )
        if result.stderr:
            print(f"Command STDERR: {result.stderr}")
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing command '{e.cmd}':\nStdout: {e.stdout}\nStderr: {e.stderr}"
        print(error_msg)
        raise RuntimeError(error_msg)


def get_git_log():
    try:
        stdout, _ = run_command("git log --oneline --decorate=short")
        logs = []
        for line in stdout.splitlines():
            if line:
                parts = line.split(" ", 1)
                logs.append(
                    {"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""}
                )
        return logs
    except RuntimeError as e:
        print(f"Failed to get Git log: {e}")
        return []


def get_current_status():
    status = {
        "git_head": "N/A",
        "git_message": "N/A",
        "dvc_data_status": {},
        "params_content": "File not found or unreadable.",
    }
    try:
        git_head, _ = run_command("git rev-parse HEAD")
        status["git_head"] = git_head

        git_msg, _ = run_command(f"git log -1 --pretty=%B {git_head}")
        status["git_message"] = git_msg.strip()

        dvc_file_path = os.path.join("data", "raw_data.csv.dvc")
        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, "r") as f:
                dvc_meta = yaml.safe_load(f)
                if "outs" in dvc_meta and dvc_meta["outs"]:
                    dvc_data_hash = dvc_meta["outs"][0].get(
                        "md5", dvc_meta["outs"][0].get("hash", "N/A")
                    )
                    status["dvc_data_status"] = {"raw_data.csv": dvc_data_hash}
        else:
            status["dvc_data_status"] = {
                "raw_data.csv": "Not DVC tracked or file missing."
            }

        try:
            with open("params.yaml", "r") as f:
                status["params_content"] = f.read()
        except FileNotFoundError:
            status["params_content"] = "params.yaml not found."

    except Exception as e:
        print(f"Error getting current status: {e}")
        flash(f"상태 정보를 가져오는 중 오류 발생: {e}", "error")
    return status


@app.route("/")
def index():
    status = get_current_status()
    git_logs = get_git_log()
    return render_template("index.html", status=status, git_logs=git_logs)


@app.route("/checkout", methods=["POST"])
def checkout():
    commit_hash = request.form["commit_hash"]
    print(f"\n--- {commit_hash} 버전으로 전환 시도 ---")

    try:
        flash(f"Git 커밋 {commit_hash}으로 체크아웃 중...", "info")
        stdout, stderr = run_command(f"git checkout {commit_hash}")
        print(f"Git Checkout Output: {stdout}")
        if stderr:
            flash(f"Git Checkout 경고: {stderr}", "warning")

        flash("DVC 데이터를 복원 중...", "info")
        stdout, stderr = run_command("dvc checkout")
        print(f"DVC Checkout Output: {stdout}")
        if stderr:
            flash(f"DVC Checkout 경고: {stderr}", "warning")

        flash("DVC 파이프라인 재실행 중...", "info")
        stdout, stderr = run_command("dvc repro")
        print(f"DVC Repro Output: {stdout}")
        if stderr:
            flash(f"DVC Repro 경고: {stderr}", "warning")

        flash(
            f"성공적으로 {commit_hash} 버전으로 전환되었고 파이프라인이 재실행되었습니다.",
            "success",
        )
        print(f"--- {commit_hash} 버전 전환 완료 ---")

    except RuntimeError as e:
        flash(f"버전 전환 중 오류 발생: {e}", "error")
        print(f"--- {commit_hash} 버전 전환 실패 ---")

    return redirect(url_for("index"))


@app.route("/start_mlflow_ui")
def start_mlflow_ui():
    flash(
        "MLflow UI를 시작하려면 터미널에서 'mlflow ui'를 실행하세요. 잠시 후 http://127.0.0.1:5050으로 접속 가능합니다.",
        "info",
    )
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("🌍 Flask 애플리케이션이 http://localhost:5001 에서 실행됩니다.")
    print(
        "📊 MLflow UI는 별도 터미널에서 'mlflow ui' 명령어로 실행해야 합니다 (http://127.0.0.1:5050)."
    )
    app.run(debug=True, port=5001)
