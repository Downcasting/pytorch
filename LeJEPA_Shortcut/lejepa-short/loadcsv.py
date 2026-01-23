import wandb
import pandas as pd
from tqdm import tqdm

# 1. 설정
csv_path_no = "LeJEPA normal.csv"
csv_paths = ['train_invariance.csv', 'train_lejepa.csv', 'train_probe.csv', 'train_sigreg.csv']
csv_names = ['train/inv', 'train/lejepa', 'train/probe', 'train/sigreg']

project_name = "LeJEPA-shortcut"
run_name = "Baseline_Sorted" # 이름 변경 권장

# CSV 헤더 설정
target_col_prefix = "dulcet-rain-4 - " 
step_col = "Step"

# 2. 모든 데이터를 리스트에 담기 (업로드 준비)
all_logs = []

# (1) Test Acc 데이터 담기
df_csv_no = pd.read_csv(csv_path_no)
print(f"Processing {csv_path_no}...")
for _, row in df_csv_no.iterrows():
    step = int(row[step_col])
    val = row[target_col_prefix + "test/acc"]
    
    # 딕셔너리 형태로 리스트에 저장 (나중에 정렬하기 위해)
    all_logs.append({
        "step": step,
        "data": {
            "test/acc": val,
            "test/acc_clean": val,
            "trainer/global_step": step # 기록용으로 남겨둠 (X축 설정용 아님)
        }
    })

# (2) 나머지 CSV 데이터 담기
for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i])
    print(f"Processing {csv_paths[i]}...")
    metric_name = csv_names[i] # 예: train/inv
    col_name = target_col_prefix + metric_name # 예: dulcet-rain-4 - train/inv
    
    for _, row in df.iterrows():
        step = int(row[step_col])
        val = row[col_name]
        
        all_logs.append({
            "step": step,
            "data": {
                metric_name: val,
                "trainer/global_step": step
            }
        })

# 3. [핵심] Step 기준으로 데이터 정렬하기
# 이렇게 하면 시간이 섞여 있어도 낮은 Step부터 차례대로 정렬됩니다.
print("Sorting data by step...")
all_logs.sort(key=lambda x: x['step'])

# 4. WandB 업로드
wandb.init(project=project_name, name=run_name, job_type="baseline")

# define_metric 제거 -> 기본 Step 사용 (기존 Run들과 호환성 확보)

print("Uploading to WandB...")
for log_entry in tqdm(all_logs):
    # step 인자를 명시하여 순서대로 기록
    wandb.log(log_entry['data'], step=log_entry['step'])

print("Upload Finished!")
wandb.finish()