import pandas as pd

# 1. 기존 파일 읽기 (이름이 slidevqa_test.parquet라고 가정)
filename = "./data/slidevqa_test.parquet" 
df = pd.read_parquet(filename)

# 2. 데이터 20배로 복제하기 (1개 -> 20개)
# ignore_index=True를 해야 인덱스가 0,1,2... 로 깔끔하게 재정렬됩니다.
df_expanded = pd.concat([df] * 20, ignore_index=True)

# 3. 다시 저장
df_expanded.to_parquet(filename)

print(f"데이터 개수가 {len(df)}개에서 {len(df_expanded)}개로 늘어났습니다.")