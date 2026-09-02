import pandas as pd

train = pd.read_csv("/workspace/datasets/TMHINTQI/sets/train.csv")
val   = pd.read_csv("/workspace/datasets/TMHINTQI/sets/val.csv")
test  = pd.read_csv("/workspace/datasets/TMHINTQI/sets/test.csv")

train_sys = set(train["system_id"])
val_sys   = set(val["system_id"])
test_sys  = set(test["system_id"])

print("Sistemas no train:", len(train_sys))
print("Sistemas no val:",   len(val_sys))
print("Sistemas no test:",  len(test_sys))
print()
print("Val sistemas que aparecem no train:",  len(val_sys  & train_sys), "/", len(val_sys))
print("Test sistemas que aparecem no train:", len(test_sys & train_sys), "/", len(test_sys))
print()
print("Sistemas EXCLUSIVOS do val:",  val_sys  - train_sys)
print("Sistemas EXCLUSIVOS do test:", test_sys - train_sys)