import pandas as pd

acc_logs = []
loss_logs = []

template_dacc = ['Server', 'Defense', 'accuracy:']
template_global = ['Global', 'Model', 'Test', 'accuracy:']

fp1 = open(r"H:\By-FL\logs\2022-11-11\20.00.56\info.log")

for line in fp1.readlines():
    split_line = str.split(line)
    if set(template_dacc) < set(split_line):
        acc_logs.append(split_line[-1])
    if set(template_global) < set(split_line):
        loss_logs.append(split_line[-1])



csv_acc_logs = pd.DataFrame(data=acc_logs)
csv_acc_logs.to_csv(r'./logs/viewer/detect_acc1.csv', index=False)

csv_loss_logs = pd.DataFrame(data=loss_logs)
csv_loss_logs.to_csv(r'./logs/viewer/globel_acc1.csv', index=False)
