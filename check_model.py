import pandas as pd

data = pd.read_csv('model_check_v1.csv')
pd.options.display.max_colwidth = 200
option_max = data['option'][data['c-td'] == data['c-td'].max()]
print(data['option'][data['c-td'] == data['c-td'].max()])
print(data['c-td'][data['c-td'] == data['c-td'].max()])
print(data['c-index'][data['c-td'] == data['c-td'].max()])
print(data['auc'][data['c-td'] == data['c-td'].max()])

# #"{'nodes': [10, 20, 40], 'batch': 4096, 'decay': 0.01, 'weighted_decay': 0, 'index': 1}"
# option_list = []
# start = 0
# print(len(option_max))
# for index in range(len(option_max)):
#     print(option_max[index])
#     if option_max[index] == "," and option_max[index+2]=="'":
#         option_list.append(option_max[start:index])
#         start = index
#
# print(option_list)
