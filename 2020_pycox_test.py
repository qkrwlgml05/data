import numpy as np
import pandas as pd
from pandas import DataFrame as df
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt  # Some useful functions

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_censored

import os
from ck_autoencoder_models import *

import torch
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def enc_using_trained_ae(filename, TARGET='male', ALPHA=0.01, N_ITER=100, L1R=-9999):
    dirname = './pan_V2/'
    filename_m_Xtr = 'data_pan_male_Xtrain_v1_short.csv'
    filename_m_Xts = 'data_pan_male_Xtest_v1_short.csv'
    filename_f_Xtr = 'data_pan_female_Xtrain_v1_short.csv'
    filename_f_Xts = 'data_pan_female_Xtest_v1_short.csv'

    filename_m_ytr = 'data_pan_male_Ytrain_v1_processed.csv'
    filename_m_yts = 'data_pan_male_Ytest_v1_processed.csv'
    filename_f_ytr = 'data_pan_female_Ytrain_v1_processed.csv'
    filename_f_yts = 'data_pan_female_Ytest_v1_processed.csv'


    s_time = time.time()

    if TARGET == 'male':
        X_tr = pd.read_csv(dirname + filename_m_Xtr)
        Y_tr = pd.read_csv(dirname + filename_m_ytr)

        X_ts = pd.read_csv(dirname + filename_m_Xts)
        Y_ts = pd.read_csv(dirname + filename_m_yts)
    else:
        X_tr = pd.read_csv(dirname + filename_f_Xtr)
        Y_tr = pd.read_csv(dirname + filename_f_ytr)

        X_ts = pd.read_csv(dirname + filename_f_Xts)
        Y_ts = pd.read_csv(dirname + filename_f_yts)

    # Y_tr = CoxformY(Y_tr)
    # Y_ts = CoxformY(Y_ts)
    X_tr = X_tr.drop('UR_SG3', axis=1)
    X_ts = X_ts.drop('UR_SG3', axis=1)

    X_tr_orig = X_tr
    X_ts_orig = X_ts

    normalizer = StandardScaler()
    X_tr = normalizer.fit_transform(X_tr)
    X_ts = normalizer.transform(X_ts)



    if TARGET == 'male':
        ae1 = autoencoder_m3().to(device)
    else:
        ae1 = autoencoder_f3().to(device)

    model_filename = filename
    ae1 = torch.load(model_filename)

    ae1.eval()
    ae1.cuda()

    # feature generation
    X_tr = torch.from_numpy(X_tr).float()
    with torch.no_grad():
        Z_tr = ae1.encoder(Variable(X_tr).to(device)).cpu()

    Z_tr_df = pd.DataFrame(Z_tr.cpu().numpy())

    X_ts = torch.from_numpy(X_ts).float()
    with torch.no_grad():
        Z_ts = ae1.encoder(Variable(X_ts).to(device)).cpu()

    Z_ts_df = pd.DataFrame(Z_ts.cpu().numpy())


    return Z_tr_df, Z_ts_df

class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, Data, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.Data, self.class_vector = Data[0], Data[1][1].float()
        self.n_splits = int(len(self.class_vector) / batch_size)
        print(self.class_vector)
        print(len(self.class_vector))
    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        # X = th.randn(self.class_vector.size(0), 2).numpy()
        X = self.Data.numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)
        print(y)
        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def pycox_deep(filename, Y_train, Y_test, opt, choice):
    # choice = {'lr_rate': l, 'batch': b, 'decay': 0, 'weighted_decay': wd, 'net': net, 'index': index}
    X_train, X_test = enc_using_trained_ae(filename, TARGET=opt, ALPHA=0.01, N_ITER=100, L1R=-9999)
    path = './models/analysis/'
    check = 0
    savename = 'model_check_autoen_m5_test_batch+dropout+wd.csv'
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if savename in file:
                check = 1

    # X_train = X_train.drop('UR_SG3', axis=1)
    # X_test = X_test.drop('UR_SG3', axis=1)

    x_train = X_train
    x_test = X_test
    x_train['SVDTEPC_G'] = Y_train['SVDTEPC_G']
    x_train['PC_YN'] = Y_train['PC_YN']
    x_test['SVDTEPC_G'] = Y_test['SVDTEPC_G']
    x_test['PC_YN'] = Y_test['PC_YN']

    ## DataFrameMapper ##
    cols_standardize = list(X_train.columns)
    cols_standardize.remove('SVDTEPC_G')
    cols_standardize.remove('PC_YN')

    standardize = [(col, None) for col in cols_standardize]

    x_mapper = DataFrameMapper(standardize)

    _ = x_mapper.fit_transform(X_train).astype('float32')
    X_train = x_mapper.transform(X_train).astype('float32')
    X_test = x_mapper.transform(X_test).astype('float32')

    get_target = lambda df: (df['SVDTEPC_G'].values, df['PC_YN'].values)
    y_train = get_target(x_train)
    durations_test, events_test = get_target(x_test)
    in_features = X_train.shape[1]
    print(in_features)
    num_nodes = choice['nodes']
    out_features = 1
    batch_norm = True  # False for batch_normalization
    dropout = 0.01
    output_bias = False
    # net = choice['net']
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)


    print("training")
    model = CoxPH(net, tt.optim.Adam)
    # lrfinder = model.lr_finder(X_train, y_train, batch_size)

    # lr_best = lrfinder.get_best_lr()
    lr_best = 0.0001
    model.optimizer.set_lr(choice['lr_rate'])

    weighted_decay = choice['weighted_decay']
    verbose = True

    batch_size = choice['batch']
    epochs = 100

    if weighted_decay == 0:
        callbacks = [tt.callbacks.EarlyStopping(patience=epochs)]
        # model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose=verbose)
    else:
        callbacks = [tt.callbacks.DecoupledWeightDecay(weight_decay=choice['decay'])]
        # model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose)

    ''''''
    # dataloader = model.make_dataloader(tt.tuplefy(X_train, y_train),batch_size,True)
    datas = tt.tuplefy(X_train, y_train).to_tensor()
    print(datas)
    make_dataset = tt.data.DatasetTuple;
    DataLoader = tt.data.DataLoaderBatch
    dataset = make_dataset(*datas)
    dataloader = DataLoader(dataset, batch_size, False, sampler=StratifiedSampler(datas, batch_size))
    # dataloader = DataLoader(dataset,batch_size, True)
    model.fit_dataloader(dataloader, epochs, callbacks, verbose)
    # model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose)
    # model.partial_log_likelihood(*val).mean()

    print("predicting")
    baseline_hazards = model.compute_baseline_hazards(datas[0], datas[1])
    baseline_hazards = df(baseline_hazards)

    surv = model.predict_surv_df(X_test)
    surv = 1 - surv
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

    print("scoring")
    c_index = ev.concordance_td()
    print("c-index(", opt, "): ", c_index)

    if int(c_index * 10) == 0:
        hazardname = 'pycox_model_hazard_m5_v2_' + opt + '_0'
        netname = 'pycox_model_net_m5_v2_' + opt + '_0'
        weightname = 'pycox_model_weight_m5_v2_' + opt + '_0'
    else:
        hazardname = 'pycox_model_hazard_m5_' + opt + '_'
        netname = 'pycox_model_net_m5_' + opt + '_'
        weightname = 'pycox_model_weight_m5_' + opt + '_'

    baseline_hazards.to_csv('./test/'+hazardname + str(int(c_index * 100)) + '_' + str(index) + '.csv', index=False)
    netname = netname + str(int(c_index * 100)) + '_' + str(index) + '.sav'
    weightname = weightname + str(int(c_index * 100)) + '_' + str(index) + '.sav'
    model.save_net('./test/' + netname)
    model.save_model_weights('./test/' + weightname)

    pred = df(surv)
    pred = pred.transpose()
    surv_final = []
    pred_final = []

    for i in range(len(pred)):
        pred_final.append(float(1-pred[Y_test['SVDTEPC_G'][i]][i]))
        surv_final.append(float(pred[Y_test['SVDTEPC_G'][i]][i]))

    Y_test_cox = CoxformY(Y_test)
    #print(surv_final)
    c_cox, concordant, discordant,_,_ = concordance_index_censored(Y_test_cox['PC_YN'], Y_test_cox['SVDTEPC_G'], surv_final)
    c_cox_pred = concordance_index_censored(Y_test_cox['PC_YN'], Y_test_cox['SVDTEPC_G'], pred_final)[0]
    print("c-index(", opt, ") - sksurv: ", round(c_cox, 4))
    print("cox-concordant(", opt, ") - sksurv: ", concordant)
    print("cox-disconcordant(", opt, ") - sksurv: ", discordant)
    print("c-index_pred(", opt, ") - sksurv: ", round(c_cox_pred, 4))

    fpr, tpr, _ = metrics.roc_curve(Y_test['PC_YN'], pred_final)
    auc = metrics.auc(fpr, tpr)
    print("auc(", opt, "): ", round(auc, 4))

    if check == 1:
        model_check = pd.read_csv(path+savename)
    else:
        model_check = df(columns=['option', 'gender', 'c-td', 'c-index', 'auc'])
    line_append = {'option':str(choice), 'gender':opt, 'c-td':round(c_index,4), 'c-index':round(c_cox_pred,4), 'auc':round(auc,4)}
    model_check = model_check.append(line_append, ignore_index=True)
    model_check.to_csv(path+savename, index=False)

    del X_train
    del X_test

    return surv_final

def CoxformY(Y_test):
    Y_test_cox = Y_test['PC_YN'].astype(np.int)
    Y_test_cox = df(Y_test_cox)
    Y_test_cox['PC_YN'] = Y_test_cox['PC_YN'].astype(np.bool_)
    Y_test_cox['SVDTEPC_G'] = Y_test['SVDTEPC_G'].astype(np.float)
    Y_test_cox = Y_test_cox.to_records(index=False)
    Y_test_cox = np.asarray(Y_test_cox)
    Y_test_cox.dtype = [('PC_YN', '?'), ('SVDTEPC_G', '<f8')]
    return Y_test_cox


node_list = [[10], [10, 20], [10, 20,40], [10, 20, 40, 20], [10, 20, 40, 40, 20], [10, 20, 40, 40, 20], [10, 20, 40, 40, 20, 10], [10, 20, 40, 40, 40, 20]]
batch_list = [4096, 2048, 1024]
wd_list = [0.01, 0.001, 0.0001, 0.00001]
lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
weight_decay = [0, 1]
index = 0
dir_X = './auen_model/'
dir_Y = '.'
task_option = 'female'
filename = 'm5_0_female_tr_lr0.0005_wd1e-06_epoch1500000.pt'
Y_tr_female = pd.read_csv(dir_Y + '/pan_V2/data_pan_' + task_option + '_Ytrain_v1_processed.csv')
Y_ts_female = pd.read_csv(dir_Y + '/pan_V2/data_pan_' + task_option + '_Ytest_v1_processed.csv')

for l in lr_list:
    for b in batch_list:
        for n in node_list:
            choice = {'nodes': n, 'lr_rate': l, 'batch': b, 'weighted_decay': 0, 'decay': 0, 'batch_norm': False, 'dropout': 0.01, 'index': index}
            pycox_deep(dir_X+filename, Y_tr_female, Y_ts_female, task_option, choice)
            index += 1

for l in lr_list:
    for b in batch_list:
        for n in node_list:
            choice = {'nodes': n, 'lr_rate': l, 'batch': b, 'weighted_decay': 0, 'decay': 0, 'batch_norm': True, 'dropout': 0, 'index': index}
            pycox_deep(dir_X+filename, Y_tr_female, Y_ts_female, task_option, choice)
            index += 1

for l in lr_list:
    for b in batch_list:
        for n in node_list:
            choice = {'nodes': n, 'lr_rate': l, 'batch': b, 'weighted_decay': 0.0001, 'decay': 1, 'batch_norm': False, 'dropout': 0, 'index': index}
            pycox_deep(dir_X+filename, Y_tr_female, Y_ts_female, task_option, choice)
            index += 1


# dir_X = '/home/hailers/pan_trained_ae/'
# dir_Y = '.'
# task_option = 'male'
# filename = 'm3_0_male_tr_lr0.0005_wd1e-05_epoch700000.pt'
# Y_tr_female = pd.read_csv(dir_Y + '/pan_V2/data_pan_' + task_option + '_Ytrain_v1_processed.csv')
# Y_ts_female = pd.read_csv(dir_Y + '/pan_V2/data_pan_' + task_option + '_Ytest_v1_processed.csv')
#
# for l in lr_list:
#     for b in batch_list:
#         for n in node_list:
#             choice = {'nodes': n, 'lr_rate': l, 'batch': b, 'weighted_decay': 0, 'decay': 0, 'index': index}
#             pycox_deep(dir_X+filename, Y_tr_female, Y_ts_female, task_option, choice)
#             index += 1
#
#
# for b in batch_list:
#     for n in node_list:
#         for d in wd_list:
#             choice = {'nodes': n, 'lr_rate': 0.0001, 'batch': b, 'weighted_decay': 1, 'decay': d, 'index': index}
#             pycox_deep(dir_X+filename, Y_tr_female, Y_ts_female, task_option, choice)
#             index += 1

