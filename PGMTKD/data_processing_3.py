from sklearn.model_selection import StratifiedShuffleSplit
import os
import re
import numpy as np
from scipy import signal
from tqdm import tqdm
def dataProcessing_3(file_path,  length=5120, use_sliding_window=True, step_size=5120):
    train_filenames = os.listdir(file_path)
    def capture(path, filenames,start_row=0, end_row=None):
        data = {}
        for i in tqdm(filenames):
            file_path = os.path.join(path, i)
            max_rows = None
            if end_row is not None:
                max_rows = end_row - start_row
            file = np.genfromtxt(open(file_path, "rb"), delimiter=",",dtype=float,skip_header=start_row, max_rows=max_rows)
            data[i] = file
        return data
    
    def NormalizeMult(data):
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            return data
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data
    
    def slice(data):
        data_keys = data.keys()
        Data_Samples = []
        Test_DataSamples = []
        Labels = []
        Test_Labels = []
        for key in tqdm(data_keys):
            slice_data = data[key]
            samp_num = len(slice_data) // 20480
            start = 0
            end_index = len(slice_data)
            resample_rate = 4
            resample_data = signal.resample(slice_data, len(slice_data) // resample_rate)
            reference_sample = resample_data[0:512]
            baseline_mean = np.mean(reference_sample)
            diff1 = np.zeros(int(614400 / (4 * 5120)))
            num_samples = len(reference_sample) // 512
            for i in range(1, num_samples):
                sample = resample_data[i * 512: (i + 1) * 512 - 1]
                sample_mean = np.mean(sample)
                resample_data[i * 512: (i + 1) * 512 - 1] = resample_data[i * 512: (i + 1) * 512 - 1] - (sample_mean - baseline_mean)
                diff1[i] = sample_mean - baseline_mean
            resample_data = NormalizeMult(resample_data)
            match = re.search(r'电流-([A-Z+]+)-S', key)
            if match:
                key_part = match.group(1)
                key_mapping = {
                    'A': 0,
                    'C': 1,
                    'G': 2,
                    'K': 3,
                    'N': 4,
                    'B': 5,
                    'B+G': 6,
                    'C+K+N': 7
                }
                class_num = key_mapping.get(key_part, -1)
            else:
                class_num = -1
            for i in range(samp_num):
                if use_sliding_window:
                    if i >= round(0.8 * samp_num):
                        test_sample = resample_data[start:start + length]
                        if len(test_sample) == length:
                            Test_DataSamples.append([test_sample]*4)
                            Test_Labels.append([class_num]*4)
                    else:
                        sample = resample_data[start:start + length]
                        if len(sample) == length:
                            Data_Samples.append([sample]*4)
                            Labels.append([class_num]*4)
                else:
                    if i > round(0.8 * samp_num):
                        test_sample = resample_data[start:start + length]
                        if len(test_sample) == length:
                            Test_DataSamples.append([test_sample]*4)
                            Test_Labels.append([class_num]*4)
                    else:
                        random_start = np.random.randint(low=0, high=(end_index - length))
                        sample = resample_data[random_start:random_start + length]
                        if len(sample) == length:
                            Data_Samples.append([sample]*4)
                            Labels.append([class_num]*4)
                start = start + step_size
        Data_Samples = np.array(Data_Samples)
        Labels = np.array(Labels)
        Test_Labels = np.array(Test_Labels)
        Test_DataSamples = np.array(Test_DataSamples)
        return Data_Samples, Labels, Test_DataSamples, Test_Labels
    train_data = capture(file_path, train_filenames, start_row=0, end_row=614400)
    Train_X, Train_Y, Test_X, Test_Y = slice(train_data)
    return Train_X, Test_X, Train_Y, Test_Y

def scalar_stand(Train_X, Test_X):
    mean = np.mean(Train_X, axis=(0, 1), keepdims=True)
    std = np.std(Train_X, axis=(0, 1), keepdims=True)
    Train_X = (Train_X - mean) / std
    Test_X = (Test_X - mean) / std
    return Train_X, Test_X

def valid_test_slice(Train_X, Train_Y, valid_size=0.2):
    ss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size)
    for train_index, test_index in ss.split(Train_X, Train_Y):
        X_train, X_test = Train_X[train_index], Train_X[test_index]
        Y_train, Y_test = Train_Y[train_index], Train_Y[test_index]
        return X_train, Y_train, X_test, Y_test
















