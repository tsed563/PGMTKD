import os
import re
import numpy as np
from tqdm import tqdm
def dataProcessing_2(file_path,  length=5120, use_sliding_window=True, step_size=5120):
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

    def slice(data):
        data_keys = data.keys()
        Data_Samples = []
        Test_DataSamples = []
        Labels = []
        Test_Labels = []
        for key in tqdm(data_keys):
            slice_data = data[key]
            samp_num = len(slice_data) // step_size
            start = 0
            end_index = len(slice_data)
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
                        test_sample = slice_data[start:start + length]
                        if len(test_sample) == length:
                            Test_DataSamples.append(test_sample)
                            Test_Labels.append(class_num)
                    else:
                        sample = slice_data[start:start + length]
                        if len(sample) == length:
                            Data_Samples.append(sample)
                            Labels.append(class_num)
                else:
                    if i > round(0.8 * samp_num):
                        test_sample = slice_data[start:start + length]
                        if len(test_sample) == length:
                            Test_DataSamples.append(test_sample)
                            Test_Labels.append(class_num)
                    else:
                        random_start = np.random.randint(low=0, high=(end_index - length))
                        sample = slice_data[random_start:random_start + length]
                        if len(sample) == length:
                            Data_Samples.append(sample)
                            Labels.append(class_num)
                start = start + step_size
        Data_Samples = [sample[:, :4] for sample in Data_Samples]
        Test_DataSamples = [sample[:, :4] for sample in Test_DataSamples]
        Data_Samples = np.array(Data_Samples)
        Labels = np.array(Labels)
        Test_Labels = np.array(Test_Labels)
        Test_DataSamples = np.array(Test_DataSamples)
        return Data_Samples, Labels, Test_DataSamples, Test_Labels
    train_data = capture(file_path, train_filenames, start_row=0, end_row=614400)
    Train_X, Train_Y, Test_X, Test_Y = slice(train_data)
    return Train_X,Test_X, Train_Y, Test_Y

def collect_data(batch):
    data=[item[0] for item in batch]
    labels =[item[1] for item in batch]
    y=np.array(labels)
    data=np.array(data)
    S_P=data[:,:,0:1]
    S_V=data[:,:,1:4]
    S_P1=data[:,:,4:5]
    return [y,S_P,S_V,S_P1]








