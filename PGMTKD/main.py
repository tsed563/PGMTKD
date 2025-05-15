from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
import numpy
import time
import argparse
import random
from sklearn import metrics
import torch
from transformers import GPT2Model, GPT2Config
import os
from data_processing_2 import dataProcessing_2,collect_data
from data_processing_3 import dataProcessing_3,scalar_stand,valid_test_slice
import torchvision.models as models
from torchvision.models import ResNet50_Weights
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
seed=123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--EPOCH', type=int, default=100, help='Get the EPOCH')
parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=12, help='Get the TRAIN_BATCH_SIZE')
parser.add_argument('--TEST_BATCH_SIZE', type=int, default=12, help='Get the TEST_BATCH_SIZE')
parser.add_argument('--gpu_num', type=int, default=0, help='Get the GPU_NUM')
parser.add_argument('--kd_T', default=8.0, type=float, help='T for Temperature scaling')
parser.add_argument('--kd_mode', default='cse', choices=['cse', 'mse'], type=str, help='')
parser.add_argument('--losstype', default='cross', choices=['cross', 'KL'], type=str, help='Get the losstype')
parser.add_argument('--LR', type=float, default=1e-3, help='Get the LR')
parser.add_argument('--LRtype', default='fix', choices=['fix', 'ReduceLROnPlateau'], type=str, help='Get the LRtype')
parser.add_argument('--regular_coeff', type=float, default=1e-4, help='Get the regularization coefficient')
parser.add_argument('--distill_decay', action='store_true', default=False, help='distillation decay')
parser.add_argument("--hidden_dim", type=int, default=32, help='teacher_model_hidden_dim')
parser.add_argument("--gpt_n_embd", type=int, default=768, help='gpt_paramater')
parser.add_argument("--gpt_n_layer", type=int, default=12, help='gpt_paramater')
parser.add_argument("--gpt_n_head", type=int, default=12, help='gpt_paramater')
parser.add_argument("--gpt_resid_pdrop", type=float, default=0.1, help='gpt_paramater')
parser.add_argument("--gpt_attn_pdrop", type=float, default=0.1, help='gpt_paramater')
parser.add_argument("--gpt_embd_pdrop", type=float, default=0.1, help='gpt_paramater')
parser.add_argument('--gpt_activation_function', default='gelu', choices=['relu', 'silu', 'gelu', 'tanh', 'gelu_new'], type=str, help='gpt_paramater')
parser.add_argument("----num_chunks", type=int, default=15, help='--num_chunks')
parser.add_argument("----chunk_len", type=int, default=640, help='--chunk_len')
parser.add_argument("----chunk_stride", type=int, default=320, help='--chunk_stride')
parser.add_argument('--Isinit', action='store_true', default=False, help='Isinit')
parser.add_argument('--ACCU_TRAIN_BATCH_SIZE', type=int, default=48, help='Get the ACCU_TRAIN_BATCH_SIZE')
args = parser.parse_args()

class CrossTransformer(nn.Module):
    def __init__(self, d1, d2, seq_length, feature_dim, outputdim):
        super(CrossTransformer, self).__init__()
        self.fc_map_a = nn.Linear(d1, feature_dim)
        self.fc_map_b = nn.Linear(d2, feature_dim)
        self.layer_norm_a = nn.LayerNorm(normalized_shape=[seq_length, d1])
        self.layer_norm_b = nn.LayerNorm(normalized_shape=[seq_length, d2])
        self.dropout = nn.Dropout(0.1)
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.pos_encoder = nn.Embedding(seq_length, feature_dim)
        self.self_attn1 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.self_attn2 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)
        self.fusion_linear = nn.Linear(feature_dim * 2, outputdim)

    def forward(self, seq1, seq2):
        seq1 = self.layer_norm_a(seq1)
        seq1 = self.dropout(F.relu(self.fc_map_a(seq1)))
        seq2 = self.layer_norm_b(seq2)
        seq2 = self.dropout(F.relu(self.fc_map_b(seq2)))
        pos = torch.arange(self.seq_length, device=seq1.device).unsqueeze(0).repeat(seq1.size(0), 1)
        seq1 = seq1 + self.pos_encoder(pos)
        seq2 = seq2 + self.pos_encoder(pos)
        seq1 = self.self_attn1(seq1.permute(1, 0, 2)).permute(1, 0, 2)
        seq2 = self.self_attn2(seq2.permute(1, 0, 2)).permute(1, 0, 2)
        seq1_att, _ = self.cross_attn1(seq1, seq2, seq2)
        seq2_att, _ = self.cross_attn2(seq2, seq1, seq1)
        combined_features = torch.cat([seq1_att, seq2_att], dim=-1)
        combined_features = torch.mean(combined_features, dim=1)
        fused_features = self.fusion_linear(combined_features)
        return fused_features
class finalclassifier(nn.Module):
    def __init__(self, gpt_n_embd, num_chunks, num_classes):
        super(finalclassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(gpt_n_embd*num_chunks, gpt_n_embd*num_chunks//2),
            nn.ReLU(),
            nn.Linear(gpt_n_embd*num_chunks//2, gpt_n_embd*num_chunks//4),
            nn.ReLU(),
            nn.Linear(gpt_n_embd*num_chunks//4, gpt_n_embd),
            nn.ReLU(),
        )
        self.classifier2 = nn.Linear(gpt_n_embd, num_classes)
    def forward(self, fused_features):
        fused_features = fused_features.contiguous().view(fused_features.size(0), -1)
        middle_feature = self.classifier(fused_features)
        logits = self.classifier2(middle_feature)
        return middle_feature, logits

class CustomResNet50(nn.Module):
    def __init__(self, input_channels=1,num_embed=768):
        super(CustomResNet50, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.model_layer=nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.model_fc1 = nn.Linear(in_features=resnet.fc.in_features, out_features=num_embed)
    def forward(self, x):
        y = self.model_feat(x)
        y = self.model_layer(y)
        logits = self.model_fc1(y)
        return logits
class PhyModel(nn.Module):
    def __init__(self, input_channels=1,num_embed=768):
        super(PhyModel, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.model_layer=nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.model_fc1 = nn.Linear(in_features=30, out_features=num_embed)
        self.model_fc3 = nn.Linear(in_features=resnet.fc.in_features, out_features=30)

    def forward(self, x):
        inputs1 = torch.fft.fft(torch.complex(x, torch.zeros_like(x))).abs()
        positions = [10, 20, 29, 39, 49, 59, 68, 79, 89, 98, 107, 120, 147, 294, 437, 581, 732, 880, 1027, 1180, 1324,137, 284, 427, 571, 722, 870, 1017, 1170, 1314]#S1
        extracted_values = inputs1[:, :, positions, :]
        extracted_values = extracted_values.view(x.size(0), 30)
        y = self.model_feat(x)
        y = self.model_layer(y)
        y_30 = self.model_fc3(y)
        fused_30=extracted_values+y_30
        logits=self.model_fc1(fused_30)
        return logits,fused_30

class CNN_1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 9, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(9, 18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(18, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, is_ca=False):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if is_ca:
            loss = (nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T**2)).sum(-1)
        else:
            loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss

def init_weights(m, feature_dim):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=feature_dim ** -0.5)
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.constant_(m.in_proj_bias, 0)
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.out_proj.bias, 0)

def find_max_values_and_indices(data_list):
    if not data_list:
        return None

    max_value = max(data_list)
    indices = [index for index, value in enumerate(data_list) if value == max_value]

    return max_value, indices

def find_max_value_by_indices(array, indices):
    if not array or not indices:
        return None
    max_value = array[indices[0]]
    max_index = indices[0]
    for index in indices[1:]:
        if array[index] > max_value:
            max_value = array[index]
            max_index = index

    return max_index, max_value

def pressure_pulsation_model( params, data):
        params=params.to(device)
        a = params[:, :12]
        b = params[:, 12:21]
        c = params[:, 21:30]
        t = torch.arange(0, 1, 1 / 5120, dtype=torch.float).to(device)
        t = t.view(1, -1)
        pressure_signal = torch.zeros(data.size(0), 5120).to(device)
        dict1 = {0: 10, 1: 20, 2: 29, 3: 39, 4: 49, 5: 59, 6: 68, 7: 79, 8: 89, 9: 98, 10: 107, 11: 120}
        dict2 = {0: 147, 1: 294, 2: 437, 3: 581, 4: 732, 5: 880, 6: 1027, 7: 1180, 8: 1324}#S1
        x_complex = data.type(torch.complex64)
        yf = torch.fft.fft(x_complex).to(device)
        for m in range(len(dict1)):
            angle = torch.atan2(-yf[:, dict1[m]].imag, yf[:, dict1[m]].real).to(device)
            angle = angle.view(-1, 1)
            pressure_signal += 2 / 5120 * a[:, m:m + 1] * torch.cos(2 * dict1[m] * np.pi * t - angle)
            pressure_signal=pressure_signal.to(device)
        for n in range(len(dict2)):
            angle = torch.atan2(-yf[:, dict2[n]].imag, yf[:, dict2[n]].real).to(device)
            angle = angle.view(-1, 1)
            a1 = 2 / 5120 * (b[:, n:n + 1] + c[:, n:n + 1] * torch.cos(2 * np.pi * 10 * t))
            a2 = torch.cos(2 * np.pi * dict2[n] * t - angle)
            pressure_signal += a1 * a2
            pressure_signal=pressure_signal.to(device)
        pressure_signal += torch.mean(data)
        pressure_signal=pressure_signal.to(device)
        return pressure_signal

def custom_loss(S,fused):
        ddd=torch.from_numpy(S).float().to(device)
        ddd = ddd.view(ddd.size(0), 5120)
        reconstructed_signal = pressure_pulsation_model(fused, ddd)
        reconstructed_signal = reconstructed_signal.unsqueeze(1)
        loss=torch.mean(torch.square(ddd - reconstructed_signal))
        loss_phy = loss.sum()
        return loss_phy

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_acc = None
        self.best_model = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_model = model.state_dict()
        elif val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)

num_epochs = args.EPOCH
TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = args.TEST_BATCH_SIZE
gpu_num = args.gpu_num
hidden_dim = args.hidden_dim
gpt_n_embd = args.gpt_n_embd
gpt_n_layer = args.gpt_n_layer
gpt_n_head = args.gpt_n_head
gpt_resid_pdrop = args.gpt_resid_pdrop
gpt_attn_pdrop = args.gpt_attn_pdrop
gpt_embd_pdrop = args.gpt_embd_pdrop
gpt_activation_function = args.gpt_activation_function
num_chunks = args.num_chunks
chunk_len = args.chunk_len
chunk_stride = args.chunk_stride
Isinit = args.Isinit
ACCU_TRAIN_BATCH_SIZE = args.ACCU_TRAIN_BATCH_SIZE

frequency = 1280
chunk_len = frequency // 2
chunk_stride = frequency // 4
classnum = 8
num_fold = 1

string_gpu = "cuda:" + str(gpu_num)
device = torch.device(string_gpu if torch.cuda.is_available() else "cpu")

all_acc_student_with_teacher1 = []
all_f1_student_with_teacher1 = []
all_prelist_student_with_teacher1 = []
all_truelist_student_with_teacher1 = []
all_maxindex1 = []
all_acc_student_with_teacher2 = []
all_f1_student_with_teacher2 = []
all_prelist_student_with_teacher2 = []
all_truelist_student_with_teacher2 = []
all_maxindex2 = []
all_acc_student_with_teacher3 = []
all_f1_student_with_teacher3 = []
all_prelist_student_with_teacher3 = []
all_truelist_student_with_teacher3 = []
all_maxindex3 = []
all_acc_student_with_teacher4 = []
all_f1_student_with_teacher4 = []
all_prelist_student_with_teacher4 = []
all_truelist_student_with_teacher4 = []
all_maxindex4 = []
for Subject_num in range(1):
    print("Subject " + str(Subject_num) + "-------------------------")
    student_average_acc = 0
    student_acc_list = []
    teacher_average_acc = 0
    teacher_acc_list = []

    student_with_teacher_average_acc_val = []
    student_with_teacher_average_acc = []
    student_with_teacher_average_f1 = []
    student_with_teacher_prelist = []
    student_with_teacher_truelist = []
    for fold in range(num_fold):
        print("\n\n\n\nThis is Fold!!!!!!!!!!------------------------------------", str(fold + 1))
        train_x, test_x, train_y, test_y = dataProcessing_3(file_path="D:/process date/data_1_S1")
        train_x = train_x.reshape(768, 5120, 1)
        test_x = test_x.reshape(192, 5120, 1)
        train_x1, test_x1, train_y1, test_y1 = dataProcessing_2(file_path="D:/process date/data_2_S1")
        Train_X1, Test_X1 = scalar_stand(train_x1, test_x1)
        Train_X2 = np.concatenate((Train_X1, train_x), axis=2)
        Test_X2 = np.concatenate((Test_X1, test_x), axis=2)
        Train_X, Train_Y, Val_X, Val_Y = valid_test_slice(Train_X2, train_y1, 0.25)
        truelist_sub =test_y1
        train_data = TensorDataset(torch.tensor(Train_X, dtype=torch.float32), torch.tensor(Train_Y, dtype=torch.long))
        train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE,collate_fn=collect_data,shuffle=True)
        val_data = TensorDataset(torch.tensor(Val_X, dtype=torch.float32), torch.tensor(Val_Y, dtype=torch.long))
        val_loader = DataLoader(dataset=val_data, batch_size=TEST_BATCH_SIZE,collate_fn=collect_data,shuffle=False)
        test_data = TensorDataset(torch.tensor(Test_X2, dtype=torch.float32), torch.tensor(test_y1, dtype=torch.long))
        test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE,collate_fn=collect_data, shuffle=False)
        len_train_data = len(train_data)
        len_val_data = len(val_data)
        len_test_data = len(test_data)

        print('\n\nTraining...')
        num_inputs1 = 1
        num_inputs2 = 3
        student=CNN_1D(1,8).to(device)
        gpt_config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=gpt_n_embd,
            n_layer=gpt_n_layer,
            n_head=gpt_n_head,
            resid_pdrop=gpt_resid_pdrop,
            attn_pdrop=gpt_attn_pdrop,
            embd_pdrop=gpt_embd_pdrop,
            activation_function=gpt_activation_function
        )
        gpt_model = GPT2Model.from_pretrained("gpt2", config=gpt_config,cache_dir="D:/gpt2").to(device)
        gpt_model.eval()
        learnable_token = torch.randn((1, gpt_n_embd), requires_grad=True).to(device)
        teacher1 = CrossTransformer(num_inputs1, num_inputs2, chunk_len, hidden_dim, gpt_n_embd).to(device)
        finalclass_teacher1 = finalclassifier(gpt_n_embd, num_chunks, classnum).to(device)

        teacher2 = CustomResNet50(input_channels=1, num_embed=gpt_n_embd).to(device)
        finalclass_teacher2 = finalclassifier(gpt_n_embd, 1, classnum).to(device)

        teacher3 = PhyModel(input_channels=1, num_embed=gpt_n_embd).to(device)
        finalclass_teacher3 = finalclassifier(gpt_n_embd, 1, classnum).to(device)

        teacher4 = CustomResNet50(input_channels=1, num_embed=gpt_n_embd).to(device)
        finalclass_teacher4 = finalclassifier(gpt_n_embd, 1, classnum).to(device)
        if Isinit:
            print("initializ！！！！")
            teacher1.apply(lambda m: init_weights(m, hidden_dim))
            finalclass_teacher1.apply(lambda m: init_weights(m, hidden_dim))
            finalclass_teacher2.apply(lambda m: init_weights(m, hidden_dim))
            finalclass_teacher3.apply(lambda m: init_weights(m, hidden_dim))
            finalclass_teacher4.apply(lambda m: init_weights(m, hidden_dim))
            student.apply(lambda m: init_weights(m, hidden_dim))

        optimizer = torch.optim.AdamW([{'params': teacher1.parameters()},{'params': finalclass_teacher1.parameters()},{'params': teacher2.parameters()},{'params': finalclass_teacher2.parameters()},
                                       {'params': teacher3.parameters()},{'params': finalclass_teacher3.parameters()},{'params': teacher4.parameters()},{'params': finalclass_teacher4.parameters()},
                                       {'params': student.parameters()}], lr=args.LR, weight_decay=args.regular_coeff)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(args.kd_T)
        criterion_construct = torch.nn.MSELoss(reduction='mean')
        early_stopping = EarlyStopping(patience=10, delta=0.001)
        Best_trainacc_1 = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            train_loss = .0
            train_acc = .0
            teacher1.train()
            finalclass_teacher1.train()
            teacher2.train()
            finalclass_teacher2.train()
            teacher3.train()
            finalclass_teacher3.train()
            teacher4.train()
            finalclass_teacher4.train()
            student.train()
            trainstepsum = 0
            accumulation_steps = ACCU_TRAIN_BATCH_SIZE // TRAIN_BATCH_SIZE
            for trainstep, (y,S_P,S_V,S_P1) in enumerate(train_loader):

                batch_x1_teacher1 = torch.from_numpy(S_P).float().to(device)
                batch_x2_teacher1 = torch.from_numpy(S_V).float().to(device)
                batch_x1_teacher1 = batch_x1_teacher1.unfold(1, chunk_len,
                                                             chunk_stride)
                batch_x1_teacher1 = batch_x1_teacher1.permute(0, 1, 3, 2)
                batch_x1_teacher1 = batch_x1_teacher1.contiguous().view(-1, batch_x1_teacher1.size(2),
                                                                        batch_x1_teacher1.size(
                                                                            3))
                batch_x2_teacher1 = batch_x2_teacher1.unfold(1, chunk_len,
                                                             chunk_stride)
                batch_x2_teacher1 = batch_x2_teacher1.permute(0, 1, 3, 2)
                batch_x2_teacher1 = batch_x2_teacher1.contiguous().view(-1, batch_x2_teacher1.size(2),
                                                                        batch_x2_teacher1.size(3))
                y=torch.from_numpy(y).long().to(device)
                batch_x1_teacher1, batch_x2_teacher1, batch_y= batch_x1_teacher1.to(device), batch_x2_teacher1.to(device), y
                fusedfeature = teacher1(batch_x1_teacher1, batch_x2_teacher1)
                fusedfeature = fusedfeature.contiguous().view(-1, num_chunks, fusedfeature.size(-1))
                losses = []
                for i in range(2, num_chunks + 1):
                    masked_data = fusedfeature.clone()
                    masked_data[:, i:, :] = 0
                    masked_data[:, i - 1, :] = learnable_token
                    outputs = gpt_model(inputs_embeds=masked_data)
                    last_hidden_states = outputs.last_hidden_state
                    predicted_hidden_state = last_hidden_states[:, i - 1, :]
                    actual_hidden_state = fusedfeature[:, i - 1, :]
                    loss = criterion_construct(predicted_hidden_state,
                                     actual_hidden_state.detach())
                    losses.append(loss)
                construct_loss_teacher1 = sum(losses) / (num_chunks - 1)
                feat_t1, logit_t1 = finalclass_teacher1(fusedfeature)
                class_loss_teacher1 = criterion_cls(logit_t1, batch_y)
                loss_teacher1 = class_loss_teacher1 + construct_loss_teacher1

                batch_x1_teacher2 = S_V[:, np.newaxis, :]
                batch_x1_teacher2 = torch.from_numpy(batch_x1_teacher2).float().to(device)
                feat_t2 = teacher2(batch_x1_teacher2)
                _, logit_t2 = finalclass_teacher2(feat_t2)
                loss_teacher2 = criterion_cls(logit_t2, batch_y)

                batch_x1_teacher3 = S_P1[:, np.newaxis, :]
                batch_x1_teacher3 = torch.from_numpy(batch_x1_teacher3).float().to(device)
                feat_t3,fused_30= teacher3(batch_x1_teacher3)
                _, logit_t3 = finalclass_teacher3(feat_t3)
                loss_teacher3 = criterion_cls(logit_t3, batch_y)

                batch_x1_teacher4 = S_P[:, np.newaxis, :]
                batch_x1_teacher4 = torch.from_numpy(batch_x1_teacher4).float().to(device)
                feat_t4 = teacher4(batch_x1_teacher4)
                _, logit_t4 = finalclass_teacher4(feat_t4)
                loss_teacher4 = criterion_cls(logit_t4, batch_y)

                S_P = S_P.reshape(-1, 1, 5120)
                batch_x1_student = torch.from_numpy(S_P).float().to(device)
                logit_s = student(batch_x1_student)
                loss_student = criterion_cls(logit_s, batch_y)

                criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
                loss_t_list = [criterion_cls_lc(logit_t1, batch_y), criterion_cls_lc(logit_t2, batch_y),criterion_cls_lc(logit_t3, batch_y),criterion_cls_lc(logit_t4, batch_y)]
                loss_t = torch.stack(loss_t_list, dim=0)
                attention = (1.0 - F.softmax(loss_t, dim=0))
                loss_dist1_list1 = [criterion_div(logit_s, logit_t1, is_ca=True),
                                 criterion_div(logit_s, logit_t2, is_ca=True),criterion_div(logit_s, logit_t3, is_ca=True),criterion_div(logit_s, logit_t4, is_ca=True)]
                loss_dist1 = torch.stack(loss_dist1_list1, dim=0)
                bsz1 = loss_dist1.shape[1]
                loss_dist1 = (torch.mul(attention, loss_dist1).sum()) / (1.0 * bsz1 * 2)
                loss_phy=custom_loss(S_P1,fused_30)
                loss =loss_teacher1 + loss_teacher2 + loss_teacher3 + loss_teacher4 + loss_student + loss_phy + loss_dist1
                loss = loss / accumulation_steps
                loss.backward()
                if (trainstep + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.data.item()
                trainpred_y = (torch.max(logit_s, 1)[1]).cpu().numpy()
                accuracy0 = (trainpred_y == batch_y.squeeze().cpu().numpy()).astype(int).sum()
                train_acc = train_acc + accuracy0
                trainstepsum = trainstepsum + 1
            print('Train loss: {:.6f}, Train acc: {:.6f}'.format(train_loss / trainstepsum, train_acc / len_train_data))
            if args.LRtype == 'ReduceLROnPlateau':
                scheduler.step(train_loss / trainstepsum)
            if (train_acc / len_train_data) > Best_trainacc_1:
                Best_trainacc_1 = train_acc / len_train_data
            if (epoch + 1) % 1 == 0:
                teacher1.eval()
                finalclass_teacher1.eval()
                teacher2.eval()
                finalclass_teacher2.eval()
                teacher3.eval()
                finalclass_teacher3.eval()
                teacher4.eval()
                finalclass_teacher4.eval()
                student.eval()
                with torch.no_grad():
                    valstepsum = 0
                    eval_loss = .0
                    eval_acc = .0
                    for trainstep, (y,S_P,S_V,S_P1) in enumerate(val_loader):
                        S_P = S_P.reshape(-1, 1, 5120)
                        batch_x2_student = torch.from_numpy(S_P).float().to(device)
                        logit_s1 = student(batch_x2_student)
                        y = torch.from_numpy(y).long().to(device)
                        batch_y1 = y
                        loss_student = criterion_cls(logit_s1, batch_y1)
                        loss = loss_student
                        eval_loss += loss.data.item()
                        valpred_y = (torch.max(logit_s1, 1)[1]).cpu().numpy()
                        accuracy1 = (valpred_y == batch_y1.squeeze().cpu().numpy()).astype(int).sum()
                        eval_acc = eval_acc + accuracy1
                        valstepsum = valstepsum + 1
                    print('Eval loss: {:.6f}, Eval acc: {:.6f}'.format(eval_loss / valstepsum,eval_acc / len_val_data))
                    student_with_teacher_average_acc_val.append(eval_acc / len_val_data)

                    teststepsum = 0
                    test_loss = .0
                    test_acc = .0
                    prelist_temp = []
                    for trainstep, (y,S_P,S_V,S_P1) in enumerate(test_loader):
                        S_P = S_P.reshape(-1, 1, 5120)
                        batch_x3_student = torch.from_numpy(S_P).float().to(device)
                        logit_s2 = student(batch_x3_student)
                        y = torch.from_numpy(y).long().to(device)
                        batch_y2 = y
                        loss_student = criterion_cls(logit_s2, batch_y2)
                        loss = loss_student
                        test_loss += loss.data.item()
                        testpred_y = (torch.max(logit_s2, 1)[1]).cpu().numpy()
                        accuracy2 = (testpred_y == batch_y2.squeeze().cpu().numpy()).astype(int).sum()
                        test_acc = test_acc + accuracy2
                        teststepsum = teststepsum + 1
                        prelist_temp.extend(testpred_y)
                    print('Test loss: {:.6f}, Test acc: {:.6f}'.format(test_loss / teststepsum, test_acc / len_test_data))
                    accacc = metrics.accuracy_score(truelist_sub, prelist_temp)
                    f1f1 = metrics.f1_score(truelist_sub, prelist_temp, average='weighted')

                    student_with_teacher_average_acc.append(accacc)
                    student_with_teacher_average_f1.append(f1f1)
                    student_with_teacher_prelist.append(prelist_temp)
                    student_with_teacher_truelist.append(truelist_sub)
                    early_stopping(eval_acc / len_val_data,student)
                    if early_stopping.early_stop:
                        print(early_stopping.best_acc)
                        print("Stopping early.")
                        break

        print("Fold:", str(fold + 1), "student_with_teacher Train Best Acc:", Best_trainacc_1)

    print("--------------------------------------------------------------")
    max_value_acc1, indices = find_max_values_and_indices(student_with_teacher_average_acc)
    max_index, max_value_f11 = find_max_value_by_indices(student_with_teacher_average_f1, indices)
    max_prelist1 = student_with_teacher_prelist[max_index]
    max_truelist1 = student_with_teacher_truelist[max_index]
    max_index1 = max_index
    print("Best Test ACC-based:")
    print("max_acc:", max_value_acc1)
    print("max_f1:", max_value_f11)
    print("max_prelist:", max_prelist1)
    print("max_truelist:", max_truelist1)
    print("max_index:", max_index1)

    max_value_f12, indices = find_max_values_and_indices(student_with_teacher_average_f1)
    max_index, max_value_acc2 = find_max_value_by_indices(student_with_teacher_average_acc, indices)
    max_prelist2 = student_with_teacher_prelist[max_index]
    max_truelist2 = student_with_teacher_truelist[max_index]
    max_index2 = max_index
    print("Best Test F1-based:")
    print("max_acc:", max_value_acc2)
    print("max_f1:", max_value_f12)
    print("max_prelist:", max_prelist2)
    print("max_truelist:", max_truelist2)
    print("max_index:", max_index2)

    max_value_acc_val, indices = find_max_values_and_indices(student_with_teacher_average_acc_val)
    max_index, max_value_acc3 = find_max_value_by_indices(student_with_teacher_average_acc, indices)
    max_value_f13 = student_with_teacher_average_f1[max_index]
    max_prelist3 = student_with_teacher_prelist[max_index]
    max_truelist3 = student_with_teacher_truelist[max_index]
    max_index3 = max_index
    print("Best Val ACC-based + Best Test ACC:")
    print("max_acc:", max_value_acc3)
    print("max_f1:", max_value_f13)
    print("max_prelist:", max_prelist3)
    print("max_truelist:", max_truelist3)
    print("max_index:", max_index3)

    max_value_acc_val, indices = find_max_values_and_indices(student_with_teacher_average_acc_val)
    max_index, max_value_f14 = find_max_value_by_indices(student_with_teacher_average_f1, indices)
    max_value_acc4 = student_with_teacher_average_acc[max_index]
    max_prelist4 = student_with_teacher_prelist[max_index]
    max_truelist4 = student_with_teacher_truelist[max_index]
    max_index4 = max_index
    print("Best Val ACC-based + Best Test F1:")
    print("max_acc:", max_value_acc4)
    print("max_f1:", max_value_f14)
    print("max_prelist:", max_prelist4)
    print("max_truelist:", max_truelist4)
    print("max_index:", max_index4)

    print("---------------------------------------------------------------------")
    print(time.asctime(time.localtime(time.time())))
    all_acc_student_with_teacher1.append(max_value_acc1)
    all_f1_student_with_teacher1.append(max_value_f11)
    all_truelist_student_with_teacher1.append(max_truelist1)
    all_prelist_student_with_teacher1.append(max_prelist1)
    all_maxindex1.append(max_index1)
    print("Best Test ACC-based:")
    print("Now all_acc_student_with_teacher1:", all_acc_student_with_teacher1)
    print("Now all_f1_student_with_teacher1:", all_f1_student_with_teacher1)
    print("Now all_truelist_student_with_teacher1:", all_truelist_student_with_teacher1)
    print("Now all_prelist_student_with_teacher1:", all_prelist_student_with_teacher1)
    print("Now all_maxindex1:", all_maxindex1)

    all_acc_student_with_teacher2.append(max_value_acc2)
    all_f1_student_with_teacher2.append(max_value_f12)
    all_truelist_student_with_teacher2.append(max_truelist2)
    all_prelist_student_with_teacher2.append(max_prelist2)
    all_maxindex2.append(max_index2)
    print("Best Test F1-based:")
    print("Now all_acc_student_with_teacher2:", all_acc_student_with_teacher2)
    print("Now all_f1_student_with_teacher2:", all_f1_student_with_teacher2)
    print("Now all_truelist_student_with_teacher2:", all_truelist_student_with_teacher2)
    print("Now all_prelist_student_with_teacher2:", all_prelist_student_with_teacher2)
    print("Now all_maxindex2:", all_maxindex2)

    all_acc_student_with_teacher3.append(max_value_acc3)
    all_f1_student_with_teacher3.append(max_value_f13)
    all_truelist_student_with_teacher3.append(max_truelist3)
    all_prelist_student_with_teacher3.append(max_prelist3)
    all_maxindex3.append(max_index3)
    print("Best Val ACC-based + Best Test ACC:")
    print("Now all_acc_student_with_teacher3:", all_acc_student_with_teacher3)
    print("Now all_f1_student_with_teacher3:", all_f1_student_with_teacher3)
    print("Now all_truelist_student_with_teacher3:", all_truelist_student_with_teacher3)
    print("Now all_prelist_student_with_teacher3:", all_prelist_student_with_teacher3)
    print("Now all_maxindex3:", all_maxindex3)

    all_acc_student_with_teacher4.append(max_value_acc4)
    all_f1_student_with_teacher4.append(max_value_f14)
    all_truelist_student_with_teacher4.append(max_truelist4)
    all_prelist_student_with_teacher4.append(max_prelist4)
    all_maxindex4.append(max_index4)
    print("Best Val ACC-based + Best Test F1:")
    print("Now all_acc_student_with_teacher4:", all_acc_student_with_teacher4)
    print("Now all_f1_student_with_teacher4:", all_f1_student_with_teacher4)
    print("Now all_truelist_student_with_teacher4:", all_truelist_student_with_teacher4)
    print("Now all_prelist_student_with_teacher4:", all_prelist_student_with_teacher4)
    print("Now all_maxindex4:", all_maxindex4)

print("---------------------------------------------------------------------")
print("Best Test ACC-based:")
print("Final all_acc_student_with_teacher1:", all_acc_student_with_teacher1)
print("Final all_f1_student_with_teacher1:", all_f1_student_with_teacher1)
print("Final all_truelist_student_with_teacher1:", all_truelist_student_with_teacher1)
print("Final all_prelist_student_with_teacher1:", all_prelist_student_with_teacher1)
print("Final all_maxindex1:", all_maxindex1, max(all_maxindex1))
print(sum(all_acc_student_with_teacher1) / len(all_acc_student_with_teacher1))
print(sum(all_f1_student_with_teacher1) / len(all_f1_student_with_teacher1))

print("---------------------------------------------------------------------")
print("Best Test F1-based:")
print("Final all_acc_student_with_teacher2:", all_acc_student_with_teacher2)
print("Final all_f1_student_with_teacher2:", all_f1_student_with_teacher2)
print("Final all_truelist_student_with_teacher2:", all_truelist_student_with_teacher2)
print("Final all_prelist_student_with_teacher2:", all_prelist_student_with_teacher2)
print("Final all_maxindex2:", all_maxindex2, max(all_maxindex2))
print(sum(all_acc_student_with_teacher2) / len(all_acc_student_with_teacher2))
print(sum(all_f1_student_with_teacher2) / len(all_f1_student_with_teacher2))

print("---------------------------------------------------------------------")
print("Best Val ACC-based + Best Test ACC:")
print("Final all_acc_student_with_teacher3:", all_acc_student_with_teacher3)
print("Final all_f1_student_with_teacher3:", all_f1_student_with_teacher3)
print("Final all_truelist_student_with_teacher3:", all_truelist_student_with_teacher3)
print("Final all_prelist_student_with_teacher3:", all_prelist_student_with_teacher3)
print("Final all_maxindex3:", all_maxindex3, max(all_maxindex3))
print(sum(all_acc_student_with_teacher3) / len(all_acc_student_with_teacher3))
print(sum(all_f1_student_with_teacher3) / len(all_f1_student_with_teacher3))

print("---------------------------------------------------------------------")
print("Best Val ACC-based + Best Test F1:")
print("Final all_acc_student_with_teacher4:", all_acc_student_with_teacher4)
print("Final all_f1_student_with_teacher4:", all_f1_student_with_teacher4)
print("Final all_truelist_student_with_teacher4:", all_truelist_student_with_teacher4)
print("Final all_prelist_student_with_teacher4:", all_prelist_student_with_teacher4)
print("Final all_maxindex4:", all_maxindex4, max(all_maxindex4))
print(sum(all_acc_student_with_teacher4) / len(all_acc_student_with_teacher4))
print(sum(all_f1_student_with_teacher4) / len(all_f1_student_with_teacher4))
