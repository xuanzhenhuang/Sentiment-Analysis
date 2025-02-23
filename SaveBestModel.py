#-------------pre-trained model without fine-tuning----------------
'''
from transformers import *
import torch
import time
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader

data = pd.read_csv(r'./minidatasets.csv')
data.head()

data = data.dropna()



class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv('./minidatasets.csv')
        self.data = self.data.dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentences = self.data.iloc[idx]['sentences']
        labels = self.data.iloc[idx]['labels']
        return sentences, labels
    
dataset = MyDataset()
for i in range(5):
    print(dataset[i])



trainset, validset = random_split(dataset, lengths=[0.8,0.2])
print(len(trainset), len(validset))



tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def collate_func(batch):
    # sentences = [x[0] for x in batch]
    # labels = [x[1] for x in batch]
    sentences, labels = [],[]
    for item in batch:
        sentences.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(sentences, max_length=250, padding="max_length", truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels)
    return inputs



trainloader = DataLoader(trainset, batch_size=12, shuffle=True,collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=24, shuffle=False,collate_fn=collate_func)

next(enumerate(trainloader))[1]

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

if torch.cuda.is_available():
    model = model.cuda()

def evaluate():
    model.eval()
    acc_num = 0
    start = time.time()
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k,v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits,dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()
    end = time.time()
    print(f'evaluation time:{end-start}')
    return acc_num / len(validset)

print(evaluate())
'''




#----------------------fine-tuned model------------------------
'''
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
import pandas as pd
import time

data = pd.read_csv(r'./minidatasets.csv')
data.head()

data = data.dropna()


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv('./minidatasets.csv')
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]['sentences'], self.data.iloc[index]['labels']
    
    def __len__(self):
        return len(self.data)
    
dataset = MyDataset()
for i in range(5):
    print(dataset[i])


trainset, validset = random_split(dataset, lengths=[0.8,0.2])
print(len(trainset), len(validset))

max = 0
for i in range(30):
    if len(trainset[i][0]) > max:
        max = len(trainset[i][0])
    # print(len(trainset[i][0]))
print(max)


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased-sentiments-student')

def collate_func(batch):
    # sentences = [x[0] for x in batch]
    # labels = [x[1] for x in batch]
    sentences, labels = [],[]
    for item in batch:
        sentences.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(sentences, max_length=250, padding="max_length", truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels)
    return inputs


trainloader = DataLoader(trainset, batch_size=12, shuffle=True,collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=24, shuffle=False,collate_fn=collate_func)

next(enumerate(trainloader))[1]


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased-sentiments-student")

if torch.cuda.is_available():
    model = model.cuda()

optimizer = Adam(model.parameters(), lr=2e-5)

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k,v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits,dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()
    return acc_num / len(validset)

def train(epoch = 3 ,log_step = 50):
    global_step = 0
    start = time.time()
    print('start training...')
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k,v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f'epoch:{ep}, global_step:{global_step}, loss:{output.loss.item()}')
            global_step += 1
        acc = evaluate()
        print(f'epoch:{ep}, acc:{acc}')
    end = time.time()
    print(f'evaluation time:{end-start}')

train()
'''


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pandas as pd
import time

# 读取数据集
data = pd.read_csv(r'./minidatasets.csv')
data = data.dropna()

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data.iloc[index]['sentences'], self.data.iloc[index]['labels']
    
    def __len__(self):
        return len(self.data)

dataset = MyDataset()
trainset, validset = random_split(dataset, lengths=[0.8, 0.2])

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # 改动模型即可

def collate_func(batch):
    sentences, labels = zip(*batch)
    inputs = tokenizer(list(sentences), max_length=250, padding="max_length", truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels)
    return inputs

trainloader = DataLoader(trainset, batch_size=12, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=24, shuffle=False, collate_fn=collate_func)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased") # 改动模型即可
if torch.cuda.is_available():
    model = model.cuda()

optimizer = Adam(model.parameters(), lr=2e-5)

save_path = './best_model.pth'
best_accuracy = 0.0

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()
    return acc_num / len(validset)

def train(epoch=3, log_step=50):
    global best_accuracy
    global_step = 0
    start = time.time()
    print('start training...')
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f'epoch:{ep}, global_step:{global_step}, loss:{output.loss.item()}')
            global_step += 1
        acc = evaluate()
        print(f'epoch:{ep}, acc:{acc}')
        
        # 如果当前准确率比之前的最高准确率要高，则保存模型
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), save_path)  # 保存模型参数
            print(f"New best model saved with accuracy: {best_accuracy}")

    end = time.time()
    print(f'training and evaluation time:{end-start}')

train()

# 加载最佳模型进行预测
model.load_state_dict(torch.load(save_path))
if torch.cuda.is_available():
    model = model.cuda()