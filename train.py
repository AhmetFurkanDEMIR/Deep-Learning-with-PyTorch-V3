#%% --- Veri ön inceleme --- 

import pandas as pd
import Model

# eğitim setinin etiketleri yolu
path2csv="./train_labels.csv"

# etiketler okunur
labels_df=pd.read_csv(path2csv)

# ilk bes etiket
print(labels_df.head())

# etiketlerde 0 ve 1 olan veri sayısı
print(labels_df['label'].value_counts())

import matplotlib.pylab as plt
from PIL import Image, ImageDraw
import numpy as np
import os

# eğitim verilerinin yolu
path2train="./train/"

# inceleyeceğimiz verilerin siyah beyaz olmasını sağlıyoruz
color=False

# etiket değeri 1 olan veriler.
malignantIds = labels_df.loc[labels_df['label']==1]['id'].values

# bilgi sahibi olmak için verileri inceliyoruz.
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
nrows,ncols=3,3
for i,id_ in enumerate(malignantIds[:nrows*ncols]):
    full_filenames = os.path.join(path2train , id_ +'.tif')
 
    # resimler yüklenir.
    img = Image.open(full_filenames)

    draw = ImageDraw.Draw(img)
    draw.rectangle(((32, 32), (64, 64)),outline="green")

    plt.subplot(nrows, ncols, i+1) 
    if color is True:
        plt.imshow(np.array(img))
    else:
        plt.imshow(np.array(img)[:,:,0],cmap="gray")
    plt.axis('off')
    

# resim tensör boyutu
print("image shape:", np.array(img).shape)

#%% --- Veri parçalama ve tensörlere aktarma

import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os

# çıkışları sabit tutuyoruz
torch.manual_seed(0)

# verileri çekiyoruz
class histoCancerDataset(Dataset):
    
    def __init__(self, data_dir, transform,data_type="train"):      
    
        # eğitim verilerini tanımladık
        path2data=os.path.join(data_dir,data_type)

        # verilerin listesi
        filenames = os.listdir(path2data)

        # verilerin tamamı
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]

        # etiketler
        path2csvLabels=os.path.join(data_dir,"train_labels.csv")
        labels_df=pd.read_csv(path2csvLabels)

        # veri indexleri
        labels_df.set_index("id", inplace=True)

        # etiketleri alıyoruz
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]

        self.transform = transform
      
    def __len__(self):
        # return veriseti boyutu
        return len(self.full_filenames)
      
    def __getitem__(self, idx):
        # veriler ve etiketleri
        image = Image.open(self.full_filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]



# veri tensörleri oluşturmak için
import torchvision.transforms as transforms
# transformer
data_transformer = transforms.Compose([transforms.ToTensor()])

data_dir = ""
# oluşturduğumuz sınıf ile verileri alıyoruz
histo_dataset = histoCancerDataset(data_dir, data_transformer, "train")
# kaç adet veri ?
print(len(histo_dataset))

# 9. veri ve etiketi
img,label=histo_dataset[9]
print(img.shape,torch.min(img),torch.max(img))


# verileri random bir şekilde eğitim ve doğrulama verisi olarak ayırmak için.
from torch.utils.data import random_split

len_histo=len(histo_dataset)
# %80 eğitim
len_train=int(0.8*len_histo)
# %20 test
len_val=len_histo-len_train

# rastgele ayırma işlemi
train_ds,val_ds=random_split(histo_dataset,[len_train,len_val])

# adetleri
print("train dataset length:", len(train_ds))
print("validation dataset length:", len(val_ds))


# eğitim verisi
for x,y in train_ds:
    print(x.shape,y)
    break

# doğrulama verisi
for x,y in val_ds:
    print(x.shape,y)
    break


from torchvision import utils
import numpy as np

np.random.seed(0)

# eğitim verisinde çeşitlendirme yaparak, verisetinde artış sağlıyoruz.
train_transformer = transforms.Compose([
    
    # verinin yatayda döndürülme açısı.
    transforms.RandomHorizontalFlip(p=0.5),  
    
    # verinin dikeyde döndürülme açısı
    transforms.RandomVerticalFlip(p=0.5),  
    # -45, 45 derece arasında döndürme işlemi
    transforms.RandomRotation(45),

    # görüntüyü 96,96 olarak yeniden boyutlandırır.         
    transforms.RandomResizedCrop(96,scale=(0.8,1.0),ratio=(1.0,1.0)),
    
    # veriler tensör haline getirilir
    transforms.ToTensor()])             

# doğrulama verilerini sadece tensör haline getiriyoruz.
val_transformer = transforms.Compose([transforms.ToTensor()])     


# dönüştürme işlevlerinin üzerine yaz
train_ds.transform=train_transformer
val_ds.transform=val_transformer

from torch.utils.data import DataLoader

# 32 lik yığınlarla veriyi karıştırarak bir veri tensörü oluşturuyoruz.
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# veriyi karıştırmadan 64 lük yığınlarla veri tensörü
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)  


for x, y in train_dl:
    print(x.shape) # veri tensör noyutu
    print(y.shape) # etiketi
    break


for x, y in val_dl:
    print(x.shape) # tensör
    print(y.shape) # etiket
    break



# doğrulama veri kümesinin etiketleri
y_val=[y for _,y in val_ds]    


#%% --- Modelimiz ve parametreleri ---


# aldığı parametreler
params_model={
        "input_shape": (3,96,96), # aldığı tensör boyutu
        "initial_filters": 8, # filitre sayısı, 2 katına çıkarak artacaktır
        "num_fc1": 100, # tam bağlı katmandaki birimler
        "dropout_rate": 0.50, # iletim sönümü oranı
        "num_classes": 2, # 2 sınıf far
        "size": 3 # filitre boyutu
            }

# modelimi Model.py adlı dosyadaki Net adlı class dan yaratıyoruz.
cnn_model = Model.Net(params_model)


# modelin hangi donanımdan eğitileceğine karar veriyoruz.
device = torch.device("cpu")
cnn_model=cnn_model.to(device) 

# model katmanları hakkında bilgiler
print(cnn_model)

# modelin eğitileceği donanım
print(next(cnn_model.parameters()).device)

#%% --- Modeli eğitme ve yardımcı fonksiyonlar

from torchsummary import summary

# modelimizin daha detaylı resmi
summary(cnn_model, input_size=(3, 96, 96))

# eniyileme:
# ağımızın girdisi olan veri ile oluşturduğu kaybı göz önünde
# bulundurarak kendisini güncelleme mekanizması
from torch import optim
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)

# modelin eğitileceği donanım
device = torch.device("cpu")

from torch.optim.lr_scheduler import ReduceLROnPlateau

# öğrenme hızı, zamanlayıcı vb. tanımlamalar.
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)


# İlk olarak, doğru tahmin sayısını saymak için bir yardımcı işlev geliştirelim
def metrics_batch(output, target):
    # çıktı sınıfı
    pred = output.argmax(dim=1, keepdim=True)
    
    # çıktı sınıfını hedef sınıfla karşılaştır
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

# öğrenme oranı al
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# Ardından, toplu iş başına kayıp değerini hesaplamak için bir yardımcı işlev geliştireceğiz
def loss_batch(loss_func, output, target, opt=None):
    
    # kayıp
    loss = loss_func(output, target)
    
    # performans metriği
    metric_b = metrics_batch(output,target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


# tüm veri kümesi için kayıp değeri
def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    
    running_loss=0.0
    running_metric=0.0
    len_data=len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        
        # toplu işi donanımda yapmak
        xb=xb.to(device)
        yb=yb.to(device)
        
        # model çıktısı
        output=model(xb)
        
        # parti başına kayıp
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        
        # kaybı güncelleme
        running_loss+=loss_b
        
        # çalışan metriği güncelle
        if metric_b is not None:
            running_metric+=metric_b

        if sanity_check is True:
            break
    
    # ortalama kayıp değeri
    loss=running_loss/float(len_data)
    
    # ortalama metrik değer
    metric=running_metric/float(len_data)
    
    return loss, metric


# --- eğitim fonksiyonu ---

def train_val(model, params):
    
    # modelin aldığı parametreler.
    num_epochs=params["num_epochs"] # döngü sayısı
    loss_func=params["loss_func"] # kayıp fonksiyonu
    opt=params["optimizer"] # eniyileme fonksiyonu
    train_dl=params["train_dl"] # eğitim verisi
    val_dl=params["val_dl"] # doğrulama verisi
    sanity_check=params["sanity_check"] # akıllı kontrol
    lr_scheduler=params["lr_scheduler"] # öğrenme hızı, zamanlayıcı vb. tanımlamalar.
    path2weights=params["path2weights"] # en iyi modelin kaydedileceği yer
    
    # kayıp geçmişi
    loss_history={
        "train": [],
        "val": [],
    }
    
    # metrik geçmişi
    metric_history={
        "train": [],
        "val": [],
    }
    
    # en iyi performans gösteren model için ağırlıkların kopyası
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss=float('inf')
    
    # döngü
    for epoch in range(num_epochs):
        
        # mevcut öğrenme oranı
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        
        # eğitim veri kümesi üzerinde eğitim modeli
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)

        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        # doğrulama veri kümesi üzerinde doğrulama modeli   
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        
       
        # en iyi model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # ağırlıkların saklandığı dosya
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        
        # doğrulama veri kümesi için kayıp ve metriği toplamak
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 

        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
        print("-"*10) 

    # en iyi modelin ağırlığı yüklenir.
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history



import torch.nn as nn
import copy

loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

params_train={
 "num_epochs": 30,
 "optimizer": opt,
 "loss_func": loss_func,
 "train_dl": train_dl,
 "val_dl": val_dl,
 "sanity_check": False,
 "lr_scheduler": lr_scheduler,
 "path2weights": "./models/weights.pt",
}

# modelin eğitimi
cnn_model,loss_hist,metric_hist=train_val(cnn_model,params_train)

num_epochs=params_train["num_epochs"]

#%% --- Eğitimin başarımını ve kaybını görüntüleme

plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
