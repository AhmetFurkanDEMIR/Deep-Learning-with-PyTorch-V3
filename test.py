import torch.nn as nn
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os
import torchvision.transforms as transforms
import Model
import time 
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# modelin girdileri, açıklamalar modelde mevcuttr.
params_model={
 "input_shape": (3,96,96),
 "initial_filters": 8, 
 "num_fc1": 100,
 "dropout_rate": 0.25,
 "num_classes": 2,
 "size": 3
 }

#modelimiz
cnn_model = Model.Net(params_model)


path2weights="./models/weights.pt"
# öğrenmiş ağırlıklar modele aktarılır
cnn_model.load_state_dict(torch.load(path2weights))

print(cnn_model.eval())

# test cpu üzerinden gerçekleşecek
device = torch.device("cpu")
cnn_model=cnn_model.to(device) 


torch.manual_seed(0)

class histoCancerDataset(Dataset):
    
    def __init__(self, data_dir, transform,data_type="test"):      
    
        # eğitim verilerini tanımladık
        path2data=os.path.join(data_dir,data_type)

        # verilerin listesi
        filenames = os.listdir(path2data)

        # verilerin tamamı
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]

        # etiketler
        path2csvLabels=os.path.join(data_dir,"test_labels.csv")
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


def deploy_model(model,dataset,device, num_classes=2,sanity_check=False):

    len_data=len(dataset)
    

    y_out=torch.zeros(len_data,num_classes)
    

    y_gt=np.zeros((len_data),dtype="uint8")
    

    model=model.to(device)
    
    elapsed_times=[]
    with torch.no_grad():
        for i in range(len_data):
            x,y=dataset[i]
            y_gt[i]=y
            start=time.time()    
            y_out[i]=model(x.unsqueeze(0).to(device))
            elapsed=time.time()-start
            elapsed_times.append(elapsed)

            if sanity_check is True:
                break

    inference_time=np.mean(elapsed_times)*1000
    print("average inference time per image on %s: %.2f ms " %(device,inference_time))
    return y_out.numpy(),y_gt


# test verileri için
data_transformer = transforms.Compose([transforms.ToTensor()])

path2csv="test_labels.csv"
labels_df=pd.read_csv(path2csv)
labels_df.head()

data_dir = ""
# veriler
histo_test = histoCancerDataset(data_dir, data_transformer,data_type="test")
print(len(histo_test))


y_test_out,_=deploy_model(cnn_model,histo_test, device, sanity_check=False)


y_test_pred=np.argmax(y_test_out,axis=1)
print(y_test_pred.shape)


np.random.seed(0)

# görselleştirme
def show(img,y,color=True):

    npimg = img.numpy()


    npimg_tr=np.transpose(npimg, (1,2,0))
    
    if color==False:
        npimg_tr=npimg_tr[:,:,0]
        plt.imshow(npimg_tr,interpolation='nearest',cmap="gray")
    else:
 
        plt.imshow(npimg_tr,interpolation='nearest')
    plt.title("label: "+str(y))
    
grid_size=4
rnd_inds=np.random.randint(0,len(histo_test),grid_size)
print("image indices:",rnd_inds)

x_grid_test=[histo_test[i][0] for i in range(grid_size)]
y_grid_test=[y_test_pred[i] for i in range(grid_size)]

x_grid_test=utils.make_grid(x_grid_test, nrow=4, padding=2)
print(x_grid_test.shape)

plt.rcParams['figure.figsize'] = (10.0, 5)
show(x_grid_test,y_grid_test)

