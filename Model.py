import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# çıktı boyutunu otomatik olarak hesaplamak 
def findConv2dOutShape(H_in,W_in,conv,pool=2):
    
    # aldığı argümanlar
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation
    
    
    H_out=np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    
    W_out=np.floor((W_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)
    
    if pool:
        H_out/=pool
        W_out/=pool
    
    return int(H_out),int(W_out) # verilerin yüksekliğini ve genişliğinin geri döndürür.

class Net(nn.Module):

    def __init__(self, params):

        super(Net, self).__init__()

        C_in,H_in,W_in=params["input_shape"] # verilerin tensör boyutu
        init_f=params["initial_filters"] # filitre boyutu
        num_fc1=params["num_fc1"]  # tam bağlı katmandaki birimler
        num_classes=params["num_classes"] # sınıflandırma türü
        self.dropout_rate=params["dropout_rate"] # iletim sönümü oranı
        self.boyut = params["size"] # filitre boyutu
        
        # CNN katmanlarımız
        # parametreler: 1-) aldığı resmin tensör boyutu
        # 2-) filitre sayısı
        # 3-) filitre boyutu
        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=self.boyut)
        h,w=findConv2dOutShape(H_in,W_in,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=self.boyut)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=self.boyut)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=self.boyut)
        h,w=findConv2dOutShape(h,w,self.conv4)
        # tamamen bağlı katmanlar
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)
        
        
    def forward(self, x):
        
        # katmanların aktivasyon fonksiyonları
        x = F.relu(self.conv1(x))
        # En büyükleri birktirme
        # Parametre sayısını azaltmak için max pooling kullanırız.
        # en büyükleri biriktirmek.
        # her MaxPooling katmanından sonra nitelik haritalarının boyutu yarıya düşer.
        # girdi nitelik haritasından pencereler çıkarıp her kanalın en büyük değerini almaktır. 
        # Her aktivasyon haritası için ayrı ayrı uygularız (Derinliği etkilemez.)
        # Filitrelemedeki en büyük değeri alır.
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        
        # iletim sönümü : 
        # modelin aşırı uydurma yapmasını engeller.
        # Sinir ağlarının düzleştirilmesinde kullanılır.
        # verdiğimiz orana göre elemanları sıfırlar.
        x=F.dropout(x, self.dropout_rate, training= self.training)
        x = self.fc2(x)
        
        # ikili sınıflandırma yapan son katman
        return F.log_softmax(x, dim=1)