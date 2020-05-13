# Deep-Learning-with-PyTorch-V3

* Görüntü sınıflandırması (görüntü tanıma olarak da bilinir) bilgisayardaki önemli bir görevdir.
* Bu görevde, görüntülerin bir ana nesne içerdiğini varsayıyoruz. İşte amacımız
ana nesneyi sınıflandırmak.
* İki tür görüntü sınıflandırması vardır: ikili sınıflandırma
ve çok sınıflı sınıflandırma.
* Bu projede, metastatik(Göğüs) kanseri tanımlamak için bir algoritmanın nasıl oluşturulacağını öğreneceğiz.


# Adımlarımız

* Veri kümesini keşfetme
* Özel bir veri kümesi oluşturma
* Veri kümesini bölme ve Verileri dönüştürme
* Veri yükleyici oluşturma
* Sınıflandırma modelinin oluşturulması
* Kayıp fonksiyonunun tanımlanması, Optimize ediciyi tanımlama
* Modelin eğitimi ve değerlendirilmesi
* Modelin testi


# Veri seti

* Histopatolojik Kanser Tespiti yarışmasında verilen veri setini kullanacağız. Bu yarışmanın amacı, görüntü yamalarını normal veya kötü huylu olarak sınıflandırmaktır.
* Veri seti: 7.2 Gb, 277485 veri 
* Veri seti (Histopathologic Cancer Detection) = https://www.kaggle.com/c/histopathologic-cancer-detection


# Modelimiz

![rhtrh](https://user-images.githubusercontent.com/54184905/81787781-3afb9800-950a-11ea-839b-8b83447c2db6.png)


# Modelimizin kaybı ve başarımı

![Screenshot_2020-05-13_03-09-14](https://user-images.githubusercontent.com/54184905/81787922-67171900-950a-11ea-9f66-3353382a37ed.png)

![Screenshot_2020-05-13_03-09-30](https://user-images.githubusercontent.com/54184905/81787926-67afaf80-950a-11ea-9def-b229ab0b095f.png)

