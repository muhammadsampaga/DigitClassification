# DigitClassification

![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/1a9db51d-cca3-4e00-be18-a5eb1d13a97c)
 
Import semua Library yang diperlukan, seperti TensorFlow,Numpy,Pandas dan Matplotlib

![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/d3141b95-8e11-4459-bb83-de249f58c806)
 
Membuat kelas myCallback menggunakan tf.keras.callbacks.Callback. berisi method on_epoch_end yang akan dipanggil disetiap akhir epoch saat pelatihan model nantinya. 
Jika Akurasi pelatihan model > 98% maka pelatihan model akan di stop dengan cara mengaktifkan self.model.stop_training menjadi True

![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/bdb69dc5-1e2e-42af-bf83-0541b2787791)
 
Mengimport MNIST Dataset menggunakan modul tf.keras.datasets.mnist.
Dataset MNIST adalah kumpulan data gambar angka tulisan tangan berukuran 28x28 piksel
Membagi data menjadi empat bagian X_train, y_train, X_test, y_test.

X_train berisi gambar-gambar angka 0 – 9, yang akan digunakan untuk pelatihan,
y_train berisi label-label sesuai gambar angka-angka pada X_train.
X_test berisi gambar – gambar angka 0 – 9 yang akan digunakan untuk pengujian model.
y_test berisi label-label sesuai gambar angka-angka pada X_test.

Berikut gambaran sederhana bagaimana salah satu isi dataset X_train dan y_train. 
Yaitu, X_train[0] dan y_train[0]:
![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/7d5d8273-0f03-4378-ba0e-6354b0fd2bfc)

 
X_train[0] berisi gambar angka 5 dan y_train[0] berisi label angka 5


![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/39536544-9a31-4b2b-820d-6fc3890b1943)

X_train : (60000, 28, 28) artinya memiliki 60.000 gambar pelatihan dengan ukuran 28 x 28 piksel yang direpresentasikan dengan Matriks 2D
Y_train : (60000,)
Array 1D yang berisi 60000 label untuk X_train

X_test : (10000, 28, 28) artinya memiliki 10.000 gambar uji dengan ukuran 28 x 28 piksel yang direpresentasikan dengan Matriks 2D
Y_test : (10000,)
Array 1D yang berisi 10000 label untuk X_test


![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/7e9c6b84-6f60-481d-9472-9b7a3a988d51)

Representasi Matriks 2D X_train[0] yang dapat dilihat seperti angka 5. 
Nilai-nilai matriks diatas berisi antara 0 sampai 255, dimana 0 adalah warna hitam dan 255 adalah warna putih.



![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/c7b025b5-7c83-49f7-a52a-94ab99e380a6)
 
X_train, X_test = X_train / 255.0, X_test / 255.0
Menormalisasi nilai piksel pada X_train dan X_test dengan membagi nya dengan 255.
Normalisasi digunakan agar model CNN kita nantinya berkerja lebih baik.

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
Menambah dimensi baru pada X_train dan X_test pada akhir untuk memenuhi kebutuhan format input yang diterima oleh Model CNN.
Sebelum: (28, 28)  ---- > tidak dapat diterima model CNN
Sesudah: (28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
One-hot Encoding terhadap label dalam dataset. One-hot Encoding mengubah bentuk biner sesuai dengan jumlah kelas.
Pada kasus kali ini sebenarnya one-hot Encoding dapat tidak dilakukan, tapi dengan catatan nanti kita mengubah fungsi loss dibawah dengan “sparse_categorical_crossentropy”



![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/248bf769-85cc-4ed9-9e96-e59613671ad5)

Tf.keras.models.sequential berarti kita akan menggunakan Multi-Layer-Perceptron (MLP)
Conv2D Layer (32, (3,3), activation=’relu’, input_shape=(28,28,1))
Layer Input pada model kali ini memiliki 32 Filter dengan ukuran (3,3) dan Fungsi Aktivasi yang digunakan adalah ReLU. Layer ini menggunakan input shape (28, 28, 1) karena diatas tadi kita telah menambahkan dimensi pada gambar MNIST kita.

MaxPooling2D (2,2)
layer max pooling dengan ukuran filter 2x2, mengurangi ukura gambar menjadi setengah.
Pooling adalah proses untuk mengurangi resolusi gambar dengan tetap mempertahankan informasi pada gambar.
![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/8911fd47-6d30-4b14-bb53-cb9c912b4782)

 
Proses max pooling dipakai karena, jumlah filter yang digunakan pada proses konvolusi berjumlah banyak. Ketika kita menggunakan 64 filter pada konvolusi maka akan menghasilkan 64 gambar baru. Max pooling membantu mengurangi ukuran dari setiap gambar dari proses konvolusi.

Flatten()
Mentransformasikan matriks 2D menjadi vector 1D. digunakan agar dapat dipakai untuk Fully Connected Layers nantinya.

Dropout(0.5)
Menghapus secara acak neuron selama pelatihan yang akan membantu untuk mencegah overfitting. 
0.5 artinya setengah dari neuron akan di drop selama pelatihan.

Dense(128, activation= 'relu'),
Fully Connected Layer dengan 128 neuron dengan mengguanakn fungsi aktivasi ReLU	
Dense(10, activation= 'softmax')
Layer output yang memiliki 10 neuron sesuai dengan jumlah kelas pada dataset MNIST, yaitu 0 – 9. Untuk dataset yang memiliki 3 kelas atau lebih, gunakan fungsi aktivasi Softmax.
Fungsi aktivasi Softmax akan menghasilkan probabilitas dari kelas-kelas yang ada.

![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/955db9be-4494-4ba2-aec0-cb16c2b38209)
 
Pada model.compile kali saya menggunakan Loss Function nya yaitu “Categorical_crossentropy” karena kita telah melakukan one-hot-encoding diatas.
Optimizer yang digunakan adalah “adam” yang digunakan untuk mengoptimalkan bobot jaringan selama pelatihan dengan mengurangi nilai loss nya. “adam” adalah optimizer yang paling umum digunakan.
Matriks evaluasi yang akan mengukur seberapa akurat model dalam memprediksi kelas-kelas yang ada

![image](https://github.com/muhammadsampaga/DigitClassification/assets/60804326/7a1c9f85-a992-4a72-9511-eaa70a504164)
 
X_train dan y_train (label) akan digunakan untuk data pelatihan. Batch_size menggunakan ukuran sebesar 32, semakin besar batch_size maka akan semakin cepat proses modelling-nya.
Epoch jumlah iterasi model yang kita tentukan sebanyak 15 kali.
Validation_split yang akan digunakan untuk mengukur peforma model pada setiap epoch. Kali ini saya menggunakan 30% dari data sebagai data validation.
Callback yang telah kita buat diatas, yaitu akan men-stop pemodelan jika akurasi telah diatas 98% digunakan agar proses modelling tidak memakan banyak waktu.


 
Memvisualisasikan accuracy dari training dan validation, dapat dilihat terjadi kenaikan drastis dari iterasi-0 ke iterasi-1 dan terjadi peningkatan stabil seterusnya.

 
Mengetest model kita pada X_test dan y_test dan mendapatkan accuracy score 99% dan loss score sebanyak 0.02%



 
 
Mengetest model kita dengan gambar yang kita punya. Dengan memanfaatkan google colab untuk mengunggah gambar menggunakan modul google.colab import files

