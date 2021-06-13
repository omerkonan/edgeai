# AI Platform Code Base

Database üzerinden gelecek verilerin Amazon Sagemaker kullanılarak eğitilmesi, elde edilen modelin c kütüphanesi haline dönüştürülmesi ve bu kütüphanenin STWIN1K üzerinde test edilmesi süreçlerini barındırmaktadır. Süreç sagemaker_stater.py ile başlamaktır.

    ~cd /src/
    ~python sagemaker_starter.py

Sagemaker ile elde edilen tensorflow modeli create_c_library.py içerisinde bulunan createCLibrary() methodu ile c kütüphanesi haline getirilir.

    ~createCLibrary(model_path, tflite_model_name= "tflite_model", c_model_name="tflite_model_library")

#### modelpath: Keras model path
#### tflite_model_name: create tflite model with this name
#### c_model_name: create c library with this name

Oluşturulan "c_model_name.h" isimli c kütüpnesi embedded klasörü içerisindeki tf_lite projesi içerisine eklenir.

    ~cp ./models/c_models/c_model_name ./src/embedded/tf_lite/Core/Inc/
    
Model dosyası embedded projesine eklendikten sonra aşağıdaki komut ile proje yeniden build edilir.

    ~cd src/embedded/tflite/Debug
    ~make -j8 all
Çalıştırılan make dosyası ile "src/embedded/tflite/Debug" klasörü içerisinde "tf_lite.bin" isimli bir binary dosyası oluşturulur. Oluşturulan binary dosyası bilgisayara bağlanan boardun harici diskine kopyalandığında harici disk kendini resetler ve program başarılı bir şekilde yüklenmiş olur.

Eğer örnek board dışında farklı bir board üzerinde geliştirme yapılmak istenirse gerekli adımlar "src/embedded/readme.md" dosyası içerisinde detaylı olarak aktarılmıştır.