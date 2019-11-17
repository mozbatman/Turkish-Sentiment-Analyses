# -*- coding: utf-8 -*-

## rnn (yenilenen sinir ağı) modelleri arasında gru kullanıldı


##                  IMPORT ISLEMLERİ
import speech_recognition as sr # SESİ METİNE ÇEVİRMEK İÇİN
import numpy as np 
import pandas as pd # DATA SETİNİ OKUMA VE DÜZENLEME
# RNN MODELİNİ KURMAK İÇİN KERAS KUTUPHANESİNİ KULLANACAĞIZ
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


## Data set okuma
dataset = pd.read_csv('hepsiburada.csv') 

etiket = dataset['Rating'].values.tolist()  # pandas serisi olarak alınıyor o yüzden to list yaptık
data = dataset['Review'].values.tolist()

## Test setini ayırma %80 train %20 test
bol_say = int(len(data) * 0.80) # %80 kısım hangi sayıya kadar ogrenıyoruz
x_train, x_text = data[:bol_say], data[bol_say:] 
y_train, y_test = etiket[:bol_say], etiket[bol_say:] 


## data setteki en cok kullanılan 10000 kelimeyi  ayırarak gerisini yok sayıyoruz
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data)

# Tokenleştirme
x_train_tokens = tokenizer.texts_to_sequences(x_train) ## egitim setini tokenleştirdik
x_test_tokens = tokenizer.texts_to_sequences(x_text)

## RNN aynı boyutta veri kümelerinde çalıştığından datalar kısaltcaz veya 0 atarak uzatcaz (eşitlemek için)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens ]   ## her yorumda kac token oldugu hesaplanır
num_tokens = np.array(num_tokens) ## dizi numpy arraye ceviriyoruz
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens) ## cumlelerin ortalama kelime sayısının 2 fazlasıyla standart sapmasını topluyoruz
max_tokens = int(max_tokens)

## pading ekleme 
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens) # pad_sequences keras kutupanesının fonksıyonudur
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)





### MODELİN OLUŞTURULMASI 


model = Sequential() ## ardışık model oluşturma
embedding_size = 50 ## kullanılcak kelime vektörlerinin uzunluğu
# input içindeki kelimelerin vektörlerini output olarak verir
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer'))          ## 10000 e 50 boyutunda matris oluşturuyoruz
model.add(GRU(units= 16, return_sequences=True)) # unit = 16 00 nöron sayısı // return_sequence bir sonraki layerları ekleyecegımız ıcın true
model.add(GRU(units= 8, return_sequences=True))
model.add(GRU(units= 4)) # son noron oldugu ıcın return_sequeces oto false bırakıyoruz
model.add(Dense(1, activation='sigmoid')) # cıkıs noronu oldugu ıcın Dense kullanıldı ve sigmoid fonk 0 ile 1 arası degerler doner yanı mutlu ve mutsuz

# Optimizasyon
optimizer = Adam(lr= 1e-3)  # 1e-3 ==== 0.001 

# Modelin Derlenmesi 
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy']) # yanlızca 2 sınıf oldugu ıcın loss için binary_crossentropy kullandık // metris masarı oranını gormek ıcın kullanıldı


## MODELİN EĞİTİMİ
model.fit(x_train_pad, y_train, epochs=5, batch_size=256) # epochs == verilerin kac kez eğitileceği







r = sr.Recognizer()
import emoji

with sr.Microphone() as source:
    print('speak : ')
    audio = r.listen(source)
    
    try:
        text = r.recognize_google(audio, language='tr-tr')
        print("soylediginiz : {}".format(text))
        texts = [text]
        tokens = tokenizer.texts_to_sequences(texts)
        tokens_pad = pad_sequences(tokens, maxlen= max_tokens)
        deger = model.predict(tokens_pad)
        print(deger)
        if deger > 0.5:
            print('Pozitif : \U0001F642')
        else:
            print('Negatif : \U0001F641') 
        
    except:
        print("anlamadım")