#!/usr/bin/env python
# coding: utf-8

# In[45]:


dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'


# In[46]:


import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


# # 各パス指定

# # 分類数設定

# In[47]:


NUM_CLASSES = 27


# # 学習データ読み込み

# In[48]:


X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))


# In[49]:


y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)


# # モデル構築

# In[51]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


# In[52]:


model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)


# In[53]:


# モデルチェックポイントのコールバック
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# 早期打ち切り用コールバック
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)


# In[54]:


# モデルコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# # モデル訓練

# In[55]:


model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)


# In[56]:


# モデル評価
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)


# In[57]:


# 保存したモデルのロード
model = tf.keras.models.load_model(model_save_path)


# In[58]:


# 推論テスト
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


# # 混同行列

# In[59]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)


# # Tensorflow-Lite用のモデルへ変換

# In[60]:


# 推論専用のモデルとして保存
model.save(model_save_path, include_optimizer=False)


# In[61]:


# モデルを変換(量子化)
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)


# # 推論テスト

# In[62]:


interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()


# In[63]:


# 入出力テンソルを取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[64]:


interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))


# In[65]:


get_ipython().run_cell_magic('time', '', "# 推論実施\ninterpreter.invoke()\ntflite_results = interpreter.get_tensor(output_details[0]['index'])")


# In[66]:


print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))


# In[ ]:




