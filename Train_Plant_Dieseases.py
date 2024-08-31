import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Training image preprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",  # Changed to categorical
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

# Validation image preprocessing
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",  # Changed to categorical
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

# Display shapes of sample batches
for x, y in training_set:
    print(x, x.shape)
    print(y, y.shape)
    break

print("Validation set sample:")
for x, y in validation_set:
    print(x, x.shape)
    print(y, y.shape)
    break

# Building the model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2)) 

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Flatten())  # Flatten the output before Dense layers
model.add(Dropout(0.25))  # Add Dropout to prevent overfitting

model.add(Dense(units=1500, activation='relu'))
model.add(Dropout(0.4))  # Add Dropout to prevent overfitting

# Output layer softmax gives output in probability
model.add(Dense(units=38, activation='softmax'))

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Model Training
training_history = model.fit(x=training_set, validation_data=validation_set, epochs=10)

#Model Evaluation
train_loss,train_acc=model.evaluate(training_set)
print(train_loss,train_acc)

#save
model.save("trained_model.keras")

training_history.history

#Recording History in json
import json
with open("training_history.json","w") as f:
    json.dump(training_history.history,f)

training_history.history['accuracy'] 
#training_history.history['val_accuracy'] 


epochs=[i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')

class_name=validation_set.class_names
class_name

test_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",  # Changed to categorical
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,#passed squential from top to bottom validation set is used as it is sorted beforehand
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

y_pred=model.predict(test_set)
y_pred,y_pred.shape

predicted_categories=tf.argmax(y_pred,axis=1)
predicted_categories

true_categories=tf.concat([y for x,y in test_set],axis=0)
true_categories

Y_true=tf.argmax(true_categories,axis=1)
Y_true
from sklearn.metrics import classification_report,confusion_matrix
print (classification_report(Y_true,predicted_categories,target_names=class_name))

cm=confusion_matrix(Y_true,predicted_categories)
cm

##confusion matrix visulization
plt.figure(figsize=(40,40))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show

