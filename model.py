from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image


def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=1, target_size=(24, 24),
              class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale',
                                   class_mode=class_mode, target_size=target_size)

BS = 32
TS = (24, 24)
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(SPE, VS)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    # 32 узла, ядро размером 3х3
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    # 64 узла, ядро размером 3х3
    # случайным образом включать и выключать нейроны для улучшения сходимости
    Dropout(0.25),
    # сгладить слишком много измерений, нам нужен только вывод классификации
    Flatten(),
    # полностью подключен для получения всех необходимых данных
    Dense(128, activation='relu'),
    # еще один выпад ради сведения
    Dropout(0.5),
    # вывести softmax, чтобы сжать матрицу до выходных вероятностей
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

model.save('models/cnnmodel.h5', overwrite=True)