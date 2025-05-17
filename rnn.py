import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense # type: ignore

# توليد بيانات مثال
def generate_data(sequence_length=100, num_samples=1000):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.random()
        sequence = [start]
        for _ in range(sequence_length):
            sequence.append(sequence[-1] + np.random.normal(scale=0.1))
        X.append(sequence[:-1])
        y.append(sequence[1:])
    
    return np.array(X), np.array(y)

# تحضير البيانات
sequence_length = 20
X_train, y_train = generate_data(sequence_length, 10000)
X_test, y_test = generate_data(sequence_length, 1000)

# تعديل شكل البيانات لتناسب RNN (عينات, خطوات زمنية, خصائص)
X_train = X_train.reshape(-1, sequence_length, 1)
y_train = y_train.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)
y_test = y_test.reshape(-1, sequence_length, 1)

# بناء نموذج RNN
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(sequence_length, 1), return_sequences=True),
    Dense(1)
])

# تجميع النموذج
model.compile(optimizer='adam', loss='mse')

# طباعة ملخص النموذج
model.summary()

# تدريب النموذج
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# تقييم النموذج
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# عمل تنبؤات
predictions = model.predict(X_test[:5])
print("Predictions:", predictions)