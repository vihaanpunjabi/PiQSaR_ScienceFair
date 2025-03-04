import matplotlib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D line plotting
import serial




# Read CSV data
data = pd.read_csv('sensor_data.csv')


# Initialize dictionary to store objects with their respective pressure and flex values
data_dict = {}


for index, row in data.iterrows():
   obj = row['object']
   flex_pressure = [row['flex1'], row['flex2'], row['flex3'], row['flex4']]
  
   if obj in data_dict:
       data_dict[obj].append(flex_pressure)
   else:
       data_dict[obj] = [flex_pressure]


print("Data Dictionary:", data_dict)


# Extract features (pressure, flex1, flex2, flex3, flex4) and labels (object)
X = data[['flex1', 'flex2', 'flex3', 'flex4']]
y = data['object']  # Output (object classification)


# Encode labels (object) to integers
label_encoder = tf.keras.layers.StringLookup(output_mode="int", vocabulary=tf.unique(y)[0])
y_encoded = label_encoder(y)
y_encoded = tf.cast(y_encoded, dtype=tf.int32)


# Convert features and labels to tensors
X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_encoded, dtype=tf.int32)


# Split dataset into training and testing sets (80% training, 20% testing)
dataset_size = len(X_tensor)
train_size = int(0.8 * dataset_size)


X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]
X_test = X_tensor[train_size:]
y_test = y_tensor[train_size:]


# Build the neural network model
model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
   tf.keras.layers.Dense(32, activation='relu'),
   tf.keras.layers.Dense(len(label_encoder.get_vocabulary()), activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')


# Save and load the model
model.save('sensor_model_tf.h5')
# Load trained model
model = tf.keras.models.load_model('sensor_model_tf.h5')


# Initialize serial communication
ser = serial.Serial('/dev/cu.usbmodem101', 9600, timeout=1)


def predict_object(flex_values):
   input_tensor = tf.convert_to_tensor([flex_values], dtype=tf.float32)
   predictions = model.predict(input_tensor)
   predicted_class_idx = tf.argmax(predictions[0]).numpy()
   predicted_object_name = label_encoder.get_vocabulary()[predicted_class_idx]  # Convert index back to label
   return predicted_object_name


def plot_arm(flex_values):
   center = np.array([0, 0, 0])
   arm_length = 7
  
   # Normalize flex values (assuming 0 means fully bent and 255 means straight)
   angles = np.radians(180 - (np.array(flex_values) / 255 * 180))
   print(angles)
   # Define arm segments
   num_segments = 10
   segment_length = arm_length / num_segments
  
   directions = [
       np.array([1, 0, -1]),  # Arm 1 (X-axis, bending downward)
       np.array([0, 1, -1]),  # Arm 2 (Y-axis, bending downward)
       np.array([-1, 0, -1]), # Arm 3 (-X-axis, bending downward)
       np.array([0, -1, -1])  # Arm 4 (-Y-axis, bending downward)
   ]
  
   plt.clf()
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
  
   for i in range(4):
       bend_angle = angles[i]
       direction = directions[i]
      
       # Start at the center and iteratively bend each segment downward
       points = [center]
       for j in range(num_segments):
           rotation_matrix = np.array([
            [np.cos(bend_angle / num_segments), -np.sin(bend_angle / num_segments), 0],
            [np.sin(bend_angle / num_segments), np.cos(bend_angle / num_segments), 0],
            [0, 0, 1]  # Rotation around the Z-axis (adjust as needed)
        ])
           direction = rotation_matrix @ direction
           new_point = points[-1] + segment_length * direction
           points.append(new_point)
      
       points = np.array(points)
       ax.plot(points[:, 0], points[:, 1], points[:, 2], label=f"Arm {i+1} (Flex={flex_values[i]})")
  
   ax.scatter(center[0], center[1], center[2], color="red", s=100, label="Center Joint")
   ax.set_title("Quadruple Arm Bending Visualization")
   ax.set_xlabel("X Axis")
   ax.set_ylabel("Y Axis")
   ax.set_zlabel("Z Axis")
   ax.legend()
   plt.pause(1)


def main():
   plt.ion()
   while True:
       if ser.in_waiting > 0:
           line = ser.readline().decode('utf-8').strip()
           try:
               split = line.split()
               if len(split) == 4:
                   flex_values = list(map(float, split))
                   predicted_object = predict_object(flex_values)
                   predicted_object_name = predict_object(flex_values)
                   print(f'Predicted Object: {predicted_object_name}, Flex Values: {flex_values}')
                   plot_arm(flex_values)

               else:
                   print("Error: Incorrect data format received.")
           except ValueError:
               print("Error: Could not convert values.")


if __name__ == "__main__":
   main()


