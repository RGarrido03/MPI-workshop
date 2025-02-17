from torch import nn as nn, flatten
from torch.nn import functional as f


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(6, activation=tf.nn.softmax)
# ])


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=0
        )
        self.fc1 = nn.Linear(
            in_features=32 * 36 * 36, out_features=128
        )  # 36x36 comes from (150-4)/2/2
        self.fc2 = nn.Linear(in_features=128, out_features=6)  # 6 classes

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = flatten(x, start_dim=1)  # Flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.softmax(self.fc2(x), dim=1)  # Softmax for multi-class classification
        return x
