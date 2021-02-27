import tensorflow as tf
from deeplab import flag_converter
f = flag_converter.FlagConverter()
tf.app.flags = f
from deeplab import train
t = train.Trainer("/home/dwarfeus/gdrive/Coding/DartsData/trainconfig.json")
print(t)