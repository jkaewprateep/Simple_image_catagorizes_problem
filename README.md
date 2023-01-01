# Simple_image_catagorizes_problem
For study image categorizes problem

## LSTM Layer ##

```
class MyLSTMLayer( tf.keras.layers.LSTM ):
	def __init__(self, units, return_sequences, return_state):
		super(MyLSTMLayer, self).__init__( units, return_sequences=return_sequences, return_state=return_state )
		self.num_units = units
		self.return_sequences = return_sequences
		self.return_state = return_state

	def build(self, input_shape):

		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="random_normal",
			trainable=True,
		)
		self.b = self.add_weight(
			shape=(self.units,), initializer="random_normal", trainable=True
		)

	def call(self, inputs):
	
		return tf.matmul(inputs, self.w) + self.b
```

## DataSet ##


```
variables = pd.read_excel('F:\\temp\\Python\\excel\\Book 7.xlsx', index_col=None, header=[0])

list_label = [ ]
list_Image = [ ]
list_file_actual = [ ]

for Index, Image, Label in variables.values:
	print( Label )
	list_label.append( Label )
	
	image = tf.io.read_file( Image )
	image = tfio.experimental.image.decode_tiff(image, index=0)
	list_file_actual.append(image)
	image = tf.image.resize(image, [32,32], method='nearest')
	list_Image.append(image)


list_label = tf.cast( list_label, dtype=tf.int32 )
list_label = tf.constant( list_label, shape=( 33, 1, 1 ) )
list_Image = tf.cast( list_Image, dtype=tf.int32 )
list_Image = tf.constant( list_Image, shape=( 33, 1, 32, 32, 4 ) )

dataset = tf.data.Dataset.from_tensor_slices(( list_Image, list_label ))
list_Image = tf.constant( list_Image, shape=( 33, 32, 32, 4 ) ).numpy()
```

## Model ##

```
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=( 32, 32, 4 ), name="input_01"),
	tf.keras.layers.Normalization(mean=3., variance=2., name="normal_01"),
	tf.keras.layers.Normalization(mean=4., variance=6., name="normal_02"),

	tf.keras.layers.Reshape((32 * 32, 4), name="reshape_01"),
	MyLSTMLayer(96, False, False),
	tf.keras.layers.Flatten(name="flattern_01"),
	tf.keras.layers.Dense(192, activation='relu', name="dense_02"),
	tf.keras.layers.Dense(2, name="dense_03"),
])
```

## Performing Tasks ##

```
plt.figure(figsize=(6,6))
plt.title("Actors recognitions")
for i in range(len(list_Image)):
	img = tf.keras.preprocessing.image.array_to_img(
		tf.constant( list_Image[i], shape=( 32, 32, 4 ) ),
		data_format=None,
		scale=True
	)
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)
	img_array = tf.expand_dims(img_array, 0)
	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	plt.subplot(6, 6, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(list_file_actual[i])
	plt.xlabel(str(round(score[tf.math.argmax(score).numpy()].numpy(), 2)) + ":" 
		+  str(list_label_actual[tf.math.argmax(score)]))
```
## TF-Lite for simple Devices ##

```
model.save(
    saved_model_dir,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True
)

tf_lite_model_converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)
tflite_model = tf_lite_model_converter.convert()

# Save the model.
with open( saved_model_dir + '\\model.tflite', 'wb' ) as f:
	f.write(tflite_model)
```

## Files and Directory ##

| File name | Description  |
--- | --- |
| sample.py | sample codes |
| Figure_1.png | result image |
| README.md | readme file |

## Result ##

![Result image 1](https://github.com/jkaewprateep/Simple_image_catagorizes_problem/blob/main/Figure_1.png "Result image 1")
