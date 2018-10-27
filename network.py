from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten,GlobalAveragePooling2D,Dropout

def vgg_network():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=[224,224,3])
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(61, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return  base_model,model
