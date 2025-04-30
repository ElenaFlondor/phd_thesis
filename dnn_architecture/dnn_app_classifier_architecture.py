import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_dnn_model():
    # Input layers
    input_desc = Input(shape=(4096,), name="Descriptions")
    input_manifest = Input(shape=(200,), name="Manifest_Components")
    input_dynamic = Input(shape=(57,), name="Dynamic_Features")
    input_static = Input(shape=(129,), name="Static_Features")
    
    # Descriptions branch
    x_desc = Dense(512, activation='relu')(input_desc)
    x_desc = BatchNormalization()(x_desc)
    x_desc = Dropout(0.35)(x_desc)
    
    # Manifest Components branch
    x_manifest = Dense(256, activation='relu')(input_manifest)
    x_manifest = BatchNormalization()(x_manifest)
    x_manifest = Dense(64, activation='relu')(x_manifest)
    x_manifest = BatchNormalization()(x_manifest)
    x_manifest = Dense(32, activation='relu')(x_manifest)
    x_manifest = BatchNormalization()(x_manifest)
    x_manifest = Dropout(0.15)(x_manifest)

    # Dynamic Features branch
    x_dynamic = Dense(1024, activation='relu')(input_dynamic)
    x_dynamic = BatchNormalization()(x_dynamic)
    x_dynamic = Dropout(0.45)(x_dynamic)
    x_dynamic = Dense(32, activation='relu')(x_dynamic)
    x_dynamic = BatchNormalization()(x_dynamic)
    x_dynamic = Dropout(0.4)(x_dynamic)

    # Static Features branch
    x_static = Dense(128, activation='relu')(input_static)
    x_static = BatchNormalization()(x_static)
    x_static = Dropout(0.35)(x_static)
    x_static = Dense(32, activation='relu')(x_static)
    x_static = BatchNormalization()(x_static)
    x_static = Dropout(0.2)(x_static)

    # Concatenate all processed branches
    concatenated = Concatenate()([x_desc, x_manifest, x_dynamic, x_static])

    # Fully connected layers after concatenation
    x = Dense(512, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    # Output layer
    output = Dense(32, activation='linear', name="Output")(x)

    # Define the model
    model = Model(inputs=[input_desc, input_manifest, input_dynamic, input_static], outputs=output)

    # Compile the model (optional; modify based on task)
    model.compile(optimizer='adam', loss='mse', )

    return model

if __name__ == "__main__":
    model = build_dnn_model()
    model.summary()