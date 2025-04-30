import numpy as np
from tensorflow.keras import layers, Model, Input, optimizers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

# Define or load these before running the script
# Example:
# input_shapes = [(20,), (10,)]  # Replace with actual input shapes
# num_classes = 3
# X_train, y_train = ...
# X_val, y_val = ...
# class_weight_dict = ...

def build_model(hp):
    inputs = []
    processed_inputs = []
    possible_units = [32, 64, 128, 256, 512, 1024]
    
    for shape in input_shapes:
        input_layer = Input(shape=shape)
        x = input_layer
        
        previous_units = hp.Choice(f"units_{shape}_0", possible_units)
        x = layers.Dense(previous_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float(f"dropout_{shape}_0", 0.0, 0.5, step=0.05))(x)
        
        for i in range(1, hp.Int(f"num_layers_{shape}", 1, 4)):
            available_units = [u for u in possible_units if u < previous_units]
            if len(available_units) == 0:
                break
            
            units = hp.Choice(f"units_{shape}_{i}", available_units)
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(hp.Float(f"dropout_{shape}_{i}", 0.0, 0.5, step=0.05))(x)
            
            previous_units = units
        
        inputs.append(input_layer)
        processed_inputs.append(x)
    
    merged = layers.Concatenate()(processed_inputs)

    previous_units = hp.Choice("final_units_0", possible_units)
    x = layers.Dense(previous_units, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hp.Float("final_dropout_0", 0.0, 0.5, step=0.05))(x)

    for i in range(hp.Int("num_final_layers", 1, 3)):
        units = hp.Choice(f"units_final_{i}", possible_units)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float(f"final_dropout_{i}", 0.0, 0.5, step=0.05))(x)
    
    output = layers.Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Tuner setup
tuner_dir = 'tuner_results/'
project = 'dataset_e'
number_of_configurations = 100
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=number_of_configurations,
    executions_per_trial=1,
    directory=tuner_dir,
    project_name=project,
    overwrite=True,
    seed=42  # Do not call np.random.seed() here
)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        mode='auto',
        patience=1,
        restore_best_weights=True
    )
]

# Start hyperparameter search
tuner.search(
    X_train,
    y_train,
    epochs=250,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks
)
