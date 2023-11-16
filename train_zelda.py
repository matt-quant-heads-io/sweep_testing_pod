import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import os
import pandas as pd
import numpy as np
from keras.utils import np_utils


def train_zelda(combo_ids, sweep_params, mode, username):
    root_path = f"/scratch/{username}/overlay/sweep_testing_pod/data/zelda/{mode}" # For testing on matt's computer: f"/Users/matt/sweep_testing_pod/data/zelda/{mode}"
    root_path_prefix = f"{root_path}/comboID_{combo_id}"

    sweep_schema_path = f"{root_path}/sweep_schema.csv" # For testing on matt's computer: f"/Users/matt/sweep_testing_pod/data/zelda/{mode}/sweep_schema.csv"
    df_sweep_schema = pd.read_csv(sweep_schema_path)
    obs_size, goal_set_size, trajectory_length, training_dataset_size = sweep_params
    trajectories_to_cleanup = []
    
    if mode == "non_controllable":
        for sample_id in range(1,4):
            combo_id_path = f"{root_path}/comboID_{combo_id}
            sample_id_path = f"{combo_id_path}/sampleID_{sample_id}"
            if os.path.exists(f"{sample_id_path}/training.done"):
                continue

            model_path = f"{sample_id_path}/models"
            trajectories_path = f"{sample_id_path}/trajectories"

            dfs = []
            X = []
            y = []

            for file in os.listdir(trajectories_path):
                print(f"compiling df {file}")
                df = pd.read_csv(f"{trajectories_path}/{file}")[:1000]
                dfs.append(df)

            df = pd.concat(dfs)

            # df = df.sample(frac=1).reset_index(drop=True)
            y_true = df[['target']]
            y = np_utils.to_categorical(y_true)
            print(f"y: {y}")
            df.drop('target', axis=1, inplace=True)
            y = y.astype('int32')

            for idx in range(len(df)):
                x = df.iloc[idx, :].values.astype('float32').reshape((obs_size, obs_size, 8))
                X.append(x)

            X = np.array(X)

            model_abs_path = f"{model_path}/{model_count}.h5"

            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, 8), padding="SAME"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="SAME"),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="SAME"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(8, activation='softmax')
            ])

            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[tf.keras.metrics.CategoricalAccuracy()])
            mcp_save = ModelCheckpoint(model_abs_path, save_best_only=True, monitor='categorical_accuracy', mode='max')
            es = EarlyStopping(
                monitor='categorical_accuracy',
                min_delta=0,
                patience=50,
                verbose=0,
                mode='max',
                baseline=0.999,
                restore_best_weights=False,
                start_from_epoch=0
            )
            history = model.fit(X, y, epochs=250, steps_per_epoch=64, verbose=2, callbacks=[mcp_save, es])

            with open(f"{sample_id_path}/training.done", "w") as f:
                f.writelines(history)

            trajectories_to_cleanup.append(trajectories_path)
    
    elif mode == "controllable":
        for sample_id in range(1,4):
            combo_id_path = f"{root_path}/comboID_{combo_id}
            sample_id_path = f"{combo_id_path}/sampleID_{sample_id}"
            if os.path.exists(f"{sample_id_path}/training.done"):
                continue

            model_path = f"{sample_id_path}/models"
            trajectories_path = f"{sample_id_path}/trajectories"

            dfs = []
            X = []
            y = []

            for file in os.listdir(trajectories_path):
                print(f"compiling df {file}")
                df = pd.read_csv(f"{trajectories_path}/{file}")
                dfs.append(df)

            df = pd.concat(dfs)

            df = df.sample(frac=1).reset_index(drop=True)
            y_true = df[['target']]
            y = np_utils.to_categorical(y_true)
            df.drop('target', axis=1, inplace=True)
            y = y.astype('int32')


            num_enemies_signed = np_utils.to_categorical(df[["num_enemies_signed"]]-1)
            nearest_enemy_signed = np_utils.to_categorical(df[["nearest_enemy_signed"]]-1)
            path_length_signed = np_utils.to_categorical(df[["path_length_signed"]]-1)

            signed_inputs = np.column_stack((num_enemies_signed, num_enemies_signed, num_enemies_signed))

            df.drop("num_regions_signed", axis=1, inplace=True)
            df.drop("num_enemies_signed", axis=1, inplace=True)
            df.drop("nearest_enemy_signed", axis=1, inplace=True)
            df.drop("path_length_signed", axis=1, inplace=True)

            for idx in range(len(df)):
                x = df.iloc[idx, :].values.astype('int32').reshape((obs_size, obs_size, 8))
                X.append(x)

            X = np.array(X)

            model_abs_path = f"{model_path}/{model_count}.h5"
            inputs = [
                Input(shape=(obs_size, obs_size, 8), name="obs"),
                Input(shape=(signed_inputs.shape[1],), name="signed_inputs"),
            ]

            x = Conv2D(
                128,
                (3, 3),
                activation="relu",
                input_shape=(obs_size, obs_size, 8),
                padding="SAME",
            )(inputs[0])
            x = MaxPooling2D(2, 2)(x)
            x = Conv2D(128, (3, 3), activation="relu", padding="SAME")(x)
            x = Conv2D(256, (3, 3), activation="relu", padding="SAME")(x)
            x = Flatten()(x)
            x = Concatenate()([x] + inputs[1:])
            x = Dense(128)(x)

            final_output = [
                Dense(8, activation="softmax")(x),
            ]

            conditional_counting_cnn_model = Model(
                inputs=inputs, outputs=final_output, name="cnn_cond_counting_model"
            )
            conditional_counting_cnn_model.summary()

            conditional_counting_cnn_model.compile(
                loss=[
                    tf.keras.losses.CategoricalCrossentropy(name="cnn_cond_counting_model_loss")
                ],
                optimizer=SGD(),
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(name="cnn_cond_counting_model_acc")
                ],
            )

            counting_mcp_save = ModelCheckpoint(
                model_abs_path,
                save_best_only=True,
                monitor="cnn_cond_counting_model_acc",
                mode="max",
            )
            es = EarlyStopping(
                monitor='categorical_accuracy',
                min_delta=0,
                patience=50,
                verbose=0,
                mode='max',
                baseline=0.999,
                restore_best_weights=False,
                start_from_epoch=0
            )

            counting_history = conditional_counting_cnn_model.fit(
                [
                    X,
                    signed_inputs
                ],
                y,
                epochs=250,
                steps_per_epoch=64,
                verbose=2,
                callbacks=[counting_mcp_save, es],
            )

            with open(f"{sample_id_path}/training.done", "w") as f:
                f.writelines(counting_history)

            trajectories_to_cleanup.append(trajectories_path)

    return trajectories_to_cleanup



