import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import os
import pandas as pd
import numpy as np
from keras.utils import np_utils


def train_zelda(domain, mode, username, debug):
    root_path = f"/scratch/{username}/sweep_testing_pod/data/{domain}/{mode}" # For testing on matt's computer: f"/Users/matt/sweep_testing_pod/data/{domain}/{mode}"

    sweep_schema_path = f"{root_path}/sweep_schema.csv" # For testing on matt's computer: f"/Users/matt/sweep_testing_pod/data/{domain}/{mode}/sweep_schema.csv"
    df_sweep_schema = pd.read_csv(sweep_schema_path)
    
    
    if mode == "non_controllable":
        for index, row in df_sweep_schema.iterrows():
            obs_size = row["sweep_param_obs_size"]
            prefix_filename_of_model = row["prefix_filename_of_model"]
            root_path = row["path_to_trajectories_dir_loc"].split("/trajectories")[0]
            model_path = f'{root_path}/{row["prefix_filename_of_model"].split("model_")[0]}'
            trajectories_path = row["path_to_trajectories_dir_loc"]

            print(f"obs_size: {obs_size}")
            
            for model_count in range(1,4):

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

                for idx in range(len(df)):
                    x = df.iloc[idx, :].values.astype('float32').reshape((obs_size, obs_size, 8))
                    X.append(x)

                X = np.array(X)

                model_abs_path = f"{model_path}/models/model_{model_count}.h5"

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
                history = model.fit(X, y, epochs=250, steps_per_epoch=64, verbose=2, callbacks=[mcp_save])
    
    elif mode == "controllable":
        for index, row in df_sweep_schema.iterrows():
            obs_size = row["sweep_param_obs_size"]
            prefix_filename_of_model = row["prefix_filename_of_model"]
            root_path = row["path_to_trajectories_dir_loc"].split("/trajectories")[0]
            model_path = f'{root_path}/{row["prefix_filename_of_model"].split("model_")[0]}'
            trajectories_path = row["path_to_trajectories_dir_loc"]

            for model_count in range(1,4):

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

                model_abs_path = f"{model_path}/models/model_{model_count}.h5"
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

                counting_history = conditional_counting_cnn_model.fit(
                    [
                        X,
                        signed_inputs
                    ],
                    y,
                    epochs=250,
                    steps_per_epoch=64,
                    verbose=2,
                    callbacks=[counting_mcp_save],
                )




