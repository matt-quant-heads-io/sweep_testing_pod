import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import np_utils
import pandas as pd
import numpy as np

import constants


def get_paths_to_training_data(
    mode, goal_set_size, trajectory_length, training_dataset_size
):
    print(f"training_dataset_size: {training_dataset_size}")
    if training_dataset_size == 1_000_000:
        training_length_suffixes = ["1000000"]
    elif training_dataset_size == 100_000:
        training_length_suffixes = ["100000"]
    elif training_dataset_size == 500_000:
        training_length_suffixes = ["500000"]

    trajectories_dir = f"{constants.ZELDA_DATA_ROOT}/{mode}/trajectories"
    trajectory_filepaths_to_load = [
        f"{trajectories_dir}/goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}.csv"
        for training_dataset_size in training_length_suffixes
    ]

    return trajectory_filepaths_to_load


def train_zelda(combo_id, sweep_params, mode):
    # logger.info(f"Calling train_zelda with params {sweep_params}")
    print(f"Calling train_zelda with params {sweep_params}")
    models_to_skip_dir = f"{constants.ZELDA_DATA_ROOT}/{mode}/models_to_skip"
    if not os.path.exists(models_to_skip_dir):
        os.makedirs(models_to_skip_dir)

    obs_size, goal_set_size, trajectory_length, training_dataset_size = sweep_params
    models_to_train = [1, 2, 3]
    model_skip_filenames = [
        f"obssz_{obs_size}_goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}_{model_num}.csv"
        for model_num in models_to_train
    ]

    for model_num, model_skip_filename in zip(models_to_train, model_skip_filenames):
        if model_skip_filename in os.listdir(models_to_skip_dir):
            models_to_train.remove(model_num)
            print(f"Skipping model training for {model_skip_filename}.")

    # combo_id_dir = f"{constants.ZELDA_DATA_ROOT}/{mode}/comboID_{combo_id}"
    # if not os.path.exists(combo_id_dir):
    #     os.makedirs(combo_id_dir)

    models_dir = f"{constants.ZELDA_DATA_ROOT}/{mode}/models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if mode == "non_controllable":
        for model_num in models_to_train:
            dfs = []
            X = []
            y = []

            training_data_files_locs = get_paths_to_training_data(
                mode, goal_set_size, trajectory_length, training_dataset_size
            )

            try:
                for abs_filepath in training_data_files_locs:
                    print(f"Loading df {abs_filepath}")
                    df = pd.read_csv(abs_filepath)
                    dfs.append(df)
            except:
                return

            df = pd.concat(dfs)

            df = df.sample(frac=1).reset_index(drop=True)
            y_true = df[["target"]]
            y = np_utils.to_categorical(y_true)
            df.drop("target", axis=1, inplace=True)
            y = y.astype("int")

            for idx in range(len(df)):
                x = (
                    df.iloc[
                        idx,
                        (21 - obs_size)
                        * 8 : (((21 - obs_size) * 8) + (8 * obs_size**2)),
                    ]
                    .values.astype("int")
                    .reshape((obs_size, obs_size, 8))
                )
                X.append(x)

            X = np.array(X)

            model_abs_path = f"{models_dir}/obssz_{obs_size}_goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}_{model_num}.h5"

            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        128,
                        (3, 3),
                        activation="relu",
                        input_shape=(obs_size, obs_size, 8),
                        padding="SAME",
                    ),
                    tf.keras.layers.MaxPooling2D(2, 2)
                    if obs_size >= 3
                    else tf.keras.layers.MaxPooling2D(1, 1),
                    tf.keras.layers.Conv2D(
                        128, (3, 3), activation="relu", padding="SAME"
                    ),
                    tf.keras.layers.Conv2D(
                        256, (3, 3), activation="relu", padding="SAME"
                    ),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(8, activation="softmax"),
                ]
            )

            model.compile(
                loss="categorical_crossentropy",
                optimizer="SGD",
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
                ],
            )
            mcp_save = ModelCheckpoint(
                model_abs_path,
                save_best_only=True,
                monitor="categorical_accuracy",
                mode="max",
            )
            # es = EarlyStopping(
            #     monitor="categorical_accuracy",
            #     min_delta=0.00001,
            #     patience=250,
            #     verbose=0,
            #     mode="max",
            #     baseline=0.9999,
            #     restore_best_weights=True,
            #     start_from_epoch=10,
            # )
            history = model.fit(
                X,
                y,
                epochs=500,
                steps_per_epoch=4096,
                verbose=2,
                callbacks=[mcp_save],
            )

            df_history = pd.DataFrame(history.history)
            df_history.to_csv(
                f"{models_to_skip_dir}/obssz_{obs_size}_goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}_{model_num}.csv",
                index=False,
            )

    elif mode == "controllable":
        for model_num in models_to_train:
            dfs = []
            X = []
            y = []

            training_data_files_locs = get_paths_to_training_data(
                mode, goal_set_size, trajectory_length, training_dataset_size
            )

            try:
                for abs_filepath in training_data_files_locs:
                    print(f"Loading df {abs_filepath}")
                    df = pd.read_csv(abs_filepath)
                    dfs.append(df)
            except Exception as e:
                print(e)
                return

            df = pd.concat(dfs)

            df = df.sample(frac=1).reset_index(drop=True)
            y_true = df[["target"]]
            y = np_utils.to_categorical(y_true)
            df.drop("target", axis=1, inplace=True)
            y = y.astype("int")

            num_enemies_signed = np_utils.to_categorical(df[["num_enemies_signed"]] - 1)
            print(f"num_enemies_signed shape: {num_enemies_signed.shape}")
            nearest_enemy_signed = np_utils.to_categorical(
                df[["nearest_enemy_signed"]] - 1
            )
            print(f"nearest_enemy_signed shape: {nearest_enemy_signed.shape}")
            path_length_signed = np_utils.to_categorical(df[["path_length_signed"]] - 1)
            print(f"path_length_signed shape: {path_length_signed.shape}")

            signed_inputs = np.column_stack(
                (num_enemies_signed, nearest_enemy_signed, path_length_signed)
            )
            print(f"signed_inputs shape: {signed_inputs.shape}")

            df.drop("num_regions_signed", axis=1, inplace=True)
            df.drop("num_enemies_signed", axis=1, inplace=True)
            df.drop("nearest_enemy_signed", axis=1, inplace=True)
            df.drop("path_length_signed", axis=1, inplace=True)

            for idx in range(len(df)):
                x = (
                    df.iloc[
                        idx,
                        (21 - obs_size)
                        * 8 : (((21 - obs_size) * 8) + (8 * obs_size**2)),
                    ]
                    .values.astype("int")
                    .reshape((obs_size, obs_size, 8))
                )
                X.append(x)

            X = np.array(X)

            model_abs_path = f"{models_dir}/obssz_{obs_size}_goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}_{model_num}.h5"
            inputs = [
                Input(
                    shape=(
                        obs_size,
                        obs_size,
                        8,
                    ),
                    name="obs",
                ),
                Input(shape=(signed_inputs.shape[1],), name="signed_inputs"),
            ]

            x = Conv2D(
                128,
                (3, 3),
                activation="relu",
                input_shape=(obs_size, obs_size, 8),
                padding="SAME",
            )(inputs[0])
            x = MaxPooling2D(2, 2)(x) if obs_size >= 3 else MaxPooling2D(1, 1)(x)
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
                    tf.keras.losses.CategoricalCrossentropy(
                        name="cnn_cond_counting_model_loss"
                    )
                ],
                optimizer=SGD(learning_rate=0.001),
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(
                        name="cnn_cond_counting_model_acc"
                    )
                ],
            )

            # Cat Acc
            counting_mcp_save = ModelCheckpoint(
                model_abs_path,
                save_best_only=True,
                monitor="cnn_cond_counting_model_acc",
                mode="max",
            )
            # es = EarlyStopping(
            #     monitor="cnn_cond_counting_model_acc",
            #     min_delta=0.00001,
            #     patience=950,
            #     verbose=0,
            #     mode="max",
            #     baseline=0.9999,
            #     restore_best_weights=False,
            #     start_from_epoch=10,
            # )

            counting_history = conditional_counting_cnn_model.fit(
                [X, signed_inputs],
                y,
                epochs=500,
                steps_per_epoch=4096,
                verbose=2,
                # callbacks=[counting_mcp_save, es],
                callbacks=[counting_mcp_save],
            )

            df_history = pd.DataFrame(counting_history.history)
            df_history.to_csv(
                f"{models_to_skip_dir}/obssz_{obs_size}_goalsz_{goal_set_size}_trajlen_{trajectory_length}_tdsz_{training_dataset_size}_{model_num}.csv",
                index=False,
            )


# train_zelda("", (5, 10, 77, 1000000), "controllable")


# training_data_files_locs = get_paths_to_training_data("controllable", 10, 38, 100000)

# dfs = []
# try:
#     for abs_filepath in training_data_files_locs:
#         print(f"Loading df {abs_filepath}")
#         df = pd.read_csv(abs_filepath)
#         dfs.append(df)
# except:
#     pass

# df = pd.concat(dfs)

# df = df.sample(frac=1).reset_index(drop=True)
# y_true = df[["target"]]
# y = np_utils.to_categorical(y_true)
# print(f"y shape: {y.shape}")
# df.drop("target", axis=1, inplace=True)
# y = y.astype("int")

# num_enemies_signed = np_utils.to_categorical(df[["num_enemies_signed"]] - 1)
# print(f"num_enemies_signed shape: {num_enemies_signed.shape}")
# nearest_enemy_signed = np_utils.to_categorical(df[["nearest_enemy_signed"]] - 1)
# print(f"nearest_enemy_signed shape: {nearest_enemy_signed.shape}")
# path_length_signed = np_utils.to_categorical(df[["path_length_signed"]] - 1)
# print(f"path_length_signed shape: {path_length_signed.shape}")

# signed_inputs = np.column_stack(
#     (num_enemies_signed, nearest_enemy_signed, path_length_signed)
# )
# print(f"signed_inputs shape: {signed_inputs.shape}")

# df.drop("num_regions_signed", axis=1, inplace=True)
# df.drop("num_enemies_signed", axis=1, inplace=True)
# df.drop("nearest_enemy_signed", axis=1, inplace=True)
# df.drop("path_length_signed", axis=1, inplace=True)
