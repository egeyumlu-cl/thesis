import pickle as pkl
import numpy as np
import os

import sklearn.metrics
import torch
import torch.nn as nn
import autoencoder as autoencoderLib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import scipy.stats as stats

import regularized_loss
import wandb
import fdd_loss
import contrastive_loss
import csv

api_key = "8f9dff561ae50c71efa0074e6ac57fa8fadae9f5"

learning_rate = 0.002
epoch = 20

directory = "subjects"
n = 400000
people_in_training = 1
people_in_outliers = 1
all_subject_ids = []
total_people = people_in_training + people_in_outliers


# threshold = 7.90 RMSE


#
# parameter_dict = {
#     'threshold': {
#         'distribution': 'uniform',
#         'min': 0.0,
#         'max': 1.0,
#     }
# }

# sweep_config = {
#     "name": "Lambda Sweep",
#     "method": "random",
#     "metric": {
#         'name': 'final_error',
#         'goal': 'minimize',
#     },
#     "parameters": parameter_dict,
# }
#
# sweep_id = wandb.sweep(sweep_config, project="FDD-Sweep")

def write_string_to_csv(string):
    with open("results.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([string])
        csvfile.write('\n')

def calculate_accuracy(guessed_labels, true_labels):
    assert len(guessed_labels) == len(true_labels), "Number of guessed labels and true labels must be the same."

    num_samples = len(guessed_labels)
    num_outliers = sum(true_labels)
    num_normals = num_samples - num_outliers

    # Calculate outlier accuracy
    outlier_correct = sum(1 for guessed, true in zip(guessed_labels, true_labels) if true and guessed)
    outlier_accuracy = (outlier_correct / num_outliers) * 100 if num_outliers > 0 else 0.0

    # Calculate normal accuracy
    normal_correct = sum(1 for guessed, true in zip(guessed_labels, true_labels) if not true and not guessed)
    normal_accuracy = (normal_correct / num_normals) * 100 if num_normals > 0 else 0.0

    # Calculate total accuracy
    total_correct = sum(1 for guessed, true in zip(guessed_labels, true_labels) if guessed == true)
    total_accuracy = (total_correct / num_samples) * 100

    return outlier_accuracy, normal_accuracy, total_accuracy


def train_model(autoencoder, train_loader, epochs, learning_rate, loss_per_epoch, loss_fn):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    final_loss = 0.0
    for i, epoch in enumerate(range(epochs)):
        running_loss = 0.0

        for data in train_loader:
            inputs = torch.from_numpy(data.astype(np.float64))
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss = running_loss / len(train_loader)
        loss_per_epoch.append(running_loss)

        # wandb.log({'epoch': epoch, 'loss': running_loss})
        print(f"Epoch {epoch + 1}: Average Loss = {running_loss}")
        # wandb.log({
        #     'rec_error': running_loss
        # })
        # if i == epochs - 1:
        #     (f"Epoch {epoch + 1}: Average Loss = {running_loss}")


def getGuessedValues(samples, autoencoder, threshold, loss):
    guessed = []
    for i, eval_sample in enumerate(samples):
        inputs = torch.from_numpy(np.array(eval_sample).astype(np.float64))
        result = detect_outlier(autoencoder, inputs, threshold, loss)
        guessed.append(result)
    return guessed


def getGuessedProbabilities(samples, autoencoder, loss, std, mean):
    guessed = []
    for i, eval_sample in enumerate(samples):
        inputs = torch.from_numpy(np.array(eval_sample).astype(np.float64))
        result = get_probability(mean, std, autoencoder, inputs, loss)
        guessed.append(result)
    return guessed


def getAUCforThreshold(eval_samples, autoencoder, eval_labels, rocs, threshold, loss):
    guessed = []
    for i, eval_sample in enumerate(eval_samples):
        inputs = torch.from_numpy(np.array(eval_sample).astype(np.float64))
        result = detect_outlier(autoencoder, inputs, threshold, loss)
        guessed.append(result)

    len(eval_labels)
    len(guessed)
    auc = roc_auc_score(eval_labels, guessed)
    print(eval_labels == guessed)
    rocs.append(auc)
    print(f"Threshold {threshold}: AUC = {auc}")
    # wandb.log({
    #     'auc': auc
    # })
    return auc


def plot_line_graph(x_values, y_values, x_label, y_label, title):
    plt.bar(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(45)
    plt.show()


def get_probability(mean, std_dev, autoencoder, point, loss_fn):
    loss = getLossForPoint(autoencoder, point, loss_fn)
    z_score = (loss - mean) / std_dev
    probability = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return probability


def split_array(array, n):
    split_arrays = []
    for i in range(0, len(array), n):
        split = array[i:i + n]
        if len(split) < n and i > 0:
            # If the last split is smaller than n, add samples from the start
            remaining = n - len(split)
            split = list(array[:remaining]) + list(split)
        split_arrays.append(split)
    return split_arrays


def split_array_steps(array, steps, n, number=2000, on=False):
    split_arrays = []
    eval_arrays = []
    total_samples = 0
    total_eval_samples = 0
    for i in range(0, len(array), n):
        split = array[i:i + n]
        steps_split = steps[i:i + n]
        if len(split) < n and i > 0:
            # If the last split is smaller than n, add samples from the start
            remaining = n - len(split)
            split = list(array[:remaining]) + list(split)
        sum_split = np.sum(steps_split)
        if sum_split > 100.0:
            if total_samples < number:
                split_arrays.append(split)
                total_samples = total_samples + 1
            elif (total_samples >= number) and on and total_eval_samples < 2000:
                eval_arrays.append(split)
                total_eval_samples = total_eval_samples + 1
            else:
                break

    return split_arrays, eval_arrays


def split_array_eval(array, split_size, eval_samples, eval_labels):
    for i in range(0, len(array), split_size):
        split = array[i:i + split_size]
        if len(split) < split_size and i > 0:
            # If the last split is smaller than n, add samples from the start
            remaining = split_size - len(split)
            split = array[:remaining].tolist() + split.tolist()
        eval_samples.append(split)
        eval_labels.append(False)


def detect_outlier(autoencoder, point, threshold, loss_fn):
    with torch.no_grad():
        point = torch.Tensor(point)
        output = autoencoder(point.unsqueeze(0))  # get it in the needed structure
        loss = loss_fn(output, point.unsqueeze(0))
        if loss > threshold:
            return True  # Point is an outlier
        else:
            return False  # Point is not an outlier


def getLossForPoint(autoencoder, point, loss_fn):
    with torch.no_grad():
        point = torch.Tensor(point)
        output = autoencoder(point.unsqueeze(0))  # get it in the needed structure
        loss = loss_fn(output, point.unsqueeze(0))
        return loss


def get_threshold(autoencoder, training_data, loss_fn, percentile=75):
    rec_errors = get_losses(autoencoder, training_data, loss_fn)
    threshold = np.percentile(rec_errors, percentile)
    return threshold


def get_losses(autoencoder, training_data, loss_fn):
    rec_errors = []
    for data in training_data:
        inputs = torch.from_numpy(data.astype(np.float64))
        outputs = autoencoder(inputs)
        loss = loss_fn(outputs, inputs)
        rec_errors.append(loss.detach().numpy())
    return rec_errors


def plot_histogram(loss, labels):
    fig, ax = plt.subplots()

    # Calculate the number of unique labels
    unique_labels = list(set(labels))
    num_labels = len(unique_labels)

    # Create a mapping of labels to numeric values
    label_map = {label: i for i, label in enumerate(unique_labels)}

    # Set the colormap to a categorical colormap with the number of unique labels
    cmap = plt.cm.get_cmap('tab10', num_labels)

    # Create a range of indices as the x-axis
    x = np.arange(len(loss))

    # Calculate the number of bins based on the square root of the number of samples
    num_bins = int(np.sqrt(len(loss)))

    # Plot the histogram for each label
    for label in unique_labels:
        # Filter the loss values corresponding to the current label
        filtered_loss = [l for l, lbl in zip(loss, labels) if lbl == label]

        # Assign color based on the label
        color = 'red' if label == 'Outlier' else cmap(label_map[label])

        # Plot the histogram for the current label with color based on the label
        ax.hist(filtered_loss, bins=num_bins, color=color, alpha=0.7, label=f'Label {label}')

    ax.legend()
    ax.set_xlabel('Loss')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Loss Values Colored by Labels')
    plt.show()

def main(non_outlier):
    loss_fn_type = "Regularized Loss"
    files = os.listdir(directory)

    all_people_samples = []
    eval_samples = []
    eval_labels = []
    eval_non_outlier_samples = []
    training_time_series = []
    non_outlier_id = 0
    training_initial = []

    # wandb.init(
    #     project="RMSE Lambda Learning Sweep Grid",
    # )

    # lambda_FDD = wandb.config.lambda_rate
    # lr = wandb.config.lr
    # lr = 0.0005 rmse
    lr = 0.0005
    # lambda_rate = 0.05
    # lr = 0.003
    # m = 10
    # m = wandb.config.m
    # lambda_factor = 0.909
    # loss_fn = fdd_loss.FDD_Loss(lambda_rate)
    # percentile = wandb.config.percentile
    # lr = 0.001 RMSE
    # loss_fn = regularized_loss.CustomLoss(lambda_factor)
    # lr = wandb.config.learning_rate
    # threshold = wandb.config.threshold
    loss_fn = nn.MSELoss()

    # wandb.run.name = f"RMSE-LR-{lr}"
    total_steps = []

    for j, file in enumerate(files):
        file_path = os.path.join(directory, file)  # Full path to the file

        file_base = os.path.basename(file_path)
        file_base_without_extension = os.path.splitext(file_base)[0]
        subject_id = int(file_base_without_extension)

        with open(file_path, "rb") as file:
            data = pkl.load(file)

        if j == non_outlier:
            # Select the first n rows of the DataFrame
            training_data_source = data.iloc[:n].copy()
            eval_data_source = data.iloc[n: n +  n // 2]

            non_outlier_id = subject_id
            print(f"Normal ID: {non_outlier_id}")

            # train_new, eval_new = split_array_steps(training_data_source["hr"], training_data_source["steps"].astype(float), 100, 4000, True)
            # eval_samples.append(eval_new)
            array = split_array(training_data_source["hr"], 100)
            split_array_eval(eval_data_source["hr"], 100, eval_samples, eval_labels)
            training_initial.append(array)

        else:
            # Select the first n rows of the DataFrame
            object_n = data.iloc[:n // 2].copy()
            # split_hr, eval_dump = split_array_steps(object_n["hr"], object_n["steps"].astype(float), 100)
            split = split_array(object_n["hr"], 100)
            # Add all the 120 samples
            all_people_samples.append(split)
            all_subject_ids.append(subject_id)

    # Flatten the data for training
    training_data = [sub_array for inner_array in training_initial for sub_array in inner_array]
    # eval_samples = [sub_array for inner_array in eval_samples for sub_array in inner_array]

    print(len(training_data))
    print(len(eval_samples))
    if not len(training_data) == 4000:
        return

    if not len(eval_samples) == 2000:
        return

    # Normalize the array
    np_array = np.array(training_data)
    training_mean = np.mean(np_array)
    training_std = np.std(np_array)

    # Normalized sets for samples
    normalized_training_set = (np_array - training_mean) / training_std

    # Double the autoencoder to match the data types

    autoencoder = autoencoderLib.Autoencoder()
    autoencoder = autoencoder.double()

    # wandb.watch(autoencoder, log_freq=100)
    # threshold = wandb.config.threshold

    # Train the weights and the biases in terms of reconstruction error
    loss_per_epoch = []

    for i in range(2000):
        eval_labels.append(True)

    # Train the autoencoder
    # train_model(autoencoder, normalized_training_set, epoch, lr, loss_per_epoch, loss_fn)


    total_accuracies = []
    normal_accuracies = []
    outlier_accuracies = []
    for i in range(len(all_people_samples)):
        print(i)
        current_index = i + 1
        if current_index == len(all_people_samples) + 1:
            break
        non_outliers = eval_samples
        outliers = [sub_array for inner_array in all_people_samples[current_index - 1:current_index] for sub_array
                    in
                    inner_array]
        eval_concatenated = non_outliers + outliers
        normalized_eval_set = (eval_concatenated - training_mean) / training_std
        # plot_histogram(get_losses(autoencoder, normalized_training_set, loss_fn), eval_labels)
        if not len(outliers) == 2000:
            continue
        guessed = getGuessedValues(normalized_eval_set, autoencoder,
                                    get_threshold(autoencoder, normalized_training_set, loss_fn, 15),
                                    loss_fn)
        losses_eval = get_losses(autoencoder, normalized_eval_set, loss_fn)
        flattened = []
        for element in losses_eval:
            flattened.append(float(element.item()))
        plot_histogram(flattened, eval_labels)
        # outlier_accuracy, normal_accuracy, total_accuracy = calculate_accuracy(guessed, eval_labels)
        # total_accuracies.append(total_accuracy)
        # normal_accuracies.append(normal_accuracy)
        # outlier_accuracies.append(outlier_accuracy)
        # print(f"Percentile: {15}, Accuracy: {np.mean(total_accuracies)}, Outlier: {np.mean(outlier_accuracies)}, Normal: {np.mean(normal_accuracies)}")

    return


# def main():
#     auc = 0.0
#     for i in range(5):
#         auc = auc + run()
#     print(auc)
#     print(auc/5)
#
# sweep_config = {
#     "name": "RMSE Learning Sweep Final",
#     "method": "grid",
#     "metric": {
#         'name': 'rec_error',
#         'goal': 'minimize',
#     },
#     "parameters":
#         {
#             # 'x': {'values': [60, 65, 70]},
#             # 'percentile': {'values': [10]},
#             'lr': {'values': [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]},
#             # 'lambda_rate': {'values': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]}
#             # 'lr': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.01},
#             # 'lambda_rate': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
#         }
# }
#
# sweep_id = wandb.sweep(sweep_config, project="RMSE Learning Sweep Final")
# wandb.agent(sweep_id, function=main)

# 0.57675 FDD Max AUC
# 6.9 FDD Threhold

# 'lr': {'values': [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]}, Regularized
# 'lambda_rate': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
# 'lr': {'values': [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]}, FDD
# 'lambda_rate': {'values': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]}


    # total_accuracies = []
    # normal_accuracies = []
    # outlier_accuracies = []
    # for i in range(len(all_people_samples)):
    #     current_index = i + 1
    #     if current_index == len(all_people_samples) + 1:
    #         break
    #     non_outliers = eval_samples
    #     outliers = [sub_array for inner_array in all_people_samples[current_index - 1:current_index] for sub_array in
    #                 inner_array]
    #     eval_concatenated = outliers + non_outliers
    #     normalized_eval_set = (eval_concatenated - training_mean) / training_std
    #     # plot_histogram(get_losses(autoencoder, normalized_training_set, loss_fn), eval_labels)
    #     if not len(outliers) == 2000:
    #         continue
    #     guessed = getGuessedValues(normalized_eval_set, autoencoder,
    #                                get_threshold(autoencoder, normalized_training_set, loss_fn, percentile), loss_fn)
    #     outlier_accuracy, normal_accuracy, total_accuracy = calculate_accuracy(guessed, eval_labels)
    #     total_accuracies.append(total_accuracy)
    #     normal_accuracies.append(normal_accuracy)
    #     outlier_accuracies.append(outlier_accuracy)
    #
    # print(
    #     f"Accuracy: {np.mean(total_accuracies)}, Outlier Accuracy: {np.mean(outlier_accuracies)}, Normal Accuracy: {np.mean(normal_accuracies)}")
    #
    # wandb.log({
    #     'accuracy': np.mean(total_accuracies)
    # })
    # wandb.log({
    #     'outlier_accuracy': np.mean(outlier_accuracies)
    # })
    # wandb.log({
    #     'normal_accuracy': np.mean(normal_accuracies)
    # })



# non_outliers = eval_samples
#     outliers = [sub_array for inner_array in all_people_samples[4:5] for sub_array in
#                 inner_array]
#     eval_concatenated = outliers + non_outliers
#     normalized_eval_set = (eval_concatenated - training_mean) / training_std
#     # plot_histogram(get_losses(autoencoder, normalized_training_set, loss_fn), eval_labels)
#
#     guessed = getGuessedValues(normalized_eval_set, autoencoder,
#                                get_threshold(autoencoder, normalized_training_set, loss_fn, 60), loss_fn)
#     outlier_accuracy, normal_accuracy, total_accuracy = calculate_accuracy(guessed, eval_labels)
#     auc = roc_auc_score(eval_labels, guessed)
#     print(f"Accuracy {total_accuracy}")
#     wandb.log({
#         'accuracy': total_accuracy
#     })
#     wandb.log({
#         'auc': auc
#     })
#     wandb.log({
#         'outlier_accuracy': outlier_accuracy
#     })
#     wandb.log({
#         'normal_accuracy': normal_accuracy
#     })

# for randomized_grou in [ 16, 15, 14]:
#     main(randomized_grou)
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = np.array([[49.70, 49.76, 49.97],
# [58.92, 58.95,56.44],
# [51.76, 53.55, 54.77],
# [50.50, 50.87,51.63],
# [57.93, 60.35,60.03],
# [48.85, 48.40,48.53],
# [68.39, 67.70,68.78],
# [64.58, 64.89,63.88],
# [56.85, 57.00,67.34],
# [50.79, 51.67,50.71]])
# z = ["RMSE", "Regularized","FDD"]


#
# def create_bar_graph(x_values, y_values, class_labels, title, x_label, y_label):
#     num_x = len(x_values)
#     num_y = len(y_values[0])
#     bar_width = 0.25
#
#     index = np.arange(num_x)
#     colors = ['r', 'g', 'b']  # You can customize the colors here
#
#     fig, ax = plt.subplots()
#
#     for i in range(num_y):
#         ax.bar(index + (i * bar_width), y_values[:, i], bar_width, color=colors[i], label=class_labels[i])
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     ax.set_title(title)
#     ax.set_xticks(index + (num_y * bar_width / 2))
#     ax.set_xticklabels(x_values)
#     ax.legend(loc='upper right')
#
#     # Add labels for each class in the top right corner
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles, labels, loc='upper right')
#
#     plt.show()
#
# create_bar_graph(x, y, z, "Accuracy per Individual", "Person", "Accuracy")

# Ran 4, 23, 7,21, 3, 28, 0,

for randomized_grou in [0]: # 14
    main(randomized_grou)