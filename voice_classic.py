"""
    Voice Classification using MFCC and Gaussian Mixtures
    Krzysztof Stawarz

"""

# task tools
import numpy as np
import os
import librosa
from sklearn.mixture import GaussianMixture
from typing import List
import argparse
from tqdm import tqdm

# visualization tools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se


def gmm_train(paths: List[str], n_mfcc: int, n_components: int) -> List[object]:

    """
    :param paths: list of paths to training recording
    :param n_mfcc: number of MFCC's
    :param n_components: number of Gaussian Mixture components
    :return: list of trained GM models
    """

    gmm_list: List[object] = list()

    # training process
    for file in paths:
        # loading train recording
        y, sr = librosa.load(file)

        # extract MFCC features for train recording
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # fitting GMM to MFCC features extracted
        # delete random_state=0 for different results each time
        gm = GaussianMixture(n_components=n_components, random_state=0).fit(np.transpose(m))

        gmm_list.append(gm)

    return gmm_list


def gmm_predict(paths: List[str], models: List[object], n_mfcc: int) -> List[int]:

    """
    :param paths: list of paths to a training recordings
    :param models: list of GMM models
    :param n_mfcc: number of MFCC's
    :return: list of predicted labels
    """

    predicted_labels: List[int] = list()

    # classification
    for file in paths:

        # loading test recording
        y, sr = librosa.load(file)

        # extract MFCC features for train recording
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # iterating over GMM's
        scores = [gmm.score(np.transpose(m)) for gmm in models]

        # the highest score is our predicted label
        predicted_label = np.argmax(scores) + 1
        predicted_labels.append(int(predicted_label))

    return predicted_labels


def get_accuracy(train_files: List[str], test_files: List[str],
                 n_mfcc: int = 39, n_components: int = 32) -> float:

    """
    :param train_files: list of files to train on
    :param test_files: list of files to test on
    :param n_mfcc: number of mfcc's; defaults in 39
    :param n_components: number of Gaussian Mixture components; defaults in 32
    :return: fraction of correct classified labels
    """

    actual_lbls: List[int] = [int(os.path.split(file)[-1].split(".")[0][-1]) for file in test_files]

    gmm_list: List[object] = gmm_train(train_files, n_mfcc=n_mfcc, n_components=n_components)
    predicted_lbls: List[int] = gmm_predict(test_files, gmm_list, n_mfcc=n_mfcc)

    # calculating accuracy
    accuracy: float = sum([l1 == l2 for l1, l2 in zip(actual_lbls, predicted_lbls)]) / len(train_files)

    return accuracy


def main(train_p: str, test_p: str) -> None:

    """
    :param train_p: path to a training folder
    :param test_p: path to a testing folder
    """

    # reading train recordings paths
    train_files = [os.path.join(train_p, file) for file in os.listdir(train_p)]
    train_files.sort()

    # reading test recordings paths
    test_files = [os.path.join(test_p, file) for file in os.listdir(test_p)]
    test_files.sort()

    # set some values for two dynamic parameters to calculate models accuracy based on them
    mfcc_vals = range(10, 41, 5)  # number of mfcc's in MFCC feature extraction
    n_components_vals = range(10, 41, 5)  # number of components in GaussianMixture model

    # get different classifiers accuracies
    print("\nClassifying recordings...")
    accuracies = {(mfcc, comp): get_accuracy(train_files, test_files, n_mfcc=mfcc, n_components=comp)
                  for mfcc in tqdm(mfcc_vals)
                  for comp in tqdm(n_components_vals, leave=False)}

    # create DataFrame to visualize data
    rows, columns, values = zip(*[(k[0], k[1], v) for k, v in accuracies.items()])

    df = pd.DataFrame({'n_mfcc': rows, 'n_components': columns, 'value': values})\
        .pivot(index='n_mfcc',
               columns='n_components',
               values='value')

    print("\n____ Accuracy ____")
    print(df)

    # creating custom colormap
    my_cmap = se.diverging_palette(240, 10,
                                   n=len(np.unique(list(accuracies.values()))),
                                   as_cmap=True)

    # creating seaborn heatmap
    acc_heatmap = se.heatmap(df, cmap=my_cmap, annot=True, fmt='.2f', cbar_kws={'label': 'accuracy'})

    # refactoring heatmap
    acc_heatmap.invert_yaxis()
    acc_heatmap.set_title('Classification accuracies')
    acc_heatmap.set_xlabel('Number of Gaussian Mixture components.')
    acc_heatmap.set_ylabel("Number of MFCC's")

    # saving plot to the device
    plt.savefig('accuracies.png', dpi=300, bbox_inches='tight')

    print("\nPlot saved!")

    # showing plot
    print("\nShowing plot...")
    plt.show()


if __name__ == "__main__":

    # creating command line argument parser
    parser = argparse.ArgumentParser(
        description="Give me recording of your voice and I'll tell you who you are...")

    parser.add_argument("--train_folder_path",
                        type=str,
                        help="Path to a folder with recordings to train on",
                        default="./voices/train")

    parser.add_argument("--test_folder_path",
                        type=str,
                        help="Path to a folder with recordings to test on",
                        default="./voices/test")

    args = parser.parse_args()

    main(args.train_folder_path, args.test_folder_path)
    quit()
