from sklearn import svm
import argparse
import numpy as np
import os

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description="process mesh to sdf using python multiprocessing module")
    parser.add_argument('latent_vec_dir', type=str, help='Path to latent_vector directory.')
    argument = parser.parse_args()
    load_dir = argument.latent_vec_dir

    # load test and training data
    X_train = np.load(os.path.join(load_dir, "train_data_latent_vect.npy"))
    Y_train = np.load(os.path.join(load_dir, "train_labels.npy"))
    X_test = np.load(os.path.join(load_dir, "test_data_latent_vect.npy"))
    Y_test = np.load(os.path.join(load_dir, "test_labels.npy"))


    lin_clf = svm.LinearSVC(C=0.01, intercept_scaling=True, class_weight='balanced', penalty='l2')
    lin_clf.fit(X_train, Y_train)
    train_score = lin_clf.score(X_train, Y_train)
    test_score = lin_clf.score(X_test, Y_test)
    print("ACCURACY: Training set: {0}, Test set accuracy: {1}".format(train_score, test_score))