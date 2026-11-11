# ISPR Assignments

A collection of ISPR midterm assignments by **Oleksiy Nedobiychuk**, exploring image descriptors, unsupervised representation learning, and recurrent neural networks. The experiments are presented as Jupyter notebooks with code, plots, results, and discussion.

## Notebooks

| Notebook | Topic | Main experiments |
| --- | --- | --- |
| [Assignment1_6_o_nedobiychuk.ipynb](Assignment1_6_o_nedobiychuk.ipynb) | SIFT descriptors | Preprocess horse images, extract and match SIFT features, and compare DBSCAN and K-means clustering against manually created foreground masks. Includes PCA visualizations, silhouette scores, and confusion matrices. |
| [assignment_2_3.ipynb](assignment_2_3.ipynb) | Restricted Boltzmann Machine from scratch | Implement and train an RBM with NumPy on binarized MNIST images. Use hidden activations as features for AdaBoost, an MLP, logistic regression, SVC, and Gaussian Naive Bayes. |
| [assignment_3_2.ipynb](assignment_3_2.ipynb) | Benzene prediction | Explore the UCI Air Quality dataset and compare Phased LSTM, standard LSTM, and an adapted xLSTM implementation. |
| [assignment_3_3.ipynb](assignment_3_3.ipynb) | Benzene prediction, related version | A closely related notebook covering the same recurrent-model experiments, including prediction from sensor readings and next-value prediction. |

The air-quality notebooks evaluate predictions using mean squared error, mean absolute error, Pearson correlation, and R².

## Setup

Use a Python 3 environment. The air-quality notebooks record Python **3.11.9** in their metadata; the repository does not include a dependency lockfile or a pinned environment.

From the project directory, create an environment and install the packages imported by the notebooks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy pandas scipy matplotlib seaborn scikit-learn opencv-python torch tqdm mlxtend statsmodels
jupyter lab
```

Open a notebook in JupyterLab, configure its data paths as described below, and execute the cells in order. Each notebook also contains an **Open in Colab** link. The SIFT notebook uses Google Colab APIs directly; running it locally requires replacing Drive mounting and `cv2_imshow` with local file access and image display code.

The install command supplies the main dependencies; compatibility with every notebook cell has not been verified against current package versions.

## Data and file paths

Datasets and trained model files are not included in this repository.

### Horse images and masks

The SIFT notebook expects images and manually created masks in Google Drive:

```text
/content/drive/My Drive/horse/
/content/drive/My Drive/horse_mask/
```

Provide these files or update the paths to your own directories. The notebook explicitly references `horse001_mask.png`. Images are resized to 400 × 295 pixels; masks must align with the corresponding resized images for the foreground evaluation to be meaningful.

### MNIST

The RBM notebook downloads compressed MNIST image and label files using URLs defined in its setup cells, and stores them in the working directory as:

```text
train_imgs.gz
train_labels.gz
test_imgs.gz
test_labels.gz
```

Internet access is needed for the initial download. If a configured URL is unavailable, update it or place the corresponding MNIST IDX gzip file under the expected local filename.

The model export cell uses an absolute path on the author's machine, while the following load cell reads `rbm_model.json` from the working directory. Change both cells to use the same writable path before running them.

### UCI Air Quality

Both air-quality notebooks contain code to download the UCI Air Quality archive. Set `local_path` in the first code cell to an existing writable directory. The supplied paths point to the author's local machine or `/content/drive/MyDrive/ISPR/` in Colab.

The loader uses `air_quality.zip`, reads `AirQualityUCI.csv`, and caches data as `air_quality.csv`. Preprocessing selects the `PT08.*` sensor columns and the benzene target `C6H6(GT)`, removes rows with missing dates or benzene values, and scales the features and target. The value `-200` is treated as missing for the benzene target.

The xLSTM training code writes and reloads `checkpoint.pt` in the working directory.

## Results and execution notes

The notebooks preserve exploratory results and their limitations:

- **SIFT clustering:** the reported experiments struggle to separate horses from the background using descriptor clustering alone.
- **RBM features:** the notebook reports strong SVC performance relative to several other classifiers and discusses the effect of hyperparameters.
- **Benzene prediction:** the recurrent-model experiments report poor predictive performance and discuss possible implementation, model, and tuning issues.

Some cells need adjustments before a fresh run:

- In the SIFT notebook, a `plot_des` call passes `memssage='Image'`; change the keyword to `message='Image'`.
- In the RBM notebook, import `train_test_split` before its first use; its import currently appears later in the classification section.
- The air-quality loader expects a date column named `,,,,,Date` and uses different CSV settings when saving and reading its cache. Check the parsed columns and make the date-column name and CSV settings consistent with your input file.
- The air-quality experiments scale data before splitting and use shuffled splits, including for next-value experiments. Review these choices before interpreting the results as forecasting performance.

Saved outputs document earlier runs; they do not guarantee that every notebook will execute unchanged in a new environment. RBM and recurrent-model training can take substantial time.
