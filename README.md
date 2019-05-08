## Creating the Input Directory
The input directory should have as many subdirectories as there are categories of data. For the example dataset organized by media, we called the root directory `data` and included five subdirectories, named by the type of image they contained: `drawings`, `engraving`, `iconography`, `painting`, and `sculpture`.

## Authenticating the Google Vision API
After following the instructions online to create a Google Cloud Platform project using the Google Vision API, download a key credential file, and run the following terminal command from within the root directory of the project:
```
export GOOGLE_APPLICATION_CREDENTIALS=key.json
```
This command assumes the key file is named `key.json` and is contained within the root directory of the project, but the command can be adjusted as necessary if the key file is named differently or located within a different directory.

## Running the Pipeline
For ease of use, a bash file has been provided that will run both parts of the pipeline in a single command:
```
bash run.sh <input_directory> <label_file> <output_file>
```
All parameters are optional. `<input_directory>` will default to `data`, `<label_file>` will default to `labels.tsv`, and `<output_file>` will default to `analysis.tsv`. For more fine-grained control over the pipeline, continue reading below.

### Running the Label Code
To aid in more efficient processing, there are two scripts. The first, `labels.py`, is used to generate the feature labels and interface with the Google Vision API. These features are saved to a file since feature generation is computationally costly. The script can be run as follows:
```
python3 labels.py <input_directory> <output_file>
```
The script will print updates after every 100 images are processed, but this can be changed by adjusting the `verbose_frequency` parameter at the top of the `labels.py` file.

### Running the Analysis Code
The analysis code depends on the output of the label code explained above. It performs two main functions:

1. Calculates the 10 most important labels for each category. By "most important", we mean that a simple sklearn `LogisticRegression` classifier has deemed these 10 features to have the greatest weights when trained on in the input dataset. The accuracy of the model is printed to the console, as is the top 10 labels with the greatest weights.

The accuracy is computed by running 5-fold cross-validation on the training set, but the number of folds can be changed by adjusting the `fold_count` parameter at the top of the `analysis.py` file. The same is true for the number of features to show: adjust `n` at the top of `analysis.py`.

2. Creates a label frequency overview file. The rows are all of the labels applied to the dataset and the columns are the different categories. Each cell is the number of times that label was applied to an image in that category.

The full analysis script can be run as follows:
```
python3 analysis.py <input_file> <output_file>
```