# CHIMERA Challenge Submission

> Team TBG Lab<br>
> Task 3

## Our Approach:
<img width="10012" height="4852" alt="chimera" src="https://github.com/user-attachments/assets/c6122be3-3e90-4bd8-bf64-8e5efe5ee6ab" />

1. Load WSI and divide into patches of 560x560 of the useful regions, given by the mask.
2. Extract features and coordinates using Virchow2, a foundation model for histopathology images.
3. Create clusters based on agglomerative clustering and make a single node with the mean of all patch-level features in the Cluster.
4. Create a graph of each patient's WSI, where an unweighted edge is made if the distance between the central coordinates of 2 nodes is <= 3000 pixels.
5. Extract embeddings using a custom GCN designed by us.
6. Filter out the top 2000 genes based on the BRS deviation between BRS 1/2 and 3.
7. Concatenate the embeddings of GCN, RNA seq data, and one-hot encoded data to form the final embedding.
8. Finally, train attention MLP, Random Survival Forest, CoxPH, Survival SVM, and Survival Gradient boosted trees, and take a weighted ensemble of the risk scores.

## Data Download

Follow the steps below to download the dataset.

### Step 1: Install AWS CLI

Ensure that you have the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed on your system.
You can find installation instructions for various platforms (Windows, macOS, Linux) in the link above.

### Step 2: Download Dataset

Once AWS CLI is installed, use the following command to sync the dataset from the S3 bucket:

```bash
aws s3 sync --no-sign-request s3://chimera-challenge/v2/task3 .
```

This command will download the dataset to the current directory.
Make sure you are in the desired directory where you want the data to be stored.

**Note:** The `--no-sign-request` flag ensures you can access the dataset without AWS credentials.

## Download dependencies
To set up the environment, make sure you have the following dependencies:
- **Python:** Version 3.10
- **CUDA:** Version 12.6 (ensure that your GPU supports this version)

Once the prerequisites are met, follow these steps to set up your environment:

1. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
2. Activate the virtual environment:
   - On macOS/Linux:
       ```bash
       source .venv/bin/activate
       ```
   - On Windows:
      ```bash
      .venv\Scripts\activate
      ```

3. Install the required dependencies:
    ```bash
    python -m ven .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Visualize the Dataset

To visualize the dataset, you can use the provided Jupyter notebook.
Open a terminal or command prompt, and run the following command to start the Jupyter notebook:

```bash
jupyter notebook notebooks/data_visualization.ipynb
```
## Results

Survival analysis optimization report using 5-fold cross-validation and Optuna-based hyperparameter tuning

##### Summary

    Mean C-index: 0.8211 ± 0.0344
    Best C-index: 0.8668
    Improvement over baseline: +0.1123
    Optimization trials per model: 100

##### Individual Seed Results

    Seed 42: 0.8441
    Seed 121: 0.8243
    Seed 144: 0.7661
    Seed 245: 0.8668
    Seed 1212: 0.8044

##### Key Findings

    Best Performing Models: The optimization successfully improved model performance
    Ensemble Benefits: Optimized ensembles showed consistent improvements
    Parameter Insights: Systematic hyperparameter tuning revealed optimal configurations

##### Recommendations

    Use the optimized hyperparameters for production models
    Consider the ensemble approach for the best performance
    Monitor model stability across different seeds

##### NOTE: 
We were unable to submit our model for the competition due to an error in our Docker implementation. However, we will evaluate the model on the hidden test set once it becomes publicly available.


> Developers: Madhav Arora, Sumit Kumar, Dhairya Gupta
