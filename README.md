# CHIMERA Challenge Submission

> Team TBG Lab<br>
> Task 3

## Our Approach:
<img width="10012" height="4852" alt="chimera" src="https://github.com/user-attachments/assets/c6122be3-3e90-4bd8-bf64-8e5efe5ee6ab" />

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

> Developers: Madhav Arora, Sumit Kumar, Dhairya Gupta
