# CHIMERA Challenge Submission

> Team MSD<br>
> Task 3

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