---
title: "Using a GitHub Personal Access Token (PAT) to Push/Pull from a SageMaker Notebook"
teaching: 25
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I securely push/pull code to and from GitHub within a SageMaker notebook?
- What steps are necessary to set up a GitHub PAT for authentication in SageMaker?
- How can I convert notebooks to `.py` files and ignore `.ipynb` files in version control?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Configure Git in a SageMaker notebook to use a GitHub Personal Access Token (PAT) for HTTPS-based authentication.
- Securely handle credentials in a notebook environment using `getpass`.
- Convert `.ipynb` files to `.py` files for better version control practices in collaborative projects.

::::::::::::::::::::::::::::::::::::::::::::::::

# Using a GitHub Personal Access Token (PAT) to Push/Pull from a SageMaker Notebook

When working in SageMaker notebooks, you may often need to push code updates to GitHub repositories. However, SageMaker notebooks are typically launched with temporary instances that don’t persist configurations, including SSH keys, across sessions. This makes HTTPS-based authentication, secured with a GitHub Personal Access Token (PAT), a practical solution. PATs provide flexibility for authentication and enable seamless interaction with both public and private repositories directly from your notebook. 

> **Important Note**: Personal access tokens are powerful credentials that grant specific permissions to your GitHub account. To ensure security, only select the minimum necessary permissions and handle the token carefully.


## Step 1: Generate a Personal Access Token (PAT) on GitHub

1. Go to **Settings > Developer settings > Personal access tokens** on GitHub.
2. Click **Generate new token**, select **Classic**.
3. Give your token a descriptive name (e.g., "SageMaker Access Token") and set an expiration date if desired for added security.
4. **Select the minimum permissions needed**:
   - **For public repositories**: Choose only **`public_repo`**.
   - **For private repositories**: Choose **`repo`** (full control of private repositories).
   - Optional permissions, if needed:
     - **`repo:status`**: Access commit status (if checking status checks).
     - **`workflow`**: Update GitHub Actions workflows (only if working with GitHub Actions).
5. Generate the token and **copy it** (you won’t be able to see it again).

> **Caution**: Treat your PAT like a password. Avoid sharing it or exposing it in your code. Store it securely (e.g., via a password manager like LastPass) and consider rotating it regularly.


## Step 2: Configure Git `user.name` and `user.email`
In your SageMaker or Jupyter notebook environment, run the following commands to set up your Git user information


#### Directory setup
Let's make sure we're starting at the same directory. Cd to the root directory of this instance before going further.


```python
%cd /home/ec2-user/SageMaker/
!pwd
```

    /home/ec2-user/SageMaker
    /home/ec2-user/SageMaker



```python

!git config --global user.name "Chris Endemann"
!git config --global user.email endeman@wisc.edu

```

### Explanation

- **`user.name`**: This is your GitHub username, which will appear in the commit history as the author of the changes.
- **`user.email`**: This should match the email associated with your GitHub account so that commits are properly linked to your profile.

Setting this globally (`--global`) will ensure the configuration persists across all repositories in the environment. If you’re working in a temporary environment, you may need to re-run this configuration after a restart.

## Step 3: Use `getpass` to Prompt for Username and PAT

The `getpass` library allows you to input your GitHub username and PAT without exposing them in the notebook. This approach ensures you’re not hardcoding sensitive information.



```python
import getpass

# Prompt for GitHub username and PAT securely
github_url = 'github.com/UW-Madison-DataScience/test_AWS.git' # found under Code -> Clone -> HTTPS (remote the https:// before the rest of the address)
username = input("GitHub Username: ")
token = getpass.getpass("GitHub Personal Access Token (PAT): ")
```

**Note**: After running, you may want to comment out the above code so that you don't have to enter in your login every time you run your whole notebook


### Explanation

- **`input("GitHub Username: ")`**: Prompts you to enter your GitHub username.
- **`getpass.getpass("GitHub Personal Access Token (PAT): ")`**: Prompts you to securely enter the PAT, keeping it hidden on the screen.



## Step 4: Add, Commit, and Push Changes with Manual Authentication
### 1. Navigate to the Repository Directory (adjust the path if needed):



```python
!pwd
%cd test_AWS
```

    /home/ec2-user/SageMaker
    /home/ec2-user/SageMaker/test_AWS


### 2. Preview changes: You may see elaborate changes if you are tracking ipynb files directly.


```python
!git diff 
```

    nbdiff /tmp/git-blob-PLwmtf/04_Interacting-with-code-repo.ipynb 04_Interacting-with-code-repo.ipynb
    --- /tmp/git-blob-PLwmtf/04_Interacting-with-code-repo.ipynb  2024-11-01 21:19:40.081619
    +++ 04_Interacting-with-code-repo.ipynb  2024-11-01 21:19:30.253573
    [34m[1m## replaced /cells/20/execution_count:[0m
    [31m-  55
    [32m+  79
    
    [0m[34m[1m## inserted before /cells/20/outputs/0:[0m
    [32m+  output:
    [32m+    output_type: stream
    [32m+    name: stdout
    [32m+    text:
    [32m+      [main bc28ce1] Updates from Jupyter notebooks
    [32m+       1 file changed, 875 insertions(+), 56 deletions(-)
    
    [0m[34m[1m## deleted /cells/20/outputs/0:[0m
    [31m-  output:
    [31m-    output_type: stream
    [31m-    name: stdout
    [31m-    text:
    [31m-      [main 0363cc2] Added updates from Jupyter notebook
    [31m-       7 files changed, 416 insertions(+), 91 deletions(-)
    [31m-       delete mode 100644 00_Data-storage-and-access-via-buckets.ipynb
    [31m-       create mode 100644 01_Setting-up-S3-bucket.md
    [31m-       create mode 100644 02_Setting-up-notebook-environment.md
    [31m-       create mode 100644 03_Data-storage-and-access-via-buckets.ipynb
    [31m-       rename push-git-updates.ipynb => 04_Interacting-with-code-repo.ipynb (77%)
    [31m-       rename 01_Intro-train-models.ipynb => 05_Intro-train-models.ipynb (100%)
    [31m-       rename 02_Hyperparameter-tuning.ipynb => 06_Hyperparameter-tuning.ipynb (100%)
    
    [0m[34m[1m## replaced /cells/22/execution_count:[0m
    [31m-  56
    [32m+  80
    
    [0m[34m[1m## modified /cells/22/outputs/0/text:[0m
    [36m@@ -1,4 +1,4 @@[m
     From https://github.com/UW-Madison-DataScience/test_AWS[m
      * branch            main       -> FETCH_HEAD[m
    [31m-   adfe7b1..637d64c  main       -> origin/main[m
    [32m+[m[32m   637d64c..0363cc2  main       -> origin/main[m
     Already up to date.[m
    
    [0m[34m[1m## replaced /cells/26/execution_count:[0m
    [31m-  57
    [32m+  81
    
    [0m[34m[1m## modified /cells/26/outputs/0/text:[0m
    [36m@@ -1,9 +1,9 @@[m
    [31m-Enumerating objects: 7, done.[m
    [31m-Counting objects: 100% (7/7), done.[m
    [32m+[m[32mEnumerating objects: 5, done.[m
    [32m+[m[32mCounting objects: 100% (5/5), done.[m
     Delta compression using up to 2 threads[m
    [31m-Compressing objects: 100% (6/6), done.[m
    [31m-Writing objects: 100% (6/6), 11.22 KiB | 5.61 MiB/s, done.[m
    [31m-Total 6 (delta 1), reused 0 (delta 0), pack-reused 0[m
    [32m+[m[32mCompressing objects: 100% (3/3), done.[m
    [32m+[m[32mWriting objects: 100% (3/3), 10.51 KiB | 5.25 MiB/s, done.[m
    [32m+[m[32mTotal 3 (delta 1), reused 0 (delta 0), pack-reused 0[m
     remote: Resolving deltas: 100% (1/1), completed with 1 local object.[K[m
     To https://github.com/UW-Madison-DataScience/test_AWS.git[m
    [31m-   637d64c..0363cc2  main -> main[m
    [32m+[m[32m   0363cc2..bc28ce1  main -> main[m
    
    [0m[34m[1m## inserted before /cells/29:[0m
    [32m+  code cell:
    [32m+    id: f86a6d55-5279-423f-864a-7810dd414def
    [32m+    source:
    [32m+      import subprocess
    [32m+      import os
    [32m+      
    [32m+      # List all .py files in the directory
    [32m+      scripts = [f for f in os.listdir() if f.endswith('.py')]
    [32m+      
    [32m+      # Convert each .py file to .ipynb using jupytext
    [32m+      for script in scripts:
    [32m+          output_file = script.replace('.py', '.ipynb')
    [32m+          subprocess.run(["jupytext", "--to", "notebook", script, "--output", output_file])
    [32m+          print(f"Converted {script} to {output_file}")
    
    [0m

### 3. Convert json ipynb files to .py

To avoid tracking ipynb files directly, which are formatted as json, we may want to convert our notebook to .py first (plain text). This will make it easier to see our code edits across commits. Otherwise, each small edit will have massive changes associated with it.

#### Benefits of converting to `.py` before Committing

- **Cleaner Version Control**: `.py` files have cleaner diffs and are easier to review and merge in Git.
- **Script Compatibility**: Python files are more compatible with other environments and can run easily from the command line.
- **Reduced Repository Size**: `.py` files are generally lighter than `.ipynb` files since they don’t store outputs or metadata.

Converting notebooks to `.py` files helps streamline the workflow for both collaborative projects and deployments. This approach also maintains code readability and minimizes potential issues with notebook-specific metadata in Git history. Here’s how to convert `.ipynb` files to `.py` in SageMaker without needing to export or download files:

#### Method 1: Using JupyText

1. **Install Jupytext** (if you haven’t already):


```python
!pip install jupytext

```

    Collecting jupytext
      Downloading jupytext-1.16.4-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: markdown-it-py>=1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jupytext) (3.0.0)
    Collecting mdit-py-plugins (from jupytext)
      Downloading mdit_py_plugins-0.4.2-py3-none-any.whl.metadata (2.8 kB)
    Requirement already satisfied: nbformat in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jupytext) (5.10.4)
    Requirement already satisfied: packaging in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jupytext) (21.3)
    Requirement already satisfied: pyyaml in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jupytext) (6.0.2)
    Requirement already satisfied: tomli in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jupytext) (2.0.1)
    Requirement already satisfied: mdurl~=0.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from markdown-it-py>=1.0->jupytext) (0.1.2)
    Requirement already satisfied: fastjsonschema>=2.15 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from nbformat->jupytext) (2.20.0)
    Requirement already satisfied: jsonschema>=2.6 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from nbformat->jupytext) (4.23.0)
    Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from nbformat->jupytext) (5.7.2)
    Requirement already satisfied: traitlets>=5.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from nbformat->jupytext) (5.14.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from packaging->jupytext) (3.1.4)
    Requirement already satisfied: attrs>=22.2.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jsonschema>=2.6->nbformat->jupytext) (23.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jsonschema>=2.6->nbformat->jupytext) (2023.12.1)
    Requirement already satisfied: referencing>=0.28.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jsonschema>=2.6->nbformat->jupytext) (0.35.1)
    Requirement already satisfied: rpds-py>=0.7.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jsonschema>=2.6->nbformat->jupytext) (0.20.0)
    Requirement already satisfied: platformdirs>=2.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from jupyter-core!=5.0.*,>=4.12->nbformat->jupytext) (4.3.6)
    Downloading jupytext-1.16.4-py3-none-any.whl (153 kB)
    Downloading mdit_py_plugins-0.4.2-py3-none-any.whl (55 kB)
    Installing collected packages: mdit-py-plugins, jupytext
    Successfully installed jupytext-1.16.4 mdit-py-plugins-0.4.2


1. **Run the following command** in a notebook cell to convert the current notebook to a `.py` file:

This command will create a `.py` file in the same directory as the notebook.


```python
# Replace 'your_notebook.ipynb' with your actual notebook filename
!jupytext --to py Data-storage-and-access-via-buckets.ipynb
```

    [jupytext] Reading 03_Data-storage-and-access-via-buckets.ipynb in format ipynb
    [jupytext] Updating the timestamp of 03_Data-storage-and-access-via-buckets.py


#### Method 2: Automated Script for Converting All Notebooks in a Directory

If you have multiple notebooks to convert, you can automate the conversion process by running this script, which converts all `.ipynb` files in the current directory to `.py` files:


```python
import subprocess
import os

# List all .ipynb files in the directory
notebooks = [f for f in os.listdir() if f.endswith('.ipynb')]

# Convert each notebook to .py using jupytext
for notebook in notebooks:
    output_file = notebook.replace('.ipynb', '.py')
    subprocess.run(["jupytext", "--to", "py", notebook, "--output", output_file])
    print(f"Converted {notebook} to {output_file}")

```

    [jupytext] Reading 05_Intro-train-models.ipynb in format ipynb
    [jupytext] Updating the timestamp of 05_Intro-train-models.py
    Converted 05_Intro-train-models.ipynb to 05_Intro-train-models.py
    [jupytext] Reading 03_Data-storage-and-access-via-buckets.ipynb in format ipynb
    [jupytext] Updating the timestamp of 03_Data-storage-and-access-via-buckets.py
    Converted 03_Data-storage-and-access-via-buckets.ipynb to 03_Data-storage-and-access-via-buckets.py
    [jupytext] Reading 03_Data-storage-and-access-via-buckets-test.ipynb in format ipynb
    [jupytext] Updating the timestamp of 03_Data-storage-and-access-via-buckets-test.py
    Converted 03_Data-storage-and-access-via-buckets-test.ipynb to 03_Data-storage-and-access-via-buckets-test.py
    [jupytext] Reading 06_Hyperparameter-tuning.ipynb in format ipynb
    [jupytext] Updating the timestamp of 06_Hyperparameter-tuning.py
    Converted 06_Hyperparameter-tuning.ipynb to 06_Hyperparameter-tuning.py
    [jupytext] Reading create_large_data.ipynb in format ipynb
    [jupytext] Updating the timestamp of create_large_data.py
    Converted create_large_data.ipynb to create_large_data.py
    [jupytext] Reading 04_Interacting-with-code-repo.ipynb in format ipynb
    [jupytext] Writing 04_Interacting-with-code-repo.py (destination file replaced)
    Converted 04_Interacting-with-code-repo.ipynb to 04_Interacting-with-code-repo.py


### 4. Adding .ipynb to gitigore

Adding `.ipynb` files to `.gitignore` is a good practice if you plan to only commit `.py` scripts. This will prevent accidental commits of Jupyter Notebook files across all subfolders in the repository.

Here’s how to add `.ipynb` files to `.gitignore` to ignore them project-wide:

1. **Open or Create the `.gitignore` File**:

    ```python
    !ls -a # check for existing .gitignore file
    ```
    
   - If you don’t already have a `.gitignore` file in the repository root (use '!ls -a' to check, you can create one by running:
   
     ```python
     !touch .gitignore
     ```


2. **Add `.ipynb` Files to `.gitignore`**:

   - Append the following line to your `.gitignore` file to ignore all `.ipynb` files in all folders:

     ```plaintext
     *.ipynb # Ignore all Jupyter Notebook files
     ```

   - You can add this line using a command within your notebook:
   
     ```python
     with open(".gitignore", "a") as gitignore:
         gitignore.write("\n# Ignore all Jupyter Notebook files\n*.ipynb\n")
     ```



3. **Verify and Commit the `.gitignore` File**:

   - Add and commit the updated `.gitignore` file to ensure it’s applied across the repository.

     ```python
     !git add .gitignore
     !git commit -m "Add .ipynb files to .gitignore to ignore notebooks"
     !git push origin main
     ```

This setup will:
- Prevent all `.ipynb` files from being tracked by Git.
- Keep your repository cleaner, containing only `.py` scripts for easier version control and reduced repository size. 

Now any new or existing notebooks won’t show up as untracked files in Git, ensuring your commits stay focused on the converted `.py` files.


2. **Add and Commit Changes**:




```python
!git add . # you may also add files one at a time, for further specificity over the associated commit message
!git commit -m "Updates from Jupyter notebooks" # in general, your commit message should be more specific!

```

    [main f4b268e] Updates from Jupyter notebooks
     10 files changed, 3163 insertions(+), 256 deletions(-)
     delete mode 100644 01_Setting-up-S3-bucket.md
     delete mode 100644 02_Setting-up-notebook-environment.md
     rename 03_Data-storage-and-access-via-buckets.ipynb => Accessing-S3-via-SageMaker-notebooks.ipynb (72%)
     create mode 100644 Accessing-S3-via-SageMaker-notebooks.md
     rename 04_Interacting-with-code-repo.ipynb => Interacting-with-code-repo.ipynb (93%)


3. **Pull the Latest Changes from the Main Branch**: Pull the latest changes from the remote main branch to ensure your local branch is up-to-date.

    Recommended: Set the Pull Strategy for this Repository (Merge by Default)

    All options:

    * Merge (pull.rebase false): Combines the remote changes into your local branch as a merge commit.
    * Rebase (pull.rebase true): Replays your local changes on top of the updated main branch, resulting in a linear history.
    * Fast-forward only (pull.ff only): Only pulls if the local branch can fast-forward to the remote without diverging (no new commits locally).


```python
!git config pull.rebase false # Combines the remote changes into your local branch as a merge commit.

!git pull origin main

```

    remote: Enumerating objects: 8, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (3/3), done.[K
    remote: Total 6 (delta 2), reused 6 (delta 2), pack-reused 0 (from 0)[K
    Unpacking objects: 100% (6/6), 152.14 KiB | 2.67 MiB/s, done.
    From https://github.com/UW-Madison-DataScience/test_AWS
     * branch            main       -> FETCH_HEAD
       1602325..b2a59c3  main       -> origin/main
    hint: Waiting for your editor to close the file... 7[?47h[>4;2m[?1h=[?2004h[?1004h[1;24r[?12h[?12l[22;2t[22;1t[29m[m[H[2J[?25l[24;1H"~/SageMaker/test_AWS/.git/MERGE_MSG" 6L, 300B[2;1H▽[6n[2;1H  [3;1HPzz\[0%m[6n[3;1H           [1;1H[>c]10;?]11;?[1;1H[33mMerge branch 'main' of https://github.com/UW-Madis[mon-DataScience/test_AWS
    [34m# Please enter a commit message to explain why this merge is necessary,[m[2;72H[K[3;1H[34m# especially if it merges an updated upstream into a topic branch.[m[3;67H[K[4;1H[34m#
    # Lines starting with '#' will be ignored, and an empty message aborts
    # the commit.[m
    [1m[34m~                                                                               [8;1H~                                                                               [9;1H~                                                                               [10;1H~                                                                               [11;1H~                                                                               [12;1H~                                                                               [13;1H~                                                                               [14;1H~                                                                               [15;1H~                                                                               [16;1H~                                                                               [17;1H~                                                                               [18;1H~                                                                               [19;1H~                                                                               [20;1H~                                                                               [21;1H~                                                                               [22;1H~                                                                               [23;1H~                                                                               [m[24;63H1,1[11CAll[1;1H[?25h[?4m[?25l[24;1HType  :qa  and press <Enter> to exit Vim[24;41H[K[24;63H1,1[11CAll[1;1H[?25h

If you get merge conflicts, be sure to resolve those before moving forward (e.g., use git checkout -> add -> commit). You can skip the below code if you don't have any conflicts. 


```python
# Keep your local changes in one conflicting file
# !git checkout --ours train_nn.py

# Keep remote version for the other conflicting file
# !git checkout --theirs train_xgboost.py

# # Stage the files to mark the conflicts as resolved
# !git add train_nn.py
# !git add train_xgboost.py

# # Commit the merge result
# !git commit -m "Resolved merge conflicts by keeping local changes"
```

4. **Push Changes and Enter Credentials**:


```python
# Push with embedded credentials from getpass (avoids interactive prompt)
!git push https://{username}:{token}@{github_url} main
```

    fatal: unable to access 'https://{github_url}/': URL rejected: Bad hostname


## Step 5: Pulling .py files and converting back to notebook format

Let's assume you've taken a short break from your work, and you would like to start again by pulling in your code repo. If you'd like to work with notebook files again, you can again use jupytext to convert your `.py` files back to `.ipynb`

This command will create `03_Data-storage-and-access-via-buckets-test.ipynb` in the current directory, converting the Python script to a Jupyter Notebook format. Jupytext handles the conversion gracefully without expecting the `.py` file to be in JSON format.


```python
# Replace 'your_script.py' with your actual filename
!jupytext --to notebook Data-storage-and-access-via-buckets.py --output Data-storage-and-access-via-buckets-test.ipynb

```

    [jupytext] Reading 03_Data-storage-and-access-via-buckets.py in format py
    Traceback (most recent call last):
      File "/home/ec2-user/anaconda3/envs/python3/bin/jupytext", line 8, in <module>
        sys.exit(jupytext())
      File "/home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages/jupytext/cli.py", line 497, in jupytext
        exit_code += jupytext_single_file(nb_file, args, log)
      File "/home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages/jupytext/cli.py", line 561, in jupytext_single_file
        notebook = read(nb_file, fmt=fmt, config=config)
      File "/home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages/jupytext/jupytext.py", line 431, in read
        with open(fp, encoding="utf-8") as stream:
    FileNotFoundError: [Errno 2] No such file or directory: '03_Data-storage-and-access-via-buckets.py'


### Applying to all .py files
To convert all of your .py files to notebooks, you can use the following code:


```python
import subprocess
import os

# List all .py files in the directory
scripts = [f for f in os.listdir() if f.endswith('.py')]

# Convert each .py file to .ipynb using jupytext
for script in scripts:
    output_file = script.replace('.py', '.ipynb')
    subprocess.run(["jupytext", "--to", "notebook", script, "--output", output_file])
    print(f"Converted {script} to {output_file}")

```

    [jupytext] Reading train_xgboost.py in format py
    [jupytext] Writing train_xgboost.ipynb
    Converted train_xgboost.py to train_xgboost.ipynb
    [jupytext] Reading train_nn.py in format py
    [jupytext] Writing train_nn.ipynb
    Converted train_nn.py to train_nn.ipynb



```python
!pwd
!jupyter nbconvert --to markdown Interacting-with-code-repo.ipynb

```

    /home/ec2-user/SageMaker/test_AWS
    [NbConvertApp] Converting notebook Interacting-with-code-repo.ipynb to markdown
    [NbConvertApp] Writing 24319 bytes to Interacting-with-code-repo.md


:::::::::::::::::::::::::::::::::::::: keypoints 

- Use a GitHub PAT for HTTPS-based authentication in temporary SageMaker notebook instances.
- Securely enter sensitive information in notebooks using `getpass`.
- Converting `.ipynb` files to `.py` files helps with cleaner version control and easier review of changes.
- Adding `.ipynb` files to `.gitignore` keeps your repository organized and reduces storage.

::::::::::::::::::::::::::::::::::::::::::::::::
