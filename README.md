Our data storage: [THI Sharepoint](https://thide-my.sharepoint.com/:f:/r/personal/mib1213_thi_de/Documents/DataFestGermany2025?csf=1&web=1&e=lsVGZK)

Note: You  will be needing a THI account to access this folder.

Our discussion steps to be found in [strategy.md](./strategy.md).

### How to clone this git repo?

- Open the terminal
- Go to the folder where you want to have this repository. (Don't create a folder yourself because git will create it itself)
- Run the following command:
```bash
git clone https://github.com/mib1213/DataFestGermany2025.git
```
- Now you should see the folder named `DataFestGermany2025`.
- Now go ahead and create a virtual env in this folder using conda.

### How to install conda?

- Download the miniforge installer for Windows x86_64 from [here](https://conda-forge.org/download/)
- Run the installer as Administrator.
- Open the **Miniforge Prompt** as Administrator.
- Run the command `conda init` to add the conda command in PATH variable.
- Run the following two commands to set the conda-forge channel as default:
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```
- Run the command `conda config --show channels`. You should see `conda-forge` as the ONLY channel.

**Now the conda is installed and can be used directly from the terminal!**

- Open the terminal (cmd)
- Run `conda`. You should now see the manual for conda.

### How to make a venv using conda?

- Go to the folder where you have the `environment.yml` file, if you have cloned the 'DataFestGermany2025` repository, it should be in that folder.
- Once in the same folder, run the command `conda env create -f environment.yml -y`.
- Once the environment has been created, run `conda activate datafest`. !!Make sure you were already not in any venv/conda environment by running `conda deactivate` for conda or run `deactivate` for venv!!
- Now you should see the env name at the beginning of the command prompt like `(datafest)`, if that is the case, you are in the environment.
- You can run a cmd `conda list` to see all the available dependencies. You can download more dependencies by using `conda install DEPENDENCY_NAME`, but please let me know first, since it can create dependency issues.
- You can see all the currently available conda environments by running `conda env list`.


### How to use this newly created env in VS Code?
- First make sure to open the correct folder in VS Code, in our case it should be `DataFestGermany2025` folder.
- Create a .ipynb file and open it.
- You should now see the option of "Select kernel" at the top right corner.
- Click on this option, select "Select Another kernel" -> "Python Environments" -> "datafest (Python 3.10.*)".
- If you want to use a native .py file, once opened, you need to press CTRL + SHIFT + P, then type "Select Interpreter" and then again select the same Python 3.10.* (datafest) environment.

### How to delete an env?

- First make sure you are outside of the environment, you want to delete, by running `conda deactivate`.
- Once outside, run the command `conda env remove -n ENVIRONMENT_NAME`.
- Optionally, if you want to remove by path, you can run `conda env remove -p ENVIRONMENT_PATH`.

*Once the env is deleted, you should be able to create another env using the same steps.*

### How to add unimportant/private files in .gitignore? 
For example you want to ignore a file named "test.ipynb":
- Using Terminal:
    - echo "test.ipynb" >> .gitignore
- Using Data Explorer:
    - Open the .gitignore file and add the "test.ipynb" without quotes

### How to create your own separate branch to do edits in the same repository?
If you want to make your own edits or just want to play out with code, it is recommended to create your own branch, you will then also be able to push it to this remote repo so others can also see it (optionally).

- First look into the current branch name, you are working in. Run `git branch`. You should now see the `main` as the currently active branch.
- To create a new branch, run `git branch YOUR_BRANCH_NAME`
- Now to switch to the newly created branch, run `git switch YOUR_BRANCH_NAME`
- Now check, if the new branch is active, run `git branch` again and now you should see 2 branches, one is `main` and the other is your newly created branch `YOUR_BRANCH_NAME`. But the new branch should be active this time.

### How to push the branch into github repository?
Now if you want to push (publish) your branch into the remote repository, you can follow the following steps:

### How to create an issue for problems/suggestions on the github repository?

- Go to the this link [Issues](https://github.com/mib1213/DataFestGermany2025/issues)
- Create a new issue and select an Assignee if relevant.