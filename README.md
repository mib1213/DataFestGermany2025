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

- Go to the folder where you want to create an env using the terminal. (In our case it would be `DataFestGermany2025` folder)
- Run the command `conda create -p ./env python=3.12.9`
- Once the env is created, run `conda activate ./env`. This should activate the env which can be seen at the front of the CLI in parantheses.
- Confirm if you see the correct python version by running `python --version`. It should be 3.12.9 exactly.
- Now install the dependencies, for example:
```bash
conda install pandas numpy seaborn matplotlib ipykernel missingno Jinja2 folium
```
- Similarly further packages can be installed by running `conda install package_name`

### How to use thie newly created env in VS Code?
- First make sure to open the correct folder in VS Code, in our case it should be `DataFestGermany2025` folder.
- Create a .ipynb file and open it.
- You should now see the option of "Select kernel" at the top right corner.
- Click on this option, select "Other kernels" -> "Python environments" -> "env\python.exe". (Usually it is also the "recommended" enviroment, if the correct folder is opened in VS Code.)

*Note: The versions of Python and dependencies are not confirmed yet. So most likely you will have to delete this environment later and create a new one.*

### How to delete an env?

- Just delete the folder named `env` from DataFestGermany2025

*Once the env is deleted, you should be able to create another env using the same steps.*

### How to add unimportant/private files in .gitignore? 
(coming soon...)

### How to create your own separate branch to do edits in the same repository?
(coming soon...)

### How to push the branch into github repository?
(coming soon...)

### How to create an issue for problems/suggestions on the github repository?

- Go to the this link [Issues](https://github.com/mib1213/DataFestGermany2025/issues)
- Create a new issue and select an Assignee if relevant.

### How to run LAPD notebooks?

- Download the files `Crime_Data_2020_to_Present.csv` and `mocodes.csv` from [THI Sharepoint](https://thide-my.sharepoint.com/:f:/r/personal/mib1213_thi_de/Documents/DataFestGermany2025?csf=1&web=1&e=lsVGZK).
- Save these files in the `data` folder inside the `DataFestGermany2025`.
- Once you run the `1_data_clearning.ipynb`, you will get the cleaned data in 2 different formats as a .pkl file, one is `crime_data.pkl` and the other is `exploded_crime_data.pkl`. This .pkl file can easily be imported in Pandas for data analysis as it will keep the records of datatypes.
- `ydata_profiling.ipynb` would not work for now, due to dependencies issue but you should be able to see the already generated html report in `ydata_profiling_report.html`.
- `utils.py` contains some helper functions.