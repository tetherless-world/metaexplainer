# RPI-HEALS-ExplainabilityImplementation


**Conda cheat sheet**

Install conda from: https://docs.conda.io/en/latest/miniconda.html

- It is better to make sure that we all use the same conda environment, so the below steps can help do so

- Make sure you have cmake installed on your system before you run the next command

- `conda env create -f setup/environment_edited.yml`

    replicates the `ENV_NAME` conda environment

- `conda activate ENV_NAME`

    activates the environment

- `conda env list`

    to list all packages installed in that environment. You need to activate the environment before running this command

- `conda search PackageName`

    searches for `PackageName` in the environment. You need to activate the environment before running this command

 - Run anaconda with zsh: https://stackoverflow.com/questions/31615322/zsh-conda-pip-installs-command-not-found

- To convert a notebook to a Python script, run the below command and replace notebook name in <notebook> argument:
   `jupyter nbconvert --to script <notebook>`

- To use the heals1.0 environment in your jupyter notebook; run the below:\
	`conda install nb_conda`\
        `conda install ipykernel`\
        `python -m ipykernel install --user --name heals1.0`
   You probably wouldn't need the first command if you already have nb_conda installed in the environment where jupyter is running from.

**Package install**
	
- To install pyMetamap, follow instructions at: https://github.com/AnthonyMRios/pymetamap
	
**Git cheat sheet**

- `git clone https://github.com/XXXX`

  clones a repo

- If there a task I need to accomplish, it is usually better to create a branch from master, checkout that branch, modify the code on my local machine, and push the branch to github. Then on github request a `new pull request`. Somoeone else needs to merge the new branch to the master branch. Here are the detailed steps.

    1. `git checkout master` switches to the master branch

    2. `git pull` gets the recent updates

    3. `git checkout -b BRANCH_NAME` creates a new branch

    4. modify the code locally

    5. `git add .` adds the modification to the stageing area of BRANCH_NAME

    6. `git commit -m "nice message"` commits the changes to BRANCH_NAME

    7. `git push` or `git push --set-upstream origin BRANCH_NAME` pushes the changes to github repo

    8. on github, choose the branch then click on `New pull request`. Nice [tuotrial](https://yangsu.github.io/pull-request-tutorial/) about pull request.


- `git branch -a` lists all local branches

- `git branch -d BRANCH_NAME` deletes a branch locally

- `git checkout -- <file>` to discard changes in working directory

- update branch based on master

  - `git checkout BRANCH_NAME`

  - `git merge origin/develop`

  - `git push origin BRANCH_NAME`

- `git reset FILENAME` resets or removes file from git add before commit

- `git restore FILENAME` restores file and discards local changes in the working directory

**UI:**

**Things to know**

- Dependencies and project structure are handled by the package manager for nodejs called `yarn`
  - To install `yarn`, run below:
    - `conda install -c conda-forge nodejs`
      - This will install the latest version of nodejs, and is necessary for the project to build, as the version that comes by default is often very out of date.
      - If it doesn't install the latest version of nodejs - try `conda install nodejs -c conda-forge --repodata-fn=repodata.json` or 
	`conda upgrade -c conda-forge nodejs'
    - `conda install -c conda-forge yarn`
      - This will install yarn

- The dependencies are in the `package.json` as well as the `yarn.lock` files. You should not have to touch them, as these are for the package manager

**How to build UI**

- After installing `nodejs` and `yarn`, change directory to `ui/dash-board`


  - to set up project, run README under `ui/dashboard` directory
	
**Package**

- For Metamap support, follow the below steps:
	- Download pymetamap and copy into eaas/eaascode/utils. Follow instructions from: https://github.com/AnthonyMRios/pymetamap
	- Download metamap 2020 version and install locally using these instructions: https://metamap.nlm.nih.gov/Installation.shtml
