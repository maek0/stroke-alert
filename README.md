# Stroke Alert

## Contents
- [Stroke Alert](#stroke-alert)
  - [Dependencies](#dependencies)
    - [Python](#python)
    - [NumPy](#numpy)
    - [mediapipe](#mediapipe)
    - [opencv](#opencv)
  - [Code Information](#code-information)
  - [Instructions to Run](#instructions-to-run)
  - [Github Basics](#github-basics)
    - [Github dependencies](#github-dependencies)
    - [Download the Project](#download-the-project)
    - [Git Commands](#git-commands)
    - [Possible Errors](#possible-errors)


## Dependencies

This project is working under the assumption that the system running the application has installed the following:
- Python 3.9
- NumPy
- mediapipe
- opencv

Installing NumPy, mediapipe, and opencv can be done in any order, but they all depend on the installation of Python 3.9. Instructions to install are shown below:

### Python
Visit https://www.python.org/downloads, navigate to the section for the correct OS (in this case Windows is being used), and install the Python 3.9 64-bit installer. Once downloaded, the installer can be ran to install Python 3.9 on the machine. Make sure to check 'Add Python 3.9 to PATH', or add to PATH variable manually after running. To check that installation was successful, you can open a new terminal session and run `python --version`. If installed successfully, you'll see a message listing the version of Python you have installed. Running the installer also installs pip, Python's package manager.

### NumPy
run the following:
```
pip install numpy
```

### mediapipe
run the following:
```
pip install mediapipe
```

### opencv
run the following:
```
pip install opencv-python
```

## Code Information

Both mediapipe and cv2 (opencv) are imported and referenced in the python file base.py. So trying to run this file with these not installed will result in errors.

## Instructions to Run

If you are viewing the project directory in Visual Studio Code, you just need to click the 'Run Python File' button in the top right corner of the window (looks like a play button). If viewing the project directory in terminal, and all software has been installed, you can simply run `python base.py`. Both should achieve the same results.

## Github Basics

### Github dependencies

To download this project via Github, using Git, you need to have Git installed, using [git](https://git-scm.com/downloads) - choosing your correct operating system (Windows, macOS, etc). Once installed, you should be able to run `git` in your terminal, and get a response of possible commands to run. This means that Git is correctly installed, and you can proceed to download the project.

### Download the Project

This code repository is hosted [here](https://github.com/maek0/stroke-alert). Upon opening the green 'Code' dropdown button, it should display a tab saying 'Clone with HTTPS'. Click the small clipboard icon next to the url. Then in a terminal on your computer run 
```
git clone url_you_copied
```

There should then be a folder that you can open in whatever text editor you choose, that contains all the folders and files that belong to the project.

### Git Commands

- `git branch` - shows the repository branch your code is from
- `git checkout <branch>` - this will switch you to the branch you specify
- `git checkout -b <new branch name>` - this will create and switch you to a new branch with the name you specify
- `git pull` - pulls the latest updates from the branch your code is from
- `git add <files>` - adds the changes made to the specified files. Use `git add .` if you want to add all changes made to the base directory.
- `git commit -m "commit message"` - makes a commit with the added changes. The commit message will just carry a message to specifiy what changes were made.
- `git push` - this will push all new commits to the branch in the remote repository

General order to add you changes to the remote repository:
- `git branch` - make sure you're on the branch you want to change
- `git add .` - add all of your changes
- `git commit -m "message"` - commit your changes with an appropriate commit message
- `git push` - push the changes

### Possible Errors

Note: if when running `git pull` you encounter an error saying you need to commit your changes, this means that you have made changes to the local project, while also attempting to get the most up-to-date version of the code. to resolve this, run: 
```
git add .
git commit -m "whatever commit message you want"
git pull
```

The `git add .` just adds all changes made in you current folder. The `git commit` just says you're officially committing these changes on your local machine, with the corresponding message you wrote. Doing this allows you to then pull changes from the repository, since Git knows you have accounted for your changes.

*Be aware that you if you make changes locally, and then pull from the repository, your local version may not run as expected, depending on the changes you make. One fix would be to run `git reset`. If it's still running incorrectly, you could always try recloning the repository*
