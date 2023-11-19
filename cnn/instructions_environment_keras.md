Creating a virtual environment in Python and installing TensorFlow in it is a great way to manage dependencies and avoid conflicts with other packages. Here's a step-by-step guide on how to do this on Windows, which seems to be your operating system based on the file paths in your previous message:

1. **Open Command Prompt**: 
   - You can do this by searching for `cmd` in the Windows search bar and clicking on the Command Prompt app.

2. **Create a Virtual Environment**:
   - Navigate to the directory where you want to create your virtual environment using the `cd` command. For example:
     ```cmd
     cd path\to\your\project\directory
     ```
   - Create a new virtual environment named `keras_env` (you can choose any name you prefer) by running:
     ```cmd
     python -m venv keras_env
     ```
     This command will create a new folder named `keras_env` in your current directory, containing the virtual environment.

3. **Activate the Virtual Environment**:
   - Once the virtual environment is created, you need to activate it. On Windows, run:
     ```cmd
     .\keras_env\Scripts\activate
     ```
     You'll know the virtual environment is activated because its name will appear in parentheses at the beginning of the command prompt line.

4. **Install TensorFlow**:
   - With the virtual environment activated, install TensorFlow by running:
     ```cmd
     pip install tensorflow
     ```
     This command installs TensorFlow in your virtual environment, not globally. 

5. **Verify the Installation**:
   - After installation, you can verify it by running a Python shell and importing TensorFlow:
     ```cmd
     python
     >>> import tensorflow as tf
     >>> print(tf.__version__)
     ```
     This will print the version of TensorFlow, confirming its installation.

6. **Deactivating the Virtual Environment**:
   - When you're done working in the virtual environment, you can deactivate it by simply running:
     ```cmd
     deactivate
     ```
     in the Command Prompt.

Remember, each time you want to work on your project, you'll need to activate the virtual environment using the `.\keras_env\Scripts\activate` command. This ensures that you're using the Python interpreter and packages installed in your virtual environment, rather than the global Python installation.

Don't forget to install all the import, you can do this with ```pip install ...```