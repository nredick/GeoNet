# GeoNet
GeoNet: User-friendly tool for processing geospatial data with U-Net

# Installation

If you are using Windows, ensure you are running these commands in either the Anaconda Prompt or the Python Command Prompt. On Mac or Linux, you can run these commands in the terminal.

## Clone the repository

```
git clone https://github.com/nredick/GeoNet.git
```

Alternatively, you can download the repository as a zip file and extract it using the green Code button. You can also open the repository in GitHub Desktop.

### Navigate to the repository

```
cd path/to/GeoNet
```
## Check your Python version

Ensure that you are running Python 3.6 or higher. You can check your Python version by running:
```
python3 --version 
```

> *You can download the latest python version [here](https://www.python.org/downloads/) or by using a software manager like [Homebrew](https://docs.brew.sh/Installation) (Mac/Linux) or [Chocolately](https://chocolatey.org/why-chocolatey) (Windows).*

## Create a virtual environment (optional)

It is recommended to create a virtual environment to install GeoNet. This will ensure that the dependencies for GeoNet do not interfere with other Python packages you may have installed. You can create a virtual environment using the steps below. 

*Replace `myenv` with whatever you want to call your virtual environment.*

### Using Python venv
```
python3 -m venv myenv
source myenv/bin/activate
```
### Using virtualenv 
```
pip3 install virtualenv
virtualenv myenv

myenv\Scripts\activate # Windows
source myenv/bin/activate # Mac/Linux
```

### Using Anaconda
*Alternatively, you can use Anaconda to create a virtual environment. You can download Anaconda [here](https://www.anaconda.com/products/distribution) or install it with a software manager.*
>This is the recommended method for Windows users.

```
conda update conda
conda create -n myenv python=x.x anaconda
source activate myenv
```

### Activate your virtual environment
Then, activate the virtual environment by running:

```
source myenv/bin/activate
```

Next, install the required packages by running the following command.
```
pip3 install -r requirements.txt
```

# Usage
