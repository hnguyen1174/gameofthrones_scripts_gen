## Generate Game of Thrones Script with LSTM

This project generates TV Scripts from a training scripts from Game of Thrones.

### To work on the project:

#### Clone the Repository

To run the pipeline and the app, you will need to clone the repo:

```
# Clone the repo 
$ git clone https://github.com/hnguyen1174/gameofthrones_scripts_gen.git

# update working directory
$ cd gameofthrones_scripts_gen

# Configure git
$ git config --global user.email "you@example.com"
$ git config --global user.name "Your Name"
```

#### Develop Using Colab

To work on this project, I SSH into Google Colab using VS Code through [ssh-colab](https://pypi.org/project/colab-ssh/). 

* Step 1: Register for a free account on [ngrok.com](ngrok.com).
* Step 2: Go to ngrok and copy the [aut token](https://dashboard.ngrok.com/get-started/your-authtoken)
* Step 3: In a destination Google Colab notebook, do the following:

    ```
    # Install colab_ssh
    !pip install colab_ssh --upgrade


    # Password is a password of your choice,
    # you will be asked for it later.
    from colab_ssh import launch_ssh
    launch_ssh('YOUR_NGROK_AUTH_TOKEN', 'SOME_PASSWORD')

    # Output for your vs code ssh
    # Host google_colab_ssh
	  # 	 HostName 2.tcp.ngrok.io
	  # 	 User root
	  # 	 Port <something>
    ```
    

#### Installing Miniconda and Using Virtual Environment

```
$ cd <working_folder>
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ conda create -n <env_name> python=3.7
$ conda activate <env_name>

# Install pytorch with CUDA
$ conda install pytorch torchvision cudatoolkit -c pytorch

# Install all requirements
$ pip install -r requirements.txt
```



