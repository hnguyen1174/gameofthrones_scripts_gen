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

### Sample Scripts generated by LSTM...

```
lannisters and
dany watches him secures the hunting party, and begins to the darkening
forth.
bran
you hear you were.
bran turns and starts up into the hunting party. he
seems preoccupied, gentle as he pulls into his face.
woman’s voice
i’ll be no messenger.
he touches him and begins to cry.
bran
i know you have. he wants...
he touches his finger and murmuring softly, his
strangely strangely checks the gargoyle’s stones.
bran
i know what you want to be here.
robert
(to illyrio)
you are.
ser jorah is pleased. he looks up and begins to viserys: the
loudest, the leaders of a woman’s party.
woman’s voice
(ignoring his guilty hand and
histories of the other king.
khal drogo
no.
he looks up at her hands. he is very fair.
jaime
i have no more.
he sends his horse and looks up at the ground. the
last are hooting in her shoulders. the voices
are hemmed, gentle on their horses. the surface of his fingers and
severed his legs, bran squeezes his sword.
bran
you know what you do. you don’t know
him.
he puts her eyes and looks back.
ext. winterfell- courtyard- day
the king’s party leaps the saddles. the voices of a woman’s hand.
bran
i don’t know.
he looks to his eyes. he begins to remove her hands.
he
no one.
bran
you have to be a visitor in
the previous. they dig his sword.
he looks at her, his fingers deft in the ground, a lean stool. he strokes his eyes widen and
almond. he removes his hand and looks out to his feet. he begins to remove the
evening curtain. he looks back, but she sulks to see
the ledge, the leaders man clamped forth his shoulder.
woman’s voice
i have you been the ones.
bran
no.
she touches her hair, bran does...
```