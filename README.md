# Instructions for use #
The user should build the provided conda environment using, while in the root directory of the project.
```
conda env create --name project_ds_2 --file=environment.yml
```
Once completed, all of the relevant dependencies are avaialbe to the user, meaning that:
* The data can be read and altered
* The models can be retrained and evaluated
* The pipeline can be extended and run  

Or other modifications if necessary.  

# Additional information #

## Folder structure ##

There are several subfolders in the repository:

* the source folder (`/src`),
* the journal folder (`/journal`),
* the interim report folder (`/interim_report`),
* the final report folder (`/final_report`),
* and the presentation folder (`/presentation`).

### The source subfolderfolder structure ###
In the `/app` folder is an MVP of the pipeline with a UI[^1].  
In order do run it, do the following from the root directory:
* `cd src/app`
* `uvicorn main:app --reload`
* Open [localhost](http://127.0.0.1:8000/docs) in your browser of choice.
* Click on the test _\\{model}\\{question}_ and then _Try it out_.
* Possible model options are "generative" and "extractive"


In the `/data` folder, all of the original and processed data can be found.


In the `/evaluation` folder, all of the scripts used to evaluate both standalone models and pipelines are located.


In the `/fine-tuning` folder, all of the scripts found for fine-tuning the models are.


After fine-tuning a `/models` folder will appear aswell. However, if the user doesn't want to that, the model names can be replaced by the corresponding version found [here](https://huggingface.co/SpeedaRJ).


The `/pipe` folder contains an example of the build pipeline.


`/question_generation` contains the scripts used to turn _pdf_ files into question-context-answer pairs, and their subsequent postprocessing.


And finally the `/tools` folder contains some additional functions and methods used.

## Warning ##
To use any scripts that requires a **FAISSDocumentStore** a _db_, _json_ and _faiss_ files should be in the same folder as the scripts. For all of the scripts in this repository this is provided for.

[^1]: An updated version of this is present on the `develop` branch. It has been added post official submission deadline.
