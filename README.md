# Emergent PBR Texture Generation

This repository generates the  EmergenTexture dataset, by using Agentic LLM pipeline to suggest/collect/invent and implement different models and simulations from different fields of science and art and convert them to generate PBR materials textures.

The main script for dataset generation is [Create_Textures_Models_Code.py](/media/fogbrain/6TB/EmergenTexture_Full_Code/Final_PBR_Making_Code/Create_Textures_Models_Code.py). 

## What the project does

The pipeline has three stages:

1. Ask an LLM for new scientific, mathematical, models and simulations that create spatial patterns and can be use to generate textures and PBR maps.
2. Ask an LLM to implement each idea suggested at 1 as a `generate.py` texture generator that use this model to generate PBR texture.
3. Run the generated code, validate the output maps, and keep state so interrupted runs can resume.

Generated datasets are written under `output_pbrs/<topic>/`. Each topic directory keeps its own dataset state and contains one subdirectory per generated method.

### For the full generated dataset see: [EmergenTextures Dataset](https://github.com/sagieppel/EmergenTexture-dataset-of-PBR-materials-generated-by-simulations-models-and-simulations-)

![](/samples3.jpg)

## Setup

### Requirements:

- Python 3
- LLM API preferably  openrouter and API key
- Blender API (optional for rendering the PBR on object)

#### Usage 
Set API key at API_KEYS.py
Run Create_Textures_Models_Code.py

## Main Script

[Create_Textures_Models_Code.py](/media/fogbrain/6TB/EmergenTexture_Full_Code/Final_PBR_Making_Code/Create_Textures_Models_Code.py) is the core dataset builder.

Its `generate_dataset(...)` function:

- Loads the prompt files for one topic.
- Resumes existing dataset state from `data.pkl` if present.
- Asks an idea model for new methods.
- Asks a coding model to implement each method as Python.
- Runs and validates the generated code with `code_tester.py` and `check_pbr_errors.py`.
- Stores success or failure back into the dataset so later runs can continue from where they stopped.

### Recommended usage

Use the function directly from Python so you can control topic, model, and retry count explicitly.

Example:

```python
from Create_Textures_Models_Code import generate_dataset

generate_dataset(
    dataset_dir="output_pbrs/pbr_math",
    query_dir="queries_prompts/pbr_math",
    number_of_new=5,
    number_of_code_fix_retry=0,
    idea_model="openai/gpt-5.5",
    code_model="openai/gpt-5.5",
    check_model="openai/gpt-5.5",
)
```
Note the pipeline can use different models for suggesting implementing and checking ideas.
It recommended not to suggest more than 10 ideas per round so not to confuse the LLM.
Note that when adding ideas to folder (dataset_dir) the model will read previously implemented ideas and will not repeat them.
Aka this can be run in loop to generate unlimited number of original methods that will be added to data_dir.

### Important parameters

- `dataset_dir`: output folder for one topic, for example `output_pbrs/pbr_math`
- `query_dir`: prompt folder for that topic, for example `queries_prompts/pbr_math`
- `number_of_new`: how many new methods/simulations for PBR generating to ask the idea model for
- `number_of_code_fix_retry`: how many repair rounds to allow after failed validation
- `idea_model`: model used to propose texture-generation methods
- `code_model`: model used to write generator code
- `check_model`: model used to repair or improve generated code
- `model`: fallback model if role-specific models are not provided

### What happens when it runs

- New benchmark ideas are merged into the dataset state.
- Each benchmark is turned into a `generate.py` implementation.
- The code is executed into a `textures/` directory.
- Validation checks file names, map presence, image size, and sample structure.
- Failed entries are marked so long runs do not repeat the same broken work forever.

### Running the file directly

Running [Create_Textures_Models_Code.py](/media/fogbrain/6TB/EmergenTexture_Full_Code/Final_PBR_Making_Code/Create_Textures_Models_Code.py) as a script starts a long random loop across all topic folders under `queries_prompts/`. 
This was used to generate the dataset.
Running:

```bash
python Create_Textures_Models_Code.py
```
As is should be enough to replicate the EmergenTexture dataset.

By default that loop:

- writes into `output_pbrs/`
- samples from topics such as `pbr_complex_original`, `pbr_social_science`, `pbr_biology`, `pbr_phsyics_engineering`, `pbr_creative`, and `pbr_math`
- randomly picks idea and coding models from hardcoded lists in the file

### Query folders structure
# Prompts/Query
Prompt templates that guide the generation live under `queries_prompts/<topic>/`
Each topic guide the generation into different set of models (aka, physics, chemistry, art...)

## Batch Generator Runner

[run_all_generators.py](/media/fogbrain/6TB/EmergenTexture_Full_Code/Final_PBR_Making_Code/run_all_generators.py)  Will rerun all generator scripts created in 'Create_Textures_Models_Code.py' but with different set of parameters (for example different resolution and number of samples)

```bash
python run_all_generators.py
```


## Blender Preview Renderer

[Render_PBR_Sphere.py](/media/fogbrain/6TB/EmergenTexture_Full_Code/Final_PBR_Making_Code/Render_PBR_Sphere.py) renders generated maps onto a sphere or imported object using Blender Cycles.

Short version:

- `render_single_pbr_sphere(...)` renders one PBR sample on a UV sphere
- `render_single_pbr_object(...)` renders one PBR sample on an imported OBJ or GLTF/GLB mesh
- running the file directly batch-renders one chosen sample from each generated model directory

Batch render command:

```bash
python Render_PBR_Sphere.py
```
Will go over all generated PBR and will render them on sphere (using blender API).
The image will be saved in the PBR main folder.

