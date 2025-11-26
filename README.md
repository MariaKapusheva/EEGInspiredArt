### EEG Inspired Art Generation  


Steps to run the code and generate images:  
1. Run `spectogram_analysis.py` to extract spectograms from the dataset  
2. Run `mapping.py` to analyze the spectograms, map them to symbolic elements (colours, atmpshere, emotions) and create prompts. A `eeg_prompts.json` file will be created containing them.  
3. Run `generate_images_from_prompts.py` which loads the Stable Diffusion model, gives the prompts as input and generates the final images.  

In order to evaluate the new images, there is an additional step:  
4. Run `evaluation.py` which assigns each prompt-image pair a CLIP similarity and a colour match score.   

