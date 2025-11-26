### EEG Inspired Art Generation  

The purpose of this project is to generate artistic images by analysing EEG data and mapping brain states to symbolic representations. These mappings are then integrated into a prompt for a Stable Diffusion model that generates the final output.
![WhatsApp Image 2025-11-01 at 15 49 16_4ae07cae](https://github.com/user-attachments/assets/b4bc856c-9947-4192-95e4-f406660f06a8)

Steps to run the code and generate images:  
1. Run `spectogram_analysis.py` to extract spectograms from the dataset  
2. Run `mapping.py` to analyze the spectograms, map them to symbolic elements (colours, atmpshere, emotions) and create prompts. A `eeg_prompts.json` file will be created containing them.  
3. Run `generate_images_from_prompts.py` which loads the Stable Diffusion model, gives the prompts as input and generates the final images.  

In order to evaluate the new images, there is an additional step:  
4. Run `evaluation.py` which assigns each prompt-image pair a CLIP similarity and a colour match score.  


The EEG dataset used can be obtained from: [Kaggle](https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset)  

# Other output examples: 
![WhatsApp Image 2025-11-01 at 15 49 14_613fcc7f](https://github.com/user-attachments/assets/67d260d5-d3f1-43cb-876f-567990d5d8d1)  
![WhatsApp Image 2025-11-01 at 15 49 15_a155793b](https://github.com/user-attachments/assets/ca0eb55a-fe3d-4e90-b713-b4326d683810)
