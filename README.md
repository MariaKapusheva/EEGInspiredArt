# # Evaluation metrics ideas:  
# 1. CLIP similarity = measures how semantically similar the generated image is to its prompt
- was also used in the StageDesigner paper: [link to paper](https://arxiv.org/abs/2503.02595)
- we can use a pre-trained CLIP model from [Kaggle](https://huggingface.co/openai/clip-vit-base-patch32) 

# 2. Colour consistency = checks if the colours from the prompt are predominating the generated image  
- to calculate this we could convert the image to HSV format and calculate the average hue or dominant colour cluster  

# 3. User study for qualitative analysis  
