## This project mainly focus on image retrieval with keras.

### A image retrieval demo implemented with keras based on VGG16 Convolution Neural Networks.

### Steps:
> 1. Extract  image (in the database data/training_images) features using the VGG16 CNN model.
> 2. Store the features in H5PY file.
> 3. Load the features then using the VGG16 model retrieval the image in the data/retrieval_images 
> 4. Using dot product compute the similarity between two images, more similar has higher score.
