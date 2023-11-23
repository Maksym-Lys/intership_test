# Task 2. Computer vision. Sentinel-2 image matching

- Link to task data in Google Drive: https://drive.google.com/drive/folders/1YQX_7o9mRmvRewT-9NPjFEDCLL5slzZT?usp=sharing

### Dataset preparation:
For dataset generation, I log in to [Sentinel Hub](https://www.sentinel-hub.com/) and configure requests for extracting satellite 
images in the desired season and location. I download images for each season in 10 locations, 
totaling 40 images. Each image has a size of 2500x2500 pixels, so I split each image into 100 
subimages sized 250x250.

To introduce some location shift between images for better model generalization, I randomly 
slice each 250x250 subimage into 4 images sized 224x224. Then, I allocate images from the 
first 8 blocks of 2500x2500 images for the training set. Image block-9 is allocated to the 
development set, and image block-10 to the testing set, respectively, in order to avoid data 
leakage.

The training set consists of 800 different locations, with 4 seasons per location and 4 images 
per season with shift, totaling 12,800 images.

The development set and testing set each consist of 100 different locations, with 4 seasons per 
location and 4 images per same season with shift, totaling 1,600 images for each dev and test sets.

### Data loader preparation and model training

To generate triplets, I wrote a function that iterates over all image paths in the dataset and assigns 
each as an anchor. For the positive image, it looks for image paths from the same location but from a 
different season to stimulate the model's search for inter-seasonal similarity. For the negative image, 
it chooses an image from the same season but with a different location within the same 2500x2500 block, 
in order to make these images hard to distinguish. Then, I receive triplet tuples that contain the anchor 
image path, positive image path, and negative image path.

To load triplet tensors into the model, I wrote a preprocessing function and split the data with a batch 
size of 32 triplets per batch. After that, I defined the Siamese model architecture and created a training 
loop with automatic model checkpointing and early stopping.

### Model inference
For model inference, I altered the test data to remove all images from the same location and season but 
with slight location shifts. These images were helpful for model training with triplets because they 
could introduce more variability compared to data augmentation. Thus, they should be removed to avoid 
receiving overly optimistic inference metrics.

For each image, I create a dictionary with its corresponding representation in the embedding space and 
write a function to calculate the most similar embeddings. Using this function, I calculate the 
mean Average Precision and Top-N accuracy. The Siamese model achieves these results:

Mean Average Precision (mAP):
Top-1: 42.00%
Top-2: 39.75%
Top-3: 35.83%

Top-N Accuracy:
Top-1: 42.00%
Top-3: 64.50%
Top-10: 84.00%
Top-20: 91.75%

### Model demo
The model demo operates with the same test data used in model inference but has a cell with an interface 
where you can choose a random image and the number of images with the most similar embeddings. 
This will plot the random image alongside the corresponding most similar embeddings. 
Above these predictions, SIFT was applied for easier similarity recognition.


