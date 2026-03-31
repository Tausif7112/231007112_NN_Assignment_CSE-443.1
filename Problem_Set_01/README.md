```markdown
# Pneumonia Detection: A Chest X-Ray Classification Project

## Overview

This project was all about building a image classifier model using CNN(Convolutional Neural Networks) . The main goal is to correctly tell if a chest X-ray shows 'Pneumonia' or  'Normal'. It's super important for helping doctors with early diagnosis. 

## About the Data

The dataset consists of 5,863 JPEG chest X-ray images of kids aged one to five. 

The dataset arranged in the following structure:
  - test
    - NORMAL 
    - PNEUMONIA 
  - train
    - NORMAL
    - PNEUMONIA
  - val 
    - NORMAL
    - PNEUMONIA

The dataset link is attached here: 
[Chest X-Ray Dataset (Google Drive)](https://drive.google.com/file/d/19PoAmtvHWrI3M4lGQAHiNVJx30mqVeUU/view?usp=drive_link)



## Approach

My approach was basically effective model development and robust evaluation:

1.  **Initial Exploration (EDA):** First, I really dug into the dataset to understand its characteristics, including class distribution, image dimensions, and potential issues like class imbalance or small validation sets.
2.  **Preprocessing & Augmentation:** Next, I prepared the images for the CNN by resizing, normalizing pixel values, and applying data augmentation techniques to the training data to make the model more robust.
3.  **Data Re-splitting:** Recognizing the limitations of the original validation set, I re-split the data to create more representative training, validation, and test sets.
4.  **Addressing Imbalance:** To counter class imbalance, I calculated and applied class weights during training.
5.  **Model Development & Improvement:** I started with a baseline CNN model and then iteratively improved it by adding regularization techniques like Batch Normalization and L2 regularization to combat overfitting.
6.  **Evaluation & Tuning:** Finally, I thoroughly evaluated the model's performance using various metrics and fine-tuned the classification threshold to achieve the desired balance between precision and recall.

## Methodology

Here's a detailed breakdown of the steps I followed:

### 1. First Look at the Data (EDA)

Before diving into building the model, I spent some time getting to know the data. This involved:

*   **Counting Images:** using code to count images in each class (Normal/Pneumonia) across the train, val, and test splits.
*   **Visualizing Counts:**  created bar plots to visualize the class distribution and identify imbalances.
*   **Sample Previews:**  displayed a few sample X-ray images to get a qualitative feel for the data.
*   **Image Stats:** I analyzed image widths, heights, and aspect ratios to understand their distributions.

### 2. Getting the Images Ready (Preprocessing)

CNNs like their input images to be a consistent size, so I had to do some prep work:

*   **Standard Size & Batches:** I set all images to be resized to 150x150 pixels (`IMG_HEIGHT = 150`, `IMG_WIDTH = 150`) and grouped them into batches of 32 (`BATCH_SIZE = 32`) for efficient training.
*   **Pixel Scaling:** I used `ImageDataGenerator(rescale=1./255)` to scale pixel values from 0-255 down to 0-1. This helps the model learn faster and more smoothly.
*   **Data Augmentation (for Training):** To make my model more robust and prevent it from just memorizing the training images, I used techniques like `rotation_range=20`, `width_shift_range=0.1`, `height_shift_range=0.1`, `shear_range=0.1`, `zoom_range=0.1`, and `horizontal_flip=True`. This created slightly different versions of the training images. Importantly, I *only* applied this to the training data; validation and test sets were only rescaled.

### 3. Re-splitting the Data (Because the Val Set Was Too Small!)

Since that original validation set was so tiny and hard to trust, I decided to reshuffle things:

*   **Combined Test Set:** I took the original `val` and `test` images and merged them into one big `new_test` set.
*   **New Validation Set:** Then, I grabbed about 20% of the original `train` data to create a proper, larger `validation` set. The rest became my `new_train` set. I used `shutil.copy` to move files and `random.shuffle` to ensure a random split.

### 4. Handling Imbalance with Class Weights

Even after re-splitting, the training data still had more Pneumonia cases. To make sure the model didn't just get good at recognizing Pneumonia and ignore 'Normal' cases, I calculated 'class weights' using `sklearn.utils.class_weight.compute_class_weight`. These weights were then passed to the `model.fit()` method during training, assigning a higher penalty to errors on the 'Normal' (minority) class.

### 5. Building & Improving the Model

#### My First CNN Model (The Baseline)

I started with a pretty standard CNN setup using `tf.keras.models.Sequential`:

*   Input Layer: `tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))`
*   Convolutional Blocks: Multiple `Conv2D` (32, 64, 128 filters with 3x3 kernel, 'relu' activation) followed by `MaxPooling2D` (2x2 pool size).
*   Flatten Layer: `Flatten()` to prepare data for dense layers.
*   Dense Layers: A `Dense(128, activation='relu')` for a hidden layer and `Dropout(0.5)` to help prevent overfitting.
*   Output Layer: A final `Dense(1, activation='sigmoid')` for binary classification.

I compiled it using `optimizer='adam'`, `loss='binary_crossentropy'`, and `metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]`. I used `EarlyStopping(monitor='val_loss', patience=5)` and `ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)`.

#### The Improved CNN Model

To tackle overfitting and make the model more reliable, I made some upgrades:

*   **Batch Normalization:** I added `BatchNormalization` layers after most of the `Conv2D` layers and before the activation in the dense layers. This helps stabilize training.
*   **L2 Regularization:** I also added `kernel_regularizer=regularizers.l2(0.001)` to my convolutional and dense layers to discourage large weights.
*   **Class Weights:** Applied the `class_weights_dict` calculated earlier.
*   **More Patience:** I increased `early_stopping_improved.patience` to 10 epochs for potentially slower convergence due to regularization.

## Findings

Here's what I observed and concluded throughout the project:

### Initial Data Exploration Findings:

*   **Class Imbalance:** Significant class imbalance was present; the training set had almost 3 times more Pneumonia cases, and the test set had 1.6 times more. This suggested the need for techniques to handle imbalance.
*   **Validation Set Issue:** The original validation set was too small (16 images), making validation metrics unreliable and hindering effective early stopping.
*   **Image Consistency:** Most images showed similar resolution and aspect ratios, indicating that a fixed input size (150x150) for the CNN was appropriate.

### Initial CNN Model Performance & Observations:

This first model showed clear signs of **overfitting**:

*   Training accuracy kept rising, but validation accuracy plateaued or dropped.
*   Validation loss began to climb while training loss continued to decrease.
*   Performance on the original test set:
    *   Accuracy: **0.8526**
    *   Precision (Pneumonia): **0.8531**
    *   Recall (Pneumonia): **0.9231**
    *   F1-Score: **0.8867**
    *   AUC: **0.9253**
*   **Confusion Matrix:** 62 False Positives (healthy misdiagnosed as Pneumonia) and 30 False Negatives (Pneumonia cases missed). The good recall was positive, but overfitting was a major concern.

![Initial Overfitting Plot](/content/Initial_overfitting.png)
*Figure 1: Initial Model's Training History showing signs of overfitting.*

### Improved CNN Model Performance & Observations:

After re-splitting data, applying class weights, Batch Normalization, and L2 regularization, the model showed a different performance profile on the *new test set*:

*   Test Loss: **0.6263**
*   Test Accuracy: **0.7750**
*   Precision (Pneumonia): **0.7387**
*   Recall (Pneumonia): **0.9874**
*   F1-Score: **0.8452**
*   AUC: **0.9378**

**Key Trade-offs Observed:**

*   **Fewer Missed Pneumonia Cases (Excellent!):** False Negatives (FN) dramatically reduced from 30 to just **5**. This is crucial in a medical context, as missing a diagnosis can have severe consequences.
*   **More False Alarms (A Trade-off):** False Positives (FP) increased from 62 to **139**. This means the model was more likely to incorrectly label a healthy individual as having Pneumonia. While not ideal, false positives are often preferred over false negatives in initial diagnostic screenings, as they can be clarified with further tests.

Overall accuracy dipped slightly due to the increase in false positives. The combination of class weights and regularization made the model extremely cautious about missing any Pneumonia cases.

![Improved Model Overfitting Plot](/content/last_overfitpng.png)
*Figure 2: Improved Model's Training History showing better stability with regularization and balanced data.*

### Threshold Adjustment Findings:

I found that setting the classification threshold to **0.70** provided a better balance:

*   Accuracy: **0.8266**
*   Precision (Pneumonia): **0.7983**
*   Recall (Pneumonia): **0.9648**
*   F1-Score: **0.8737**

This adjustment significantly reduced false positives (from 139 to 79) while maintaining a very high recall. The "best" threshold ultimately depends on the specific clinical priority – whether minimizing false alarms or ensuring no case is missed is more critical.

## What I Learned & What's Next

This project was a cool journey in building and tweaking a CNN model. The initial model had some overfitting issues, partly due to a small validation set. But by re-splitting the data, adding Batch Normalization and L2 regularization, and using class weights, I got a model that was much better at not missing actual Pneumonia cases. The trade-off was a few more false alarms, but I found a good balance by adjusting the classification threshold.

**If I had more time, I'd totally try these things:**

*   **Transfer Learning:** Using powerful pre-trained models like VGG16 or ResNet50 (which are already good at seeing things in images) could give us a big performance boost.
*   **Smarter Augmentation:** Exploring more advanced ways to augment medical images.
*   **Hyperparameter Tuning:** Systematically playing around with different settings for the model and optimizer to find the absolute best combination.
*   **Explainable AI (XAI):** It would be awesome to understand *why* the model makes certain predictions, especially for medical stuff. This would build a lot more trust!

## How to Run This Project Yourself

Want to see it in action? Here's how:

1.  **Get the Code:** Grab this project from its Git repository.
2.  **Open in Colab:** Upload the `.ipynb` notebook file to Google Colab.
3.  **Run Everything:** Just go through and execute all the code cells from top to bottom. It'll handle all the setup, data download, training, and evaluation for you.
4.  **Check the Results:** Look at the outputs, graphs, and confusion matrices to see how the model performed at each stage.
```
