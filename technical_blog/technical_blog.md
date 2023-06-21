# Introduction
This blog post is the second entry in a series discussing a computer vision project, conducted jointly by the Oregon Department of Fish and Wildlife (ODFW), The Environmental Defense Fund, and CVision AI. [The first blog](www.tator.io/blog/edf-odfw-blog) in the series covered the problem at a high level, detection of boats in ports leveraging computer vision. The previous blog avoids as much jargon and actual coding examples as possible, focusing instead on the general steps, procedures, and outcomes of the project. This blog does the opposite. For each aspect of the computer vision lifecycle we will cover the intuition behind the task, the methodology needed to accomplish the task, and any outputs/metrics associated. The full version of the code used at each step is available on our githubs or can be found [here](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline.git). Any pictures of ports used to demonstrate output come from video data sourced with permission, with specific port details withheld for privacy reasons.
## Computer Vision 
Before we delve into the lifecycle, let's establish a foundational understanding of computer vision and the role Convolutional Neural Networks (CNNs) play in our project. Computer Vision, at its core, is a field of artificial intelligence that trains computers to interpret and understand the visual world. In our case, this involves detecting boats in various weather conditions using video data from ports.

CNNs form the backbone of our approach. In simple terms, a CNN is a deep learning algorithm that takes in an input image, assigns importance to various aspects or objects in the image, and is able to differentiate one from the other. The preprocessing required is much lower compared to other classification algorithms, as CNNs are designed to automatically and adaptively learn spatial hierarchies of features from the raw input image data. The inputs to our model are images from video data of ports, and the output is the detection of boats in these images. Given this explanation any time the word “model” is used in future sections refer to a CNN.

Object Detection is a supervised machine learning task. As such accurate labels are required to train the model. Labeling in a computer vision project implies bounding the object of interest in a box as tightly around the image as possible. More data generally means a better model, but the principle of garbage in, garbage out still applies. If the data is not accurately labeled or if the labels are incorrect then the model will learn noise and affect the final model detections.  

###### Example Label of a Small Fishery Boat
![Example Label of a Small Fishery Boat](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/fog_it3.png?raw=true)


In this project, we employ the YOLOv5 model, a variant of the YOLO (You Only Look Once) series, renowned for their real-time object detection capabilities. YOLOv5 is particularly lauded for its efficiency and high speed, making it an excellent choice for tasks such as ours, where real-time accuracy is of utmost importance.

###### Source: Ultralytics https://github.com/ultralytics/yolov5/releases
![Yolo comparisons](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/yolo_compare.png?raw=true)

Now, having set the stage with an understanding of Computer Vision, CNNs, and the YOLOv5 model, let's delve into the heart of our project, the Computer Vision Lifecycle.
## The Computer Vision Lifecycle 

As covered in our previous blog post, we incorporated a four-step method to address the task of detecting boats in ports. This initial breakdown was done to assist with explaining the business problem and how we went about solving it. For the more technical breakdown we will be examining the same problem but from a more technical three set of steps: Data Sorting, Training and Validating the Models, and Self-Supervised Learning. The steps contain all the same ideas as the original four but with the “Inference and Fine-tuning” and “Continuous Training” sections condensed into the new Self-Supervised Learning one. For each step we have created a series of python files which can be called via the command line. This abstracts away the need to understand how the code works at a detailed level, allowing for anyone with a basic understanding of programming and computer vision to work through any computer vision task using Tator, Yolov5, and Python.  
### Step 1: Data Sorting
#### Intuition
Our computer vision model needs the video frames metadata, which we refer to as localizations, in order to train the model. The model also needs the associated bounding boxes around the boats in each localization which we refer to as labels. The essence of this step is to categorize data according to different weather conditions - Clear, Fog, and Night. It lays the groundwork for training specific models tailored to these conditions, thereby ensuring the reliability of results under each weather circumstance. 

#### Methodology 
The process is broken down into three parts: one to fetch the localizations from Tator, one to download the image to local disk, and lastly one to create the labels associated with the localizations and convert them to the proper format.

##### Fetch Localizations
Fetching localizations from Tator requires an API Token. The token allows access to the api which makes referencing specific media and pulling out the associated localizations easy. This pull grabs the metadata associated with each localization and allows for quick sorting into a series of training, validation, and test sets needed for model development. Each set of localizations has both positive and negative frames. The positive frames contain boats with their labels and the negative frames contain background. The addition of negative images is an important inclusion to increasing model performance discussed more in the next section. From the github the python file to run from the command line is called `saveLocalization.py`. One of the outputs of this code is a YAML which is referenced by the python file used in the next step.

###### Fetch Localizations Python Script
![saveLocs](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/saveLoc.png?raw=true)

##### Download Images
Now that we have acquired the metadata for the localizations and sorted them into different sets we can download the actual images locally. The process involves using the localizations metadata to uniquely identify the target frames needed to train the model. The Tator API then takes this information and downloads the images locally. Yolov5 expects a specific structure for the images to be stored in. Without this structure Yolov5 will be unable to train or validate any model, so it is important that this step creates the proper folder structure for both the images and labels (created in the next step) to be stored in. From the github the python file to run from the command line is called `locsImgCoco.py`. The YAML here is the same one created by the `saveLocalization.py` code from the previous part.

###### Download Images Python Script
![locsImgCoco](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/locsImgCoco.png?raw=true)

##### Create Labels
The last part of our data sorting procedure is to create the labels associated with the positive localizations and store them locally in the folders generated in the previous steps. Leveraging the metadata acquired from the first step we then convert the coordinates of the label into a format known as COCO. The COCO format allows for easy manipulation and control over the label itself, and functions as a starting point for referencing the labels during training. As mentioned earlier Yolo expects a specific structure for its images and labels. This structure is referenced in a YAML file which is passed to the YOLO API. In order to create this YAML file we must convert the COCO format to the one YOLO expects. This final step is trivial since the previous steps laid all the groundwork by creating the proper structure needed for this part. From the github the python file to run from the command line is called `coco_to_yolo.py`. 

###### Create Labels Python Script
![coco_to_yolo](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/coco_to_yolo.png?raw=true)

#### Outputs/Metrics
The final output is a YAML file which references the proper folder structure that YOLO expects during training. This is where the actual images and associated labels are stored in various folders locally. Pulling the localizations is a quick process but as the number of images being saved grows so does the run time of saving those images locally. Typically all the steps of the Data Sorting process take no longer than a minute with the exception of the second step, Download Images, which can take hours depending on the dataset size.
### Step 2: Training and Validating the Models
#### Intuition
This stage involves training distinct models for each weather condition, and validating their effectiveness. The models learn from the specific characteristics of each weather condition and validate their knowledge against a validation and test set related to the corresponding conditions. We also validate an ensemble model which combines the weights of each weather condition together in one model. The trade offs and advantages of each model will be discussed now.   
#### Methodology 

In order to train and validate each model we reference the YAML files created in the Data Sorting step to quickly train models using the YOLO API. This step is broken down into three parts: train base models for each condition, tune the hyperparameters associated with each condition, and finally test these models on unseen data. 

 
##### Train Models
Using the YAML files and the YOLO API we train baseline models for each weather condition. Rather than starting from scratch YOLO offers several foundation models as starting points for transfer learning. There are many options but we chose the YOLOv5 Small network which contains 25 layers and was trained on ImageNet with eighty classes. Using the weights contained in the small networks allows for quicker convergence and more accurate predictions with a smaller dataset. YOLO by default is a multi-class classification model, but a single argument to the api can constrain the number of classes. The number of classes must match the classes contained in the YAML file created in the Data Sorting step. In many ways we are not training a model in this step so much as fine tuning. Fine tuning a model is typically preferred since training a model from scratch can cost significantly more and often produces worse results. 

###### Example Training Run
![training run](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/yolo_train_run.png?raw=true)



##### Hyperparameter Tuning
Now that we have trained a series of baseline models for each weather condition we can work on tuning the hyperparameters associated with each model. The validation set is used in this step for model selection. The YOLO api makes this easy. The first parameter of interest is the number of epochs the network does before finalizing the weights. An epoch is one full pass through the data at which point the weights are updated. The more epochs the more time the network has to fit the weights of the model to the training dataset. Too many epochs generally means the model begins to overfit. 

###### Example Training Script
![Yolo Train Run](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/77e2956320d45fde763cb344699f912a9781275f/article_photos/yolo_api_code.png?raw=true)

After evaluating multiple different values we ended up at twenty epochs. The YOLO architecture takes the best weights across all epochs as the final output for the model (evaluated against the validation set). In general we found twenty was enough for the model to converge. There is also an option in YOLO called “evolve” which uses genetic algorithms to find the best hyperparameters for the given dataset. This is helpful with a truly large dataset but in our case we did not see much gain from this approach. We opted instead to use the suggested hyperparameters for the other parameters. YOLO has already done this tuning process on the ImageNet dataset and we found using those optimal parameters worked for our dataset as well. 

##### Model Validation
This part is mainly to validate the various models trained in the previous steps on unseen data. This represents the test set created in the Data Sorting step. The more our models are overfitting the worse we expect the models to perform on the test set versus the validation set. We can use the test set as feedback for what we need to change about our models as we move into the Self-Supervised Learning step. 
      
#### Outputs/Metrics
The three primary metrics of interest are Precision (P), Recall (R), and Mean Absolute Precision (mAP). There is also F1 Score which is the harmonic mean between Precision and Recall. Precision is a measure of the number of predicted positives correctly classified by the model. All else constant when comparing two models, if the Precision of one model is higher we can say that model has less false positives than the other. This means that the model is predicting less background as boats. Maximizing Precision is important as doing so can help eliminate alarm fatigue when the model predicts rocks, birds, or other backgrounds as boats.

Recall on the other hand is the measure of the number of actual positives correctly classified by the model. The worse Recall is the more the model missed boats in the images. This is because the formula for Recall includes false negatives. In our case a false negative is a boat that the model is not detecting (treating as background). Obviously the goal of the model is to identify boats so Recall is one of the most important metrics evaluated. Various post processing methods not discussed in this blog can be used to eliminate false positives measured in Precision, but those methods cannot detect a false negative. As such Recall can be seen as the more important of the two metrics.   

Finally, we also measured mAP. The metric mAP measures the tightness by which the model creates the bounding boxes for the predicted boats. The worse mAP is the more background that is captured in the label. Ideally a model would capture only a boat with as little background as possible in it. This becomes especially important in the next step when we use the predicted labels to retrain the model. Below is an example comparing one of our Clear Models to the current Production models. Clear in this case refers to a CNN trained solely on clear weather data. The results speak for themselves. More details on the specifics in the next section.

###### Example Metrics for our Clear versus Current Production model
| Data set | Model | P       | R       | F1 Score | MAP     |
|----------|-------|---------|---------|----------|---------|
| val      | Clear | 0.902   | 0.935   | 0.918    | 0.958   |
| Test     | Clear | 0.909   | 0.933   | 0.921    | 0.961   |
| val      | Prod  | 0.806   | 0.661   | 0.726    | 0.646   |
| Test     | Prod  | 0.82    | 0.667   | 0.736    | 0.676   |

### Step 3: Self-Supervised Learning 

#### Intuition
Self-supervised learning is when a model trained in a supervised fashion is used to generate labels which are then used in the retraining process. This method of iterative detection serves as an extra step during model evaluation, checkpointing the progress and growth of the model. The primary purpose of self-supervised learning, however, is to vastly grow the dataset that the model is trained on by generating new labels in mass. The downside of this is that the accuracy of the new detections is dependent on the base model, which during the early stages, may not be very accurate. As a result manual corrections to the models generated labels is necessary to ensure optimal performance during retraining. The next parts will describe the various checkpoints or iterations we conducted and the results of each. 

#### Methodology
##### Iteration 1
The first iteration of our weather segmented models was created based on a small sample of subject matter expert labeled images. These annotations served as a foundation for our early model development and form the baseline for all future models. Where the annotations were highly accurate their number was few. Across all three weather conditions there was only about 150 each. Not enough to create a model which generalized very well, but enough to get started with the semi-supervised process. For the below table each model was trained on and tested on data of its own weather type (with the exception of the ensemble model). This is to say the Clear model was trained on clear data and validated using different clear data. The ensemble model stands alone in that it was tested on the validation sets of all three weather conditions. 

###### Iteration 1 Results

|    Iteration 1      |              |           |                 | All Data  |
|------------------|-------|------|--------|-------------|
|    Validation      | Clear | Fog   | Night | Ensemble |
| Total    | 1     | 1     | 1     | 1        |
| TP       | 0.87  | 0.91  | 0.87  | 0.88     |
| FP       | 1     | 1     | 1     | 1        |
| TN       | 0.12  | 0     | 0     | 0        |
| FN       | 0     | 0.09  | 0.13  | 0.12     |
| Precision| 0.605 | 0.935 | 0.884 | 0.644    |
| Recall   | 0.95  | 0.904 | 0.842 | 0.827    |
| Accuracy | 0.99  | 0.91  | 0.87  | 0.88     |
| mAP      | 0.659 | 0.967 | 0.878 | 0.814    |

At first glance our model performance could be seen as being quite good across most segments. However based on the extremely small dataset used we were not confident that the model would generalize well to a larger dataset. This turned out to be true, but at the same time the general trends we witnessed here stuck across all iterations. In general a model trained on its individual weather condition will outperform an ensemble model by a significant margin. The promising results from the Fog and Night models also suggest that our hypothesis of breaking out models into individual weather types does indeed solve the existing shortcomings of out of the box training on a single dataset. 

##### Iteration 2
Iteration 2 is the first self-supervised learning part. Using the models trained in the previous part we conduct inference on a larger dataset from a different port. This dataset was ten times the size of the previous. As mentioned earlier the problem with this method is that if the accuracy of the original model is not great then manual corrections must be done. This process is time consuming and took several days to complete. The correction process involves two points: labeling the false positives as negatives and where possible labeling false negatives as true positives. Labeling the false negatives (instances of boats detected as background) is significantly more time consuming. The Tator platform does provide an interface which makes this significantly easier, but without weeks of time labeling all of the false negatives is not possible. That said we did label what we could find within a reasonable time span and went on to retraining the model with our new corrected dataset. 

Labeling the false positives comes with an additional boon. Considering that a false positive in this case is a rock, bird, or other background object that the model is falsely detecting as a boat, we can use that information to feed the model negative samples. A negative sample is an image included in the training set which contains no labels. Including negative samples of the background that the model is falsely detecting can boost model performance. There is one trade off however. The more negative samples given the less positives the model is trained on. In general terms this tends to translate to less false positives but more false negatives. In practice we observed that the more negative samples we passed the more “dropout” that occurred. Dropout here just means when the model is accurately predicting a boat across an entire segment of a video but will seemingly randomly stop predicting correctly for several frames. The base recommendations for positive to negative samples is ten percent. However in our case we decided to increase that ratio to fifteen percent due to the poor precision scores we observed at first. With all that said the below is the final result of our second iteration given all these considerations.

###### Iteration 2 Results 
| P     | R     | F1 Score | Production F1 | MAP   | Data set | Model    |
| ----- | ----- | -------- | ------------- | ----- | -------- | -------- |
| 0.917 | 0.955 | 0.936    | 0.608         | 0.962 | val      | Night    |
| 0.905 | 0.952 | 0.928    | 0.599         | 0.964 | Test     | Night    |
| 0.772 | 0.772 | 0.772    | 0.586         | 0.822 | val      | Fog      |
| 0.691 | 0.631 | 0.660    | 0.489         | 0.710 | Test     | Fog      |
| 0.902 | 0.935 | 0.918    | 0.726         | 0.958 | val      | Clear    |
| 0.909 | 0.933 | 0.921    | 0.736         | 0.961 | Test     | Clear    |
| 0.811 | 0.811 | 0.810    | 0.640         | 0.865 | val      | Ensemble |
| 0.824 | 0.794 | 0.805    | 0.608         | 0.827 | Test     | Ensemble |

Again where the ensemble model was validated against all the weather types, but each weather specific model was only validated against their own weather type. We have also added a Production F1 Score column to check the precision and recall of the current production model versus our weather segmented approach. The results from the previous iteration still hold but the actual model performance for fog has fallen. We believe with more data some of these issues could be resolved, but the largest factor here is that the foggy data is difficult to label even manually. With the dense fog it is almost impossible for a human to label whether the image contains a boat, so the manual correction process alone is not sufficient to solve the downsides of self-supervised learning. This said our model still outperforms the existing production model but additional steps still need to be taken to improve the performance of the model. Having completed an iteration of this size we were comfortable in concluding that weather segmented models can indeed solve the problem of fog and night data in image detection of boats. 
##### Iteration 3
Since we have already come to a conclusion regarding our initial hypothesis this final iteration was done to test how much additional effort would be needed to bring the model up to speed given a completely new and different port. In short, we were interested in testing how long the data science portion of the customer onboarding process would take given a new port with new data. For this experiment we took a new port and conducted the self-supervised process on a small sample of approximately 100 images each per weather category. In addition we tested freezing layers during training to see how quickly we could conduct the retraining process. Freezing a layer refers to holding all the weights constant during the training process. This means no new weights are learned during backpropagation for the frozen layers. In general with smaller datasets freezing layers is recommended for any deep learning model since the early layers tend to learn primitive features such as edges and colors which are general to all computer vision tasks. 

###### Iteration 3 Results
| P     | R     | F1 Score | MAP   | Data set | Model  | Freeze   |
| ----- | ----- | -------- | ----- | -------- | ------ | -------- |
| 0.916 | 0.938 | 0.927    | 0.962 | val      | Clear  | Head     |
| 0.834 | 0.814 | 0.824    | 0.919 | Test     | Clear  | Head     |
| 0.89  | 0.239 | 0.377    | 0.333 | val      | Fog    | Head     |
| 0.461 | 0.248 | 0.323    | 0.328 | Test     | Fog    | Head     |
| 0.636 | 0.668 | 0.652    | 0.638 | val      | Fog    | Backbone |
| 0.724 | 0.683 | 0.703    | 0.747 | Test     | Fog    | Backbone |
| 0.839 | 0.761 | 0.798    | 0.852 | val      | Fog    | None     |
 | 0.823 | 0.808 | 0.815    | 0.802 | Test     | Fog    | None     |

In this new port we found there was almost no night traffic so we could not train a night model. For the Freeze column Head refers to freezing up to the 24th layer such that only the head of the model is retrained. Backbone means freezing up to the 10th layer such that only the primitive features are left alone and all the other layers are retrained. The general finding from this iteration is that with only a small dataset our model generated in the previous step can achieve similar results while freezing almost all the layers. Fog requires additional effort but similar outcomes can be obtained none the less by unfreezing more layers. 


## Conclusion

In conclusion, our exploration into creating a robust, weather-segmented boat detection model has yielded significant and promising results. 

Our initial hypothesis, which proposed that a weather-segmented model would outperform a "one-size-fits-all" model in diverse weather conditions, has held up well through our experiments. Models trained on specific weather conditions (clear, fog, and night) outperformed the current production model significantly in most cases. This confirms our suspicion that the 'out-of-the-box' single model approach struggled with variability in weather conditions. Ensembling the three weather segmented models also saw a significant boost in performance across all weather segments when compared to the production model as well. 

Throughout this process, we implemented self-supervised learning, expanding our training dataset by generating new labels from the model. This iterative detection process was used to not only track the progress of our models but also to significantly grow the data they were trained on. Despite the challenges of potential inaccuracies in the early stages, manual adjustments to the model's generated labels allowed us to maintain a high level of performance during retraining.

One of the key insights gleaned from this project was the importance of adequate and representative data. Although our initial models performed well, they did not generalize adequately to larger datasets due to the limited training samples. This was especially apparent when dealing with difficult weather conditions, like fog, where even manual labeling proved challenging. 

Additionally, even when testing on customer-specific environments during our third iteration our models still held up with minimal retraining. By testing our models on data from a new port, we examined how our models could adapt to an entirely different setting with unique traffic patterns and weather conditions. While the adaptation required some effort, we found that by utilizing strategies such as layer freezing during retraining, the models could still deliver comparable performance.

Finally, our findings demonstrate the value of weather segmentation and self-supervised learning in improving boat detection models. This approach not only addresses the limitations of using a single model to handle variable weather conditions, but it also provides a pathway for continual model improvement through iterative training and label generation. As we continue to fine-tune this process, we anticipate that our models will become increasingly accurate and adaptable, ultimately providing significant value for port management and maritime traffic monitoring.

As a reminder this article is a technical write up of an executive summary that can be found [here](www.tator.io/blog/edf-odfw-blog).

