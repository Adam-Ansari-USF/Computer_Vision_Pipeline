# Introduction
This blog post is the initial entry in a series discussing a computer vision project, conducted jointly by the Oregon Department of Fish and Wildlife (ODFW), The Environmental Defense Fund, and CVision AI. This first entry is the Executive Summary of the project intended for a business audience. For a more technical overview, including the code used on the project, please reference the lengthier article found [here](https://medium.com/@adam.9001.ansari/computer-vision-for-small-scale-fisheries-e59b236588a4). Spearheading this project are the two authors of this blog: Adam Ansari and Varun Hande. Varun Hande is a data science graduate student from the University of San Francisco. He is a Machine Learning Research Scientist at the Environmental Defense Fund, where he works on improving the SmartPass project in collaboration with CVision AI. Adam Ansari has a similar background as both a Data Science graduate student at the University of San Francisco and a Machine Learning Research Scientist at the Environmental Defense Fund. Together they formed the two Data Scientists on the team who pushed the science on this project to the level demonstrated in this blog.  
 
Leveraging the Tator platform, we detect boats in ports across the United States. A challenge that not only has significant economic implications but also bears great importance for the environment. This post provides a high-level overview of the problem and how we employed the Tator Platform to resolve it. We've utilized video data sourced directly from ports for this project, with specific port details withheld for privacy reasons.
 
Ports are vital to assessing fishing activity in an area and understand the impact these activities have on the environment. Each of the ports presents a unique set of data challenges, requiring a thoughtful and adaptable approach to model training and implementation. Their complex nature, coupled with the vast array of boats that dock each day, presents a significant problem for both logistics and policy. Traditional solutions, such as manual monitoring or rudimentary automated systems, often fall short, being unable to keep pace with the constant flux or provide the nuanced understanding required. Even off-the-shelf AI solutions often falter under less than perfect conditions such as fog and nighttime scenarios. By contrast, our solution, uniquely tailored to handle a variety of environmental conditions, stands resilient solving the problem in both an automated and accurate manner.
 
##### Example Detection of a Boat in Clear Weather
![Sample Detection: Clear](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/Clear_it1.png?raw=true)
 
Harnessing the power of the Tator platform for video storage and analytics, we integrated Yolov5, a state-of-the-art deep learning model known for its accuracy and efficiency in object detection tasks. Using the port dataset, we trained our model to not only recognize a wide array of boats but also to adapt to differing port environments and withstand challenging weather conditions. The interplay of Yolov5 and Tator enables us to capitalize on robust analytics, thus improving operational efficiency and redefining standards in port management. Join us as we delve deeper into how we accomplished this task.
 
 
# Our Solution
Given that off the shelf solutions fail in less than ideal weather conditions our thought was that fine tuning a model on specific weather conditions for Clear, Fog, and Night would produce better results. Drawing on this initial hypothesis we embarked on a systematic approach towards actualizing this theory. We employed Tator for video analytics, a platform instrumental for efficient data organization and labeling. Further, we utilized Yolov5 Small as our foundation model for transfer learning, and developed three distinctive models, each tailored to one of the observed weather conditions.
 
## **Procedure**
 
Our approach to problem resolution was systematically divided into four crucial steps:
 
### **Step 1: Data Segmentation**
 
The preliminary phase categorizes the port data based on weather conditions: Clear, Fog, and Night. Given the considerable variance in these environmental circumstances, it was important to guarantee that each dataset precisely mirrored the conditions it was designed to represent. The Tator platform, with its well-documented API and user-friendly web interface, facilitated efficient data bucketing. A notable challenge was addressing ambiguities arising from mixed weather conditions. However, strategic use of Tator's web interface and saved searches significantly mitigated this issue, leading to a drastic reduction in the dataset review duration.
 
##### Tator: Saved Searches for Data Sorting
![Saved Searches](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/saved_search.png?raw=true)  
 
### **Step 2: Training and Model Validation**
 
Subsequently, we embarked on the training of three independent Yolov5 Small models, each fine-tuned for a specific weather condition. To ensure the trained models' reliability before deployment, validation was a critical aspect of this phase. Our exploration was extended to a variety of architectures and hyperparameters during the training phase to maximize each model's performance. The primary challenge lay in identifying the most promising architectures to evaluate. The industry-standard Yolov5 API served as an invaluable resource, simplifying the exploration of various hyperparameters. For more details see the more technical write up.
 
###  **Step 3: Inference and Fine-tuning**
 
Post the training and validation phases, the models were run on unseen data for inference. This step provided us with a practical scenario to assess our models' performance and uncover any performance gaps. Initial tests exhibited issues with false positives. To address this, we incorporated negative samples into our training dataset and fine-tuned our models based on these real-world observations.
 
##### Tator: Bulk Corrections for Boosting Model Performance  
![Bulk Corrections](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/bulk_corrections.png?raw=true)
 
### **Step 4: Continuous Retraining**
 
The final step in our procedure was the implementation of a continuous training loop. This enabled our models to progressively refine their learning from previous iterations and continually enhance their detection capabilities. This cyclical approach allowed us to make consistent adjustments and improvements, evolving to counter new challenges as they surfaced. This step mostly involves increasing the size and variety of the data used to train the model. Or put simply, more ports with more boats. As more the model grew, so did its understanding of our problem and its ability to adapt to any condition. Eventually our model began to even be able to detect boats in some of the worst conditions.  
 
##### Final Outcomes: Night Detection  
![Night Detection](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/night_it2.png?raw=true)
 
By adhering to this comprehensive strategy, we managed to establish a highly effective and accurate boat detection system that consistently performed under varying weather conditions. This achievement sets a new benchmark for AI-based solutions in the realm of port management.  
# Implementation
The current version of the model is three separate computer vision models one for each observed weather condition: Clear, Night, and Fog. This means that some process must be put into place to determine the condition in the image so it can be sent to the correct model. Sending an image to the wrong model can lead to poor model performance so an implementation strategy is needed to deploy the model in practice. Below are four approaches we examined and their tradeoffs.  
 
## Weather APIs
The first approach we examined was a Weather API. There are several API's available but many focus their efforts in population centers and not ports. Solutions still exist that track time of day, fog, and other weather information in less dense areas, but these tend to be more expensive. At the end of the day the decision to use a weather API for implementation comes down to the cost. If the budget is not a concern then these APIs are by far the simplest and easiest solution.  
 
## Time of Day/Pixel Detection
When the sun sets is not some grand mystery. Many free tools offer the time of day for a given location alongside the sunset which can be used to filter the boat images into the Night model when it is night out. This will separate the images into Night and Day, but not Clear and Fog. To solve this last problem a pixel detection method can be used. In foggy images there are a lot of gray pixels. By examining the number of gray pixels in an image one can quickly assess if an image contains fog. This method may not be perfect but it does get the job done with fairly good accuracy. This method is perhaps the easiest to implement and comes at a low cost compared to some of the other solutions presented in exchange for some loss in model performance.  
 
## Ensemble Method
Perhaps the most technical approach involves combining the weights in all three models into one larger model. This is not the same as training one large model as was done in the production case. The model instead takes the learnings from the individual models and applies that to where it feels those weights are best used. This combination or ensemble method certainly solves the problem but comes at a drop in accuracy. The exact tradeoffs of which are discussed in greater detail in the [technical blog](https://medium.com/@adam.9001.ansari/computer-vision-for-small-scale-fisheries-e59b236588a4).
 
## Building a Classification Model to Predict the Condition
Just like we have built a model to detect boats in an image it is also possible to create a similar model to detect the weather condition. This requires manual labeling of the data by its condition but as discussed in the data sorting segment above this was already accomplished. Training this model is relatively straightforward however the inclusion of a fourth model into the pipeline increases the overall inference time and cost of real time deployment. Deciding to go this route depends on budget and how often a given use case demands predictions.  
 
# Combining the Methods
As a final thought none of these implementation strategies exist in a vacuum. For instance, by ensembling together the Night and Fog Model one can then use the time of day alone to cheaply and quickly send the images to the correct models. We found that the Night and Fog model performed well together so this solution may be preferred over any single one depending on the use case. At the end of the day no one size fits all solution exists. Each use case will have their own preferred solution to this problem based on the cost, performance, and business requirements of the project in question.
 
# Conclusion
 
This collaborative project between the Oregon Department of Fish and Wildlife, The Environmental Defense Fund, and CVision AI has shown the potential of artificial intelligence for managing ports and monitoring the environmental impact of fishing activities. Utilizing the Tator platform in conjunction with a Yolov5 model, we've constructed a system capable of accurately detecting boats in diverse weather conditions and complex port environments.
 
Our methodology involved careful data segmentation based on weather conditions, rigorous model training and validation, and an iterative fine-tuning process. This allowed us to overcome the limitations commonly found with pre-existing solutions and build models that could adapt to specific weather conditions.
 
As this executive summary concludes, we invite you to consider a simple question: can you spot this boat?
 
##### Final Outcomes: Fog Detection
![Fog Detection](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/fog_it2.png?raw=true)
