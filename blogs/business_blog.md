# Introduction
This blog post is the initial entry in a series discussing a computer vision project, conducted jointly by the Oregon Department of Fish and Wildlife (ODFW), The Environmental Defense Fund, and CVision AI. This first entry is the Executive Summary of the project intended for a business audience. For a more technical overview, including the code used on the project, please reference the lengthier article found here. 

Leveraging the Tator platform, we detect boats in ports across the United States. A challenge that not only has significant economic implications but also bears great importance for the environment. This post provides a high-level overview of the problem and how we employed the Tator Platform to resolve it. We've utilized video data sourced directly from ports for this project, with specific port details withheld for privacy reasons.

Ports are vital to assessing fishing activity in an area and understand the impact these activities have on the environment. Each of the ports presents a unique set of data challenges, requiring a thoughtful and adaptable approach to model training and implementation. Their complex nature, coupled with the vast array of boats that dock each day, presents a significant problem for both logistics and policy. Traditional solutions, such as manual monitoring or rudimentary automated systems, often fall short, being unable to keep pace with the constant flux or provide the nuanced understanding required. Even off-the-shelf AI solutions often falter under less than perfect conditions such as fog and nighttime scenarios. By contrast, our solution, uniquely tailored to handle a variety of environmental conditions, stands resilient solving the problem in both an automated and accurate manner.

![Sample Detection: Clear](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/Clear_it1.png?raw=true)
##### Example Detection: Clear

Harnessing the power of the Tator platform for video storage and analytics, we integrated Yolov5, a state-of-the-art deep learning model known for its accuracy and efficiency in object detection tasks. Using the port dataset, we trained our model to not only recognize a wide array of boats but also to adapt to differing port environments and withstand challenging weather conditions. The interplay of Yolov5 and Tator enables us to capitalize on robust analytics, thus improving operational efficiency and redefining standards in port management. Join us as we delve deeper into how we accomplished this task.


# Our Solution
Given that off the shelf solutions fail in less than ideal weather conditions our thought was that fine tuning a model on specific weather conditions for Clear, Fog, and Night would produce better results. Drawing on this initial hypothesis we embarked on a systematic approach towards actualizing this theory. We employed Tator for video analytics, a platform instrumental for efficient data organization and labeling. Further, we utilized Yolov5 Small as our foundation model for transfer learning, and developed three distinctive models, each tailored to one of the observed weather conditions.

## **Procedure**

Our approach to problem resolution was systematically divided into four crucial steps:

### **Step 1: Data Segmentation**

The preliminary phase categorizes the port data based on weather conditions: Clear, Fog, and Night. Given the considerable variance in these environmental circumstances, it was important to guarantee that each dataset precisely mirrored the conditions it was designed to represent. The Tator platform, with its well-documented API and user-friendly web interface, facilitated efficient data bucketing. A notable challenge was addressing ambiguities arising from mixed weather conditions. However, strategic use of Tator's web interface and saved searches significantly mitigated this issue, leading to a drastic reduction in the dataset review duration.

![Saved Searches](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/saved_search.png?raw=true) 
##### Saved Searches in Tator

### **Step 2: Training and Model Validation**

Subsequently, we embarked on the training of three independent Yolov5 Small models, each fine-tuned for a specific weather condition. To ensure the trained models' reliability before deployment, validation was a critical aspect of this phase. Our exploration was extended to a variety of architectures and hyperparameters during the training phase to maximize each model's performance. The primary challenge lay in identifying the most promising architectures to evaluate. The industry-standard Yolov5 API served as an invaluable resource, simplifying the exploration of various hyperparameters. For more details see the more technical write up.

###  **Step 3: Inference and Fine-tuning**

Post the training and validation phases, the models were run on unseen data for inference. This step provided us with a practical scenario to assess our models' performance and uncover any performance gaps. Initial tests exhibited issues with false positives. To address this, we incorporated negative samples into our training dataset and fine-tuned our models based on these real-world observations.

![Bulk Corrections](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/bulk_corrections.png?raw=true)
##### Bulk Corrections in Tator

### **Step 4: Continuous Retraining**

The final step in our procedure was the implementation of a continuous training loop. This enabled our models to progressively refine their learning from previous iterations and continually enhance their detection capabilities. This cyclical approach allowed us to make consistent adjustments and improvements, evolving to counter new challenges as they surfaced. This step mostly involves increasing the size and variety of the data used to train the model. Or put simply, more ports with more boats. As more the model grew, so did its understanding of our problem and its ability to adapt to any condition. Eventually our model began to even be able to detect boats in some of the worst conditions. 

![Night Detection](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/night_it2.png?raw=true)
##### Night Detection 

By adhering to this comprehensive strategy, we managed to establish a highly effective and accurate boat detection system that consistently performed under varying weather conditions. This achievement sets a new benchmark for AI-based solutions in the realm of port management. 

# Conclusion

This collaborative project between the Oregon Department of Fish and Wildlife, The Environmental Defense Fund, and CVision AI has shown the potential of artificial intelligence for managing ports and monitoring the environmental impact of fishing activities. Utilizing the Tator platform in conjunction with a Yolov5 model, we've constructed a system capable of accurately detecting boats in diverse weather conditions and complex port environments.

Our methodology involved careful data segmentation based on weather conditions, rigorous model training and validation, and an iterative fine-tuning process. This allowed us to overcome the limitations commonly found with pre-existing solutions and build models that could adapt to specific weather conditions.

As this executive summary concludes, we invite you to consider a simple question: can you spot this boat?

![Fog Detection](https://github.com/Adam-Ansari-USF/Computer_Vision_Pipeline/blob/0f14cb88b3d7ccffbe034a58a8ad8970351ec5c5/article_photos/fog_it2.png?raw=true)
##### Fog Detection

