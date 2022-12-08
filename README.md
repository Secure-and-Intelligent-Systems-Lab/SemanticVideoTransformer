# End-to-End Semantic Video Transformer for Zero-Shot Action Recognition

Code and proposed experimentation setup for the paper [End-to-End Semantic Video Transformer for Zero-Shot Action Recognition](https://arxiv.org/abs/2203.05156). The annotated class descriptions can be found in the Annotations folder. 

## Proposed Fair ZSL Test Setup

We pool the valid test classes from several benchmark datasets to form a novel test set. Altogether, there are 30 unique classes from the UCF-101, HMDB-51, and ActivityNet datasets, as shown in the table below. We handpick each class carefully such that it does not violate the zero-shot learning (ZSL) premise.

Dataset | Class
--- | ---
UCF              | Pizza Tossing       
UCF              | Ice Dancing         
UCF              | Handstand Walking   
UCF              | Handstand Pushup    
UCF              | Mixing              
UCF              | Wall Pushups        
UCF              | Horse Race          
UCF              | Playing Dhol        
HMDB             | Draw Sword          
HMDB             | Sword Exercise      
HMDB             | Chew                
ActivityNet      | Applying sunscreen  
ActivityNet      | Beach soccer        
ActivityNet      | Cleaning shoes      
ActivityNet      | Cleaning sink       
ActivityNet      | Cutting the grass   
ActivityNet      | Doing karate        
ActivityNet      | Doing kickboxing    
ActivityNet      | Drinking beer       
ActivityNet      | Drinking coffee     
ActivityNet      | Fun sliding down    
ActivityNet      | Hand car wash       
ActivityNet      | Making an omelette  
ActivityNet      | Painting fence      
ActivityNet      | Playing water polo  
ActivityNet      | River tubing        
ActivityNet      | Snow tubing         
ActivityNet      | Starting a campfire 
ActivityNet      | Washing face        
ActivityNet      | Washing hands  

We next explain the rationale behind excluding the overlapping classes and completely irrelevant classes in the proposed test set. 

## Overlap between Datasets

In the figure below, we visualize the semantic embeddings of the classes in Kinetics, ActivityNet and UCF-101 datasets. We see that there are several classes in all the test datasets that directly overlap with the training dataset (Kinetics), which is a violation of the ZSL paradigm. 

![overlap](https://github.com/Secure-and-Intelligent-Systems-Lab/SemanticVideoTransformer/blob/main/tsne.png?raw=true)
![overlap](https://github.com/Secure-and-Intelligent-Systems-Lab/SemanticVideoTransformer/blob/main/Overlaps.png?raw=true)

## Irrelevant Classes

In the figure below, we breakdown the performance of the proposed model over all the classes in the UCF dataset (i.e., not only the ones included in the proposed test set). We observe that for several classes such as *nunchucks*, *YoYo*, *unevenbars*, the proposed approach is unable to classify even a single video correctly. This problem is not due to the proposed method, but due to the sheer dissimilarity of these classes with respect to the training classes in the Kinetics dataset. Since any practical algorithm will miss such classes, this emphasizes the need for removing classes that are completely irrelevant with respect to the training set from the test set.

![overlap](https://github.com/Secure-and-Intelligent-Systems-Lab/SemanticVideoTransformer/blob/main/bar.png?raw=true)

## Computational Efficiency 

Thanks to the scalability of the proposed SVT (semantic video transformer) model, we are able to vary the length of the input video snippet (i.e., number of frames), which also leads to an increase in the number of input tokens. In Table 3 in the paper, we see a significant increase in the performance when the number of input frames are increased from 8 to 96. Increasing the number of video frames is intuitive since it allows a model to better capture the spatiotemporal activities that span several frames. However, due to the current GPU limitations, we are unable to further increase the input length. On the other hand, even after increasing our model complexity to accommodate 96 input frames, our model is still more computationally efficient as compared to the I3D model with 8 input frames, which requires 10.8 TFLOPS for inference, in contrast to the proposed SVT-8 model, which only requires 0.79 TFLOPS, and SVT-96, which requires 7.57 TFLOPS.  
