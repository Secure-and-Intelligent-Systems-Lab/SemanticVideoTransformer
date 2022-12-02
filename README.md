# End-to-End Semantic Video Transformer for Zero-Shot Action Recognition
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
