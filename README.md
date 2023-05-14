# Voice Classification using MFCC and Gaussian Mixture Models  
  
## About
This is one of my first self-made project. The script trains on `.wav` recordings and then classifies them. 
  
## External Packages  
The following packages are required for script to work:
```text
Name                Version
python              3.8.16
numpy               1.24.2
librosa             0.10.0.post2
scikit-learn        1.2.2
tqdm                4.65.0
pandas              1.2.5
matplotlib-base     3.7.1
matplotlib-inline   0.1.6
seaborn             0.12.2
```  
Note: These are version I used while coding. Thus, it is possible that any version higher that mentioned above will work as well.
  
## How to run it?
1. Download the code or clone the repository.  
2. Download and localize recordings from my repository, download some dataset from the internet or prepare it yourself!
3. Run the following command in your Terminal:  
```text
python <path_to_a_script> --train_folder_path <train_folder_path>   
--test_folder_path <test_folder_path>
```  
In my case, using `voices` folder, the command looks like this:
```text
python voice_classic.py --train_folder_path ./voices/train
--test_folder_path ./voices/test
```  
## Results  
The classification results with default values of `n_mfcc` and `n_components `:  
```python output
Classifying recordings...
100%|█████████████████████████████████████████████████████████████████| 7/7 [00:25<00:00,  3.62s/it]

____ Accuracy ____                                                                                                                                                            
n_components        10        15        20        25        30        35        40
n_mfcc                                                                            
10            0.222222  0.111111  0.111111  0.111111  0.333333  0.333333  0.222222
15            0.222222  0.111111  0.111111  0.111111  0.111111  0.777778  0.888889
20            0.222222  0.222222  0.333333  0.222222  0.222222  0.222222  0.888889
25            0.333333  0.222222  1.000000  0.222222  0.222222  0.888889  0.888889
30            0.333333  0.222222  0.888889  0.111111  0.888889  0.888889  0.888889
35            0.222222  0.111111  0.111111  0.111111  0.888889  0.888889  0.888889
40            0.666667  0.777778  0.777778  0.888889  0.888889  0.888889  0.888889
```
I also allowed myself to visualize the output with `seaborn.heatmap()`:
![Heatmap of accuacy](/Users/stawager/Repozytoria/Portfolio/voice_classification_mfcc_gmm/readme_imgs/accuracies.png)
## Idea credits
- dr inż. Tomasz Kryjak
- mgr. inż. Hubert Szolc
