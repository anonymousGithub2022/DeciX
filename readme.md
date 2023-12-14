# Rebuttal 

## Comparsion against White-box Approaches.
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DeciX/blob/main/fig/white-box-baseline-page-001.jpg" width="720" height="400" alt="Design Overview"/><br/>
</div>    

The comparsion results against two white-box baselines are presented in the above figure, from the results, we observe that although DeciX does not outperform whitebox baselines. However,DeciX is still comparable against them. Moreover, DeciX is more generable, as these two baselines can not be applied on DeepAPI, which does not use transformer layer.


## Hyperparameter Study

| Subject  | percentage | Deduction | Augmentation | Synthetic  |
|----------|------------|-----------|--------------|------------|
| DeepAPI  | 0.2        | 22.16     | 16.90        | 12.97      |
|          | 0.4        | 23.11     | 17.71        | 12.55      |
|          | 0.6        | 20.04     | 14.32        | 11.79      |
| CodeBert | 0.2        | 38.03     | 63.08        | 66.83      |
|          | 0.4        | 38.88     | 65.55        | 67.89      |
|          | 0.6        | 35.55     | 62.31        | 65.46      |
| PyGPT2   | 0.2        | 9.80      | 14.40        | 16.48      |
|          | 0.4        | 10.12     | 14.56        | 16.78      |
|          | 0.6        | 9.89      | 12.66        | 14.32      |

The hyperparameter study of different mutation percentage are shown in the above Table, form the results, we confirm that DeciX is not senstive to the hyperparameter mutation percentage.


## Tokenizer Study

The model name of our study are presented [here](https://github.com/anonymousGithub2022/DeciX/blob/main/model_tokenizer.txt), and the similarity matrix are shown in the [file](https://github.com/anonymousGithub2022/DeciX/blob/main/fig/cal_new_en_vocab%20(2).csv)



## Commercial Code Model

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DeciX/blob/main/fig/gpt-page-002.jpg" width="720" height="400" alt="Design Overview"/><br/>
</div>    
The evaluation on commercial GPT3.5 model are shown in the above figure, from the results, we can observe that DeciX outperform the baseline methods a significant margin in the ommercial GPT3.5 model.

# DeciX


*DeciX* is a **DE**pendency-aware **C**ausal **I**nference framework for e**X**plaining the decision-making in deep learning-based code generation applications.
*DeciX* can (i) model the output-output dependency in code generation applications; (ii) handle the non-numberic data format; (iii) supporting black-box settings.
In detail, *DeciX* provides token-level explanations by constructing a causal relation graph and decomposing the edge weights in the graph. 


## A Demo Example to Explain DeciX

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/detail.png" width="720" height="800" alt="Design Overview"/><br/>
</div>    

In our step 1, we randomly select 20% of the input token and replace the selected tokens with random tokens. We then compare the mutant with the original input and get the binarized causal input/output.

In our step 2, we construct the graph with two tyeps of edges based on the input token order and output token order.

In step 3, we treat each output token as a target and train a regression model to fit the causal contribution scores. The above figure shows an example to compute the causal contribution scores for the third output token **now**. Although Decix and LIME both use a regression model in this step, Decix differentiates from LIME from the following two perspectives: (i) Decix fits a regression model on the binarized casual input/output, while LIME fits a regression model directly on the input/output. (ii) Decix considers not only the input token but also previous output tokens to fit the regression model (the green box in step 3), which is the unique part in Decix and does not exist in existing work.

In our step 4, we decompose the output-output dependency in our previous step.



## File Structure
* **src** -main source codes.
  * **./src/CodeBert** - the application of CodeBert.
  * **./src/GPT2** - the application of PyGPT2.
  * **./src/DeepAPI** -the application of DeepAPI.
  * **./src/wrapper_model.py** -the wrapper model of the mentioned applications.
  * **./src/xai** -the lib that includes the implementation of each explanation methods.
    * **./src/xai/lemna** - the implementaion of lemna.
    * **./src/xai/lime** - the implementaion of lime.
    * **./src/xai/codeexpgen** - the implementaion of our approach.
 
* **utils.py** -the basical functions to load DNNs.
* **generate_explanation.py** -the script is used for explanation each DL-based code generation applications.
* **evaluate_explanation.py** -this script is used to evaluate the accuracy of the explanations.
* **post_acc.py**.   -get the accuracy results.
* **bashXX.sh** -bash script to run experiments (**XX** are integer numbers that represent the code generation model ID).
* **requirement.txt** -the dependent libraries.

## Setup
We strongly recommend the user use the *conda* to manage the virtual environment.

First create an environment with *conda*.
`conda create -n your_env_name python=3.7`

Second, activate the virtual environment.
`conda activate your_env_name`.

Next, install the basic library dependency.
`pip install -r requirement.txt`.

Finally, download the pre-trained model weights from [model_weight](https://drive.google.com/drive/folders/1KJBahf25i9ttQr8VWF8tA9e87IBw5Q0E?usp=sharing)
and put the model weights in the directory `model_weight`.
The model weights will be `model_weight/deepAPI` and `model_weight/pytorch_model.bin`.


## Quick Start

We have run the explanation scripts offline and stored the explanation results in the directory `exp_res`. 

To quickly evaluate the explanation quality, run `bash demo_bash1.sh`, `bash demo_bash2.sh` and `bash demo_bash3.sh`. 

After that, run `python post_acc.py` to plot the figures.

All explanation results are stored in the directory `final_res`.


## How to run

We provide the bash script that generate adversarial examples and measure the efficiency in **bash1.sh**. **bash2.sh**, **bash3.sh**, are implementing the similar functionality but for different gpus. 

So just run `bash bash1.sh`, `bash bash2.sh`, and `bash bash3.sh`.
 
After get the results, run `python post_acc.py` to plot the results.


## Fine-grained Overheads of DeciX (s)

| **Model**    | **Step 1** | **Step 2** | **Step 3** | **Step 4** | **Total Overheads** |
|--------------|------------|------------|------------|------------|----------------------|
| **DeepAPI**  | 2.4       | 0.0          | 0.3        | 0.1      | 2.8                  |
| **CodeBERT** | 104.2      | 0.0          | 19.9       | 0.1      | 124.2                |
| **PyGPT2**   | 23.0      | 0.0          | 2.0        | 0.1      | 25.1                 |

The above Table shows the fine-grained overheads of DeciX explaining one data instance (For LIME and LEMNA, we use the API call to extract explanation and thus can not get the fine-grained overheads). 
The main overheads of DeciX come from step 1. This is because Decix feeds many mutants to the code generation model for inference in step 1, and the inference overhead of the code generation model under explain occupies the main overheads of this step.


## Input/Output Length 


|   Model      | min                  | avg      | std      | max   |
|--------------|----------------------|----------|----------|-------|
| **DeepAPI**  | 2                    | 10.59    | 6.89     | 48    |
| **CodeBert** | 10                   | 72.51    | 46.04    | 409   |
| **PyGPT2**   | 4                    | 17.38    | 7.73     | 47    |

The input length statistics are shown in the above table.  Notice for CodeBERT, the input length is significant larger than the input length of other subjects.

|   Model      | min                  | avg      | std      | max   |
|--------------|----------------------|----------|----------|-------|
| **DeepAPI**  | 7                |  18.26    | 13.94     | 50    |
| **CodeBert** | 10                   | 78.39    | 31.06    | 100   |
| **PyGPT2**   | 100                    | 100    |   0     | 100    |

The output length statistics are shown in the above table. Notice for PyGPT2, it will not terminate untill it reaches the  configured maximum output length.



<!--  ## Compare with Other Baselines

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/more_exp.png" width="720" height="800" alt="Design Overview"/><br/>
</div> 

The above figure shows the results compared with other baselines. The results show that WT5 and MICE perform similarly to the Random baseline. This observation is because the WT5 and MICE are trained on the NLP corpus and are not suitable for explaining the code generation applications.

## Design Overview

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/overview.jpg" width="800" height="300" alt="Design Overview"/><br/>
</div>    



The design overview of *DeciX* is shown in the above figure. 
At a high level, *DeciX* includes four main steps: (i) causal input preparation, (ii) causal graph construction, (iii) graph weight computation, and (iv) weight decomposition. For the detailed design of each step, we refer the readers to our paper.

## Accuracy of Explanations


<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/exp1.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/exp2.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/exp3.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    


The above figure shows the accuracy of explanations under three experiments. 


## Case Study


<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/case.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    

The above figure visualizes the explanation results of selected tokens in the outputs. The colored tokens are the reason that result in the target output tokens.  -->




