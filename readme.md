# DeciX


*DeciX* is a **DE**pendency-aware **C**ausal **I**nference framework for e**X**plaining the decision-making in deep learning-based code generation applications.
*DeciX* can (i) model the output-output dependency in code generation applications; (ii) handle the non-numberic data format; (iii) supporting black-box settings.
In detail, *DeciX* provides token-level explanations by constructing a causal relation graph and decomposing the edge weights in the graph. 


## A Demo Example to Explain DeciX

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/detail.jpg" width="800" height="720" alt="Design Overview"/><br/>
</div>    



## Average Overheads for Explaining One Token


## Fine-grained Overheads of DeciX

XXX


## Input/Output Length 


|   Model      | min                  | avg      | std      | max   |
|--------------|----------------------|----------|----------|-------|
| **DeepAPI**  | 2                    | 10.59    | 6.89     | 48    |
| **CodeBert** | 10                   | 72.51    | 46.04    | 409   |
| **PyGPT2**   | 4                    | 17.38    | 7.73     | 47    |

The input length statistics are shown in the above table.

|   Model      | min                  | avg      | std      | max   |
|--------------|----------------------|----------|----------|-------|
| **DeepAPI**  | 7                |  18.26    | 13.94     | 50    |
| **CodeBert** | 10                   | 78.39    | 31.06    | 100   |
| **PyGPT2**   | 100                    | 100    |   0     | 100    |

The output length statistics are shown in the above table. Notice for PyGPT2, it will not terminate untill it reaches the  configured maximum output length.


## Compare with Other Baselines




## Design Overview

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/overview.jpg" width="800" height="300" alt="Design Overview"/><br/>
</div>    



The design overview of *DeciX* is shown in the above figure. 
At a high level, *DeciX* includes four main steps: (i) causal input preparation, (ii) causal graph construction, (iii) graph weight computation, and (iv) weight decomposition. For the detailed design of each step, we refer the readers to our paper.


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

<!-- ## Accuracy of Explanations


<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/exp1.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/exp2.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/exp3.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    


The above figure shows the accuracy of explanations under three experiments. -->


## Case Study


<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/case.png" width="600" height="200" alt="Design Overview"/><br/>
</div>    

The above figure visualizes the explanation results of selected tokens in the outputs. The colored tokens are the reason that result in the target output tokens.





