# CodeGenExp


*CodeGenExp* is designed to explain deep learning-based code generation applications.
*CodeGenExp* can (i) model the Markov dependency in code generation applications; (ii) handle the non-numberic data format; (iii) supporting black-box settings.
In detail, *CodeGenExp* provides token-level explanations by constructing a causal relation graph and decomposing the edge weights in the graph. 


## Design Overview

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/overview.jpg" width="800" height="300" alt="Design Overview"/><br/>
</div>    



The design overview of *CodeGenExp* is shown in the above figure. 
At a high level, *CodeGenExp* includes four main steps: (i) causal input preparation, (ii) causal graph construction, (iii) graph weight computation, and (iv) weight decomposition. For the detailed design of each step, we refer the readers to our paper.


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
* **bashXX.sh** -bash script to run experiments (**XX** are integer numbers).

## How to run

We provide the bash script that generate adversarial examples and measure the efficiency in **bash1.sh**. **bash2.sh**, **bash3.sh**, are implementing the similar functionality but for different gpus. 

So just run `bash bash1.sh`.


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





