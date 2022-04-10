
# CodeGenExp

*CodeGenExp* is designed to explain deep learning-based code generation applications.
*CodeGenExp* can (i) model the Markov dependency in code generation applications; (ii) handle the non-numberic data format; (iii) supporting black-box settings.
In detail, *CodeGenExp* provides token-level explanations by constructing a causal relation graph and decomposing the edge weights in the graph. 


## Design Overview

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/CodeGenExp/blob/main/fig/codeexp.png" width="800" height="300" alt="Design Overview"/><br/>
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
    * * **./src/xai/codeexpgen** - the implementaion of our approach.
 
* **utils.py** -the basical functions to load DNNs.
* **generate_adv.py** -the script is used for generating test samples.
* **measure_latency.py** -this script measures the latency/energy consumption of the generated adversarial examples.
* **measure_loops.py**   -this script measures the iteration numbers of the generated adversarial examples.
* **measure_senstive.py** -this script measures the hyperparameter senstivelity.
* **gpuXX.sh** -bash script to run experiments (**XX** are integer numbers).



