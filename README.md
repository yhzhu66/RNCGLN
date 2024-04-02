# Robust Node Classification under Graph and Label Noise
The Robust Node Classification under Graph and Label Noise (RNCGLN) method is proposed to improve the robustness of node classification when both graph and label noise are present. In considering self-attention as a fully connected graph, the self-attention layers are used as the backbone network to resist noise impact from a global perspective. We further design the graph and label self-improvement modules to enhance the quality of supervision information and supplement supervision information for each other. As a result, RNCGLN leverages self-training and pseudo labels to facilitate the self-improvement process within an end-to-end learning framework.


The paper can be viewed via file: https://ojs.aaai.org/index.php/AAAI/article/view/29668 
The framework of our RNCGLN is listed below.
![image](https://github.com/yhzhu66/RNCGLN/assets/52006047/c7cac03b-7de8-4976-83b2-2fd75133c9cd)


# Runing code
The main_RNCGLN are two main functions, including configs for two different datasets. Different datasets need different hyperparameters to result their optimum performance.

# Other:
If you are interested in our work, please also cite our paper:
@inproceedings{zhu2024robust,
  title={Robust Node Classification on Graph Data with Graph and Label Noise},
  author={Zhu, Yonghua and Feng, Lei and Deng, Zhenyun and Chen, Yang and Amor, Robert and Witbrock, Michael},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={15},
  pages={17220--17227},
  year={2024}
}
