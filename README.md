# LLM for Recommendation Systems

An index of large language model (LLM) for recommendation systems.

🎉 ***News: Our LLM4Rec survey has been released.***
[A Survey on Large Language Models for Recommendation](https://arxiv.org/abs/2305.19860)

***The related work and projects will be updated soon and continuously.***



# Table of Contents

- [The papers and related projects](#The-papers-and-related-projects)
  - [No Tuning](#No-Tuning)
  - [Supervised Fine-Tuning](#Supervised-Fine-Tuning)
  - [Related Survey](#Related-Survey)
- [Single card (RTX 3090) debuggable generative language models that support Chinese corpus](#Single-card-(RTX-3090)-debuggable-generative-language-models-that-support-Chinese-corpus)



## The papers and related projects

### No Tuning

Note: The tuning here only indicates whether the LLM model has been tuned.

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                                     | LLM                   |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
|    N/A   | [Hou, Y., Zhang, J., Lin, Z., Lu, H., Xie, R., McAuley, J., & Zhao, W. X. (2023). Large Language Models are Zero-Shot Rankers for Recommender Systems. arXiv preprint arXiv:2305.08845.](https://arxiv.org/pdf/2305.08845) | arxiv     | 2023     | [Python](https://github.com/RUCAIBox/LLMRank)              | ChatGPT             |
| RankGPT  | [Sun, W., Yan, L., Ma, X., Ren, P., Yin, D., & Ren, Z. (2023). Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. *arXiv preprint arXiv:2304.09542*.](https://arxiv.org/pdf/2304.09542) | arxiv     | 2023     | [Python](https://github.com/sunnweiwei/RankGPT)              | ChatGPT/4             |
|    N/A   | [Dai, S., Shao, N., Zhao, H., Yu, W., Si, Z., Xu, C., ... & Xu, J. (2023). Uncovering ChatGPT's Capabilities in Recommender Systems. *arXiv preprint arXiv:2305.02182*.](https://arxiv.org/pdf/2305.02182) | arxiv     | 2023     | [Python](https://github.com/rainym00d/LLM4RS)                | ChatGPT            |
|    N/A   | [Liu, J., Liu, C., Lv, R., Zhou, K., & Zhang, Y. (2023). Is ChatGPT a Good Recommender? A Preliminary Study. *arXiv preprint arXiv:2304.10149*.](https://arxiv.org/pdf/2304.10149) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
| GeneRec  | [Wang, W., Lin, X., Feng, F., He, X., & Chua, T. S. (2023). Generative Recommendation: Towards Next-generation Recommender Paradigm. *arXiv preprint arXiv:2304.03516*.](https://arxiv.org/pdf/2304.03516) | arxiv     | 2023     | [Python](https://github.com/Linxyhaha/GeneRec)               |         N/A              |
| Chat-REC | [Gao, Y., Sheng, T., Xiang, Y., Xiong, Y., Wang, H., & Zhang, J. (2023). Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. *arXiv preprint arXiv:2303.14524*.](https://arxiv.org/pdf/2303.14524) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
| NIR      | [Wang, L., & Lim, E. P. (2023). Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. *arXiv preprint arXiv:2304.03153*.](https://arxiv.org/pdf/2304.03153) | arxiv     | 2023     | [Python](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec) | GPT-3.5                 |
|    N/A   | [Sileo, D., Vossen, W., & Raymaekers, R. (2022, April). Zero-Shot Recommendation as Language Modeling. In *Advances in Information Retrieval: 44th European Conference on IR Research, ECIR 2022.*.](https://arxiv.org/pdf/2112.04184) | ECIR      | 2022     | [Python](https://colab.research.google.com/drive/1f1mlZ-FGaLGdo5rPzxf3vemKllbh2esT?usp=sharing) | GPT-2                 |
| UniCRS   | [Wang, X., Zhou, K., Wen, J. R., & Zhao, W. X. (2022, August). Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 1929-1937).](https://arxiv.org/pdf/2206.09363) | KDD       | 2022     | [Python](https://github.com/RUCAIBox/UniCRS)                 | GPT-2/ DialoGPT /BART |
|   GENRE  | [Liu, Q., Chen, N., Sakai, T., & Wu, X. M. (2023). A First Look at LLM-Powered Generative News Recommendation. *arXiv preprint arXiv:2305.06566*.](https://arxiv.org/pdf/2305.06566) | arxiv     | 2023     | [Python](https://github.com/Jyonn/GENRE-requests)                | ChatGPT            |



### Supervised Fine-Tuning

| **Name** | **Paper**                                                    | **Venue**        | **Year** | **Code**                                                     | LLM      |
| -------- | ------------------------------------------------------------ | ---------------- | -------- | ------------------------------------------------------------ | -------- |
| InstructRec  | [Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.](https://arxiv.org/pdf/2305.07001) | arxiv            | 2023     | N/A | FLAN-T5-3B |
|   | [Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction.](http://export.arxiv.org/pdf/2305.06474) | arxiv            | 2023     | N/A | FLAN/ChatGPT |
| TALLRec  | [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.](https://arxiv.org/pdf/2305.00447) | arxiv            | 2023     | [Python](https://paperswithcode.com/paper/graph-convolutional-matrix-completion) | Llama-7B |
| PEPLER   | [Li, L., Zhang, Y., & Chen, L. (2023). Personalized prompt learning for explainable recommendation. *ACM Transactions on Information Systems*, *41*(4), 1-26.](https://arxiv.org/pdf/2202.07371) | TOIS             | 2023     | [Python](https://github.com/lileipisces/PEPLER)              | GPT-2    |
| P5       | [Geng, S., Liu, S., Fu, Z., Ge, Y., & Zhang, Y. (2022, September). Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In *Proceedings of the 16th ACM Conference on Recommender Systems* (pp. 299-315).](https://arxiv.org/pdf/2203.13366.pdf) | RecSys           | 2022     | [Python](https://github.com/jeykigung/P5)                    | T5       |
| M6-Rec   | [Cui, Z., Ma, J., Zhou, C., Zhou, J., & Yang, H. (2022). M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems. *arXiv preprint arXiv:2205.08084*.](https://arxiv.org/pdf/2205.08084) | arxiv            | 2022     | N/A                                                           | M6       |
|    N/A   | [Zhang, Y., Ding, H., Shui, Z., Ma, Y., Zou, J., Deoras, A., & Wang, H. (2021). Language models as recommender systems: Evaluations and limitations.](https://openreview.net/pdf?id=hFx3fY7-m9b) | NeurIPS workshop | 2021     | N/A                                                           | BERT/GPT-2    |



### Related Survey

| **Paper**                                                    | **Venue** | **Year** |
| ------------------------------------------------------------ | --------- | -------- |
| [Liu, P., Zhang, L., & Gulla, J. A. (2023). Pre-train, prompt and recommendation: A comprehensive survey of language modelling paradigm adaptations in recommender systems. *arXiv preprint arXiv:2302.03735*.](https://arxiv.org/pdf/2302.03735) | arxiv     | 2023     |



### Single card (RTX 3090) debuggable generative language models that support Chinese corpus

Some open-source and effective projects can be adpated to the recommendation systems based on Chinese textual data. Especially for the individual researchers !

| Project                                                      | Year |
| ------------------------------------------------------------ | ---- |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | 2023 |
| [THUDM](https://github.com/THUDM)/**[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)** | 2023 |
| [FreedomIntelligence](https://github.com/FreedomIntelligence)/**[LLMZoo](https://github.com/FreedomIntelligence/LLMZoo)** **Phoenix** | 2023 |
| [bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)   | 2023 |
| [LianjiaTech](https://github.com/LianjiaTech)/**[BELLE](https://github.com/LianjiaTech/BELLE)** | 2023 |

Hope our conclusion can help your work.
