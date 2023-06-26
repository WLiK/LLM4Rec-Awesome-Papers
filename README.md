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
| KAR | [Xi, Y., Liu, W., Lin, J., Zhu, J., Chen, B., Tang, R., ... & Yu, Y. (2023). Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. arXiv preprint arXiv:2306.10933.](https://arxiv.org/abs/2306.10933) | arxiv | 2023| [Python](https://gitee.com/mindspore/models/tree/master/research/recommend/KAR)| ChatGLM|
| RecAgent| [Wang, L., Zhang, J., Chen, X., Lin, Y., Song, R., Zhao, W. X., & Wen, J. R. (2023). RecAgent: A Novel Simulation Paradigm for Recommender Systems. arXiv preprint arXiv:2306.02552.](https://arxiv.org/pdf/2306.02552)| arxiv | 2023 | [Python](https://github.com/RUC-GSAI/YuLan-Rec)| ChatGPT|
| iEvaLM | [Wang, X., Tang, X., Zhao, W. X., Wang, J., & Wen, J. R. (2023). Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models. arXiv preprint arXiv:2305.13112.](https://arxiv.org/pdf/2305.13112) | arxiv | 2023 | [Python](https://github.com/rucaibox/ievalm-crs) | ChatGPT |
|    N/A   | [Hou, Y., Zhang, J., Lin, Z., Lu, H., Xie, R., McAuley, J., & Zhao, W. X. (2023). Large Language Models are Zero-Shot Rankers for Recommender Systems. arXiv preprint arXiv:2305.08845.](https://arxiv.org/pdf/2305.08845) | arxiv     | 2023     | [Python](https://github.com/RUCAIBox/LLMRank)              | ChatGPT             |
| RecLLM | [Friedman, L., Ahuja, S., Allen, D., Tan, T., Sidahmed, H., Long, C., ... & Tiwari, M. (2023). Leveraging Large Language Models in Conversational Recommender Systems. arXiv preprint arXiv:2305.07961.](https://arxiv.org/pdf/2305.07961) | arxiv | 2023 | N/A | LaMDA(video) |
|    FaiRLLM   | [Zhang, J., Bao, K., Zhang, Y., Wang, W., Feng, F., & He, X. (2023). Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation. arXiv preprint arXiv:2305.07609.](https://arxiv.org/pdf/2305.07609) | arxiv     | 2023     | [Python](https://github.com/jizhi-zhang/FaiRLLM)             | ChatGPT             |
|   GENRE  | [Liu, Q., Chen, N., Sakai, T., & Wu, X. M. (2023). A First Look at LLM-Powered Generative News Recommendation. *arXiv preprint arXiv:2305.06566*.](https://arxiv.org/pdf/2305.06566) | arxiv     | 2023     | [Python](https://github.com/Jyonn/GENRE-requests)                | ChatGPT            |
| DPLLM | [Carranza, A. G., Farahani, R., Ponomareva, N., Kurakin, A., Jagielski, M., & Nasr, M. (2023). Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models. arXiv preprint arXiv:2305.05973.](https://arxiv.org/abs/2305.05973) | arxiv | 2023 | N/A | T5 |
| N/A | [Lin, G., & Zhang, Y. (2023). Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT. arXiv preprint arXiv:2305.04518.](https://arxiv.org/abs/2305.04518) | arxiv | 2023 | N/A | ChatGPT |
|    N/A   | [Dai, S., Shao, N., Zhao, H., Yu, W., Si, Z., Xu, C., ... & Xu, J. (2023). Uncovering ChatGPT's Capabilities in Recommender Systems. *arXiv preprint arXiv:2305.02182*.](https://arxiv.org/pdf/2305.02182) | arxiv     | 2023     | [Python](https://github.com/rainym00d/LLM4RS)                | ChatGPT            |
|    N/A   | [Liu, J., Liu, C., Lv, R., Zhou, K., & Zhang, Y. (2023). Is ChatGPT a Good Recommender? A Preliminary Study. *arXiv preprint arXiv:2304.10149*.](https://arxiv.org/pdf/2304.10149) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
| RankGPT  | [Sun, W., Yan, L., Ma, X., Ren, P., Yin, D., & Ren, Z. (2023). Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. *arXiv preprint arXiv:2304.09542*.](https://arxiv.org/pdf/2304.09542) | arxiv     | 2023     | [Python](https://github.com/sunnweiwei/RankGPT)              | ChatGPT/4             |
| GeneRec  | [Wang, W., Lin, X., Feng, F., He, X., & Chua, T. S. (2023). Generative Recommendation: Towards Next-generation Recommender Paradigm. *arXiv preprint arXiv:2304.03516*.](https://arxiv.org/pdf/2304.03516) | arxiv     | 2023     | [Python](https://github.com/Linxyhaha/GeneRec)               |         N/A              |
| NIR      | [Wang, L., & Lim, E. P. (2023). Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. *arXiv preprint arXiv:2304.03153*.](https://arxiv.org/pdf/2304.03153) | arxiv     | 2023     | [Python](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec) | GPT-3.5                 |
| Chat-REC | [Gao, Y., Sheng, T., Xiang, Y., Xiong, Y., Wang, H., & Zhang, J. (2023). Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. *arXiv preprint arXiv:2303.14524*.](https://arxiv.org/pdf/2303.14524) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
|    N/A   | [Sileo, D., Vossen, W., & Raymaekers, R. (2022, April). Zero-Shot Recommendation as Language Modeling. In *Advances in Information Retrieval: 44th European Conference on IR Research, ECIR 2022.*.](https://arxiv.org/pdf/2112.04184) | ECIR      | 2022     | [Python](https://colab.research.google.com/drive/1f1mlZ-FGaLGdo5rPzxf3vemKllbh2esT?usp=sharing) | GPT-2                 |
| UniCRS   | [Wang, X., Zhou, K., Wen, J. R., & Zhao, W. X. (2022, August). Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 1929-1937).](https://arxiv.org/pdf/2206.09363) | KDD       | 2022     | [Python](https://github.com/RUCAIBox/UniCRS)                 | GPT-2/ DialoGPT /BART |




### Supervised Fine-Tuning

| **Name** | **Paper**                                                    | **Venue**        | **Year** | **Code**                                                     | LLM      |
| -------- | ------------------------------------------------------------ | ---------------- | -------- | ------------------------------------------------------------ | -------- |
| CTRL | [Li X, Chen B, Hou L, et al. CTRL: Connect Tabular and Language Model for CTR Prediction[J]. arXiv preprint arXiv:2306.02841, 2023.](Li X, Chen B, Hou L, et al. CTRL: Connect Tabular and Language Model for CTR Prediction[J]. arXiv preprint arXiv:2306.02841, 2023.) | arxiv | 2023 | N/A | P5(T5-based) |
| N/A | [Li, R., Deng, W., Cheng, Y., Yuan, Z., Zhang, J., & Yuan, F. (2023). Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights. arXiv preprint arXiv:2305.11700.](https://arxiv.org/pdf/2305.11700)| arxiv |2023 |N/A|OPT|
| PALR | [Chen, Z. (2023). PALR: Personalization Aware LLMs for Recommendation. arXiv preprint arXiv:2305.07622.](https://arxiv.org/pdf/2305.07622) | arxiv | 2023 | N/A | LLaMa |
| InstructRec  | [Zhang, J., Xie, R., Hou, Y., Zhao, W. X., Lin, L., & Wen, J. R. (2023). Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001.](https://arxiv.org/pdf/2305.07001) | arxiv            | 2023     | N/A | FLAN-T5-3B |
|  N/A | [Kang, W. C., Ni, J., Mehta, N., Sathiamoorthy, M., Hong, L., Chi, E., & Cheng, D. Z. (2023). Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction. arXiv preprint arXiv:2305.06474.](http://export.arxiv.org/pdf/2305.06474) | arxiv            | 2023     | N/A | FLAN/ChatGPT |
| TALLRec  | [Bao, K., Zhang, J., Zhang, Y., Wang, W., Feng, F., & He, X. (2023). TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. arXiv preprint arXiv:2305.00447.](https://arxiv.org/pdf/2305.00447) | arxiv            | 2023     | [Python](https://paperswithcode.com/paper/graph-convolutional-matrix-completion) | Llama-7B |
| GPT4Rec | [Li, J., Zhang, W., Wang, T., Xiong, G., Lu, A., & Medioni, G. (2023). GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation. arXiv preprint arXiv:2304.03879.](https://arxiv.org/abs/2304.03879) | arxiv | 2023 | N/A                                                          | GPT-2 |
| M6-Rec   | [Cui, Z., Ma, J., Zhou, C., Zhou, J., & Yang, H. (2022). M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems. *arXiv preprint arXiv:2205.08084*.](https://arxiv.org/pdf/2205.08084) | arxiv            | 2022     | N/A                                                           | M6       |
| N/A | [Shen, T., Li, J., Bouadjenek, M. R., Mai, Z., & Sanner, S. (2023). Towards understanding and mitigating unintended biases in language model-driven conversational recommendation. Information Processing & Management, 60(1), 103139.](https://www.sciencedirect.com/science/article/pii/S0306457322002400/pdfft?md5=dd8f44cd9e65dd103177b6799a371b27&pid=1-s2.0-S0306457322002400-main.pdf) | Inf Process Manag | 2023 | [Python](https://github.com/TinaBBB/Unintended-Bias-LMRec) | BERT |
| P5       | [Geng, S., Liu, S., Fu, Z., Ge, Y., & Zhang, Y. (2022, September). Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In *Proceedings of the 16th ACM Conference on Recommender Systems* (pp. 299-315).](https://arxiv.org/pdf/2203.13366) | RecSys           | 2022     | [Python](https://github.com/jeykigung/P5)                    | T5       |
| PEPLER   | [Li, L., Zhang, Y., & Chen, L. (2023). Personalized prompt learning for explainable recommendation. *ACM Transactions on Information Systems*, *41*(4), 1-26.](https://arxiv.org/pdf/2202.07371) | TOIS             | 2023     | [Python](https://github.com/lileipisces/PEPLER)              | GPT-2    |
|    N/A   | [Zhang, Y., Ding, H., Shui, Z., Ma, Y., Zou, J., Deoras, A., & Wang, H. (2021). Language models as recommender systems: Evaluations and limitations.](https://openreview.net/pdf?id=hFx3fY7-m9b) | NeurIPS workshop | 2021     | N/A                                                           | BERT/GPT-2    |



### Related Survey

| **Paper**                                                    | **Venue** | **Year** |
| ------------------------------------------------------------ | --------- | -------- |
|[Li, X., Zhang, Y., & Malthouse, E. C. (2023). A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News. arXiv preprint arXiv:2306.10702.](https://arxiv.org/abs/2306.10702)| arxiv | 2023 |
|[Lin, J., Dai, X., Xi, Y., Liu, W., Chen, B., Li, X., ... & Zhang, W. (2023). How Can Recommender Systems Benefit from Large Language Models: A Survey. arXiv preprint arXiv:2306.05817.](https://arxiv.org/abs/2306.05817)| arxiv| 2023 |
| [Liu, P., Zhang, L., & Gulla, J. A. (2023). Pre-train, prompt and recommendation: A comprehensive survey of language modelling paradigm adaptations in recommender systems. *arXiv preprint arXiv:2302.03735*.](https://arxiv.org/pdf/2302.03735) | arxiv     | 2023     |



### Single card (RTX 3090) debuggable generative language models that support Chinese corpus

Some open-source and effective projects can be adpated to the recommendation systems based on Chinese textual data. Especially for the individual researchers !

| Project                                                      | Year |
| ------------------------------------------------------------ | ---- |
| [baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) | 2023 |
| [YuLan-chat](https://github.com/RUC-GSAI/YuLan-Chat)         | 2023 |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | 2023 |
| [THUDM](https://github.com/THUDM)/**[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)** | 2023 |
| [FreedomIntelligence](https://github.com/FreedomIntelligence)/**[LLMZoo](https://github.com/FreedomIntelligence/LLMZoo)** **Phoenix** | 2023 |
| [bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)   | 2023 |
| [LianjiaTech](https://github.com/LianjiaTech)/**[BELLE](https://github.com/LianjiaTech/BELLE)** | 2023 |

Hope our conclusion can help your work.
