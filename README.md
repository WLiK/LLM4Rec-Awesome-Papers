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
  - [Common Datasets](#Common-Datasets)
- [Single card (RTX 3090) debuggable generative language models that support Chinese corpus](#Single-card-(RTX-3090)-debuggable-generative-language-models-that-support-Chinese-corpus)
- [Citation](#Citation)


## The papers and related projects

### No Tuning

Note: The tuning here only indicates whether the LLM model has been tuned.

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                                     | LLM                   |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| N/A | [Sanner, S., Balog, K., Radlinski, F., Wedin, B., & Dixon, L. (2023). Large Language Models are Competitive Near Cold-start Recommenders for Language-and Item-based Preferences. arXiv preprint arXiv:2307.14225.](https://arxiv.org/abs/2307.14225) | RecSys | 2023 | N/A | PaLM |
| MINT | [Mysore S, McCallum A, Zamani H. Large Language Model Augmented Narrative Driven Recommendations[J]. arXiv preprint arXiv:2306.02250, 2023.](https://arxiv.org/abs/2306.02250) | arxiv | 2023 | N/A | 175B InstructGPT |
| KAR | [Xi, Y., Liu, W., Lin, J., Zhu, J., Chen, B., Tang, R., ... & Yu, Y. (2023). Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. arXiv preprint arXiv:2306.10933.](https://arxiv.org/abs/2306.10933) | arxiv | 2023| [Python](https://gitee.com/mindspore/models/tree/master/research/recommend/KAR)| ChatGLM|
| RecAgent| [Wang, L., Zhang, J., Chen, X., Lin, Y., Song, R., Zhao, W. X., & Wen, J. R. (2023). RecAgent: A Novel Simulation Paradigm for Recommender Systems. arXiv preprint arXiv:2306.02552.](https://arxiv.org/pdf/2306.02552)| arxiv | 2023 | [Python](https://github.com/RUC-GSAI/YuLan-Rec)| ChatGPT|
| AnyPredict | [Wang Z, Gao C, Xiao C, et al. AnyPredict: Foundation Model for Tabular Prediction[J]. arXiv preprint arXiv:2305.12081, 2023.](https://arxiv.org/abs/2305.12081) | arxiv | 2023 | N/A | ChatGPT,BioBERT |
| iEvaLM | [Wang, X., Tang, X., Zhao, W. X., Wang, J., & Wen, J. R. (2023). Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models. arXiv preprint arXiv:2305.13112.](https://arxiv.org/pdf/2305.13112) | arxiv | 2023 | [Python](https://github.com/rucaibox/ievalm-crs) | ChatGPT |
|    N/A   | [Hou, Y., Zhang, J., Lin, Z., Lu, H., Xie, R., McAuley, J., & Zhao, W. X. (2023). Large Language Models are Zero-Shot Rankers for Recommender Systems. arXiv preprint arXiv:2305.08845.](https://arxiv.org/pdf/2305.08845) | arxiv     | 2023     | [Python](https://github.com/RUCAIBox/LLMRank)              | ChatGPT             |
|    FaiRLLM   | [Zhang, J., Bao, K., Zhang, Y., Wang, W., Feng, F., & He, X. (2023). Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation. arXiv preprint arXiv:2305.07609.](https://arxiv.org/pdf/2305.07609) | arxiv     | 2023     | [Python](https://github.com/jizhi-zhang/FaiRLLM)             | ChatGPT             |
|   GENRE  | [Liu, Q., Chen, N., Sakai, T., & Wu, X. M. (2023). A First Look at LLM-Powered Generative News Recommendation. *arXiv preprint arXiv:2305.06566*.](https://arxiv.org/pdf/2305.06566) | arxiv     | 2023     | [Python](https://github.com/Jyonn/GENRE-requests)                | ChatGPT            |
| DPLLM | [Carranza, A. G., Farahani, R., Ponomareva, N., Kurakin, A., Jagielski, M., & Nasr, M. (2023). Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models. arXiv preprint arXiv:2305.05973.](https://arxiv.org/abs/2305.05973) | arxiv | 2023 | N/A | T5 |
| N/A | [Lin, G., & Zhang, Y. (2023). Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT. arXiv preprint arXiv:2305.04518.](https://arxiv.org/abs/2305.04518) | arxiv | 2023 | N/A | ChatGPT |
|    N/A   | [Dai, S., Shao, N., Zhao, H., Yu, W., Si, Z., Xu, C., ... & Xu, J. (2023). Uncovering ChatGPT's Capabilities in Recommender Systems. *arXiv preprint arXiv:2305.02182*.](https://arxiv.org/pdf/2305.02182) | arxiv     | 2023     | [Python](https://github.com/rainym00d/LLM4RS)                | ChatGPT            |
|    N/A   | [Liu, J., Liu, C., Lv, R., Zhou, K., & Zhang, Y. (2023). Is ChatGPT a Good Recommender? A Preliminary Study. *arXiv preprint arXiv:2304.10149*.](https://arxiv.org/pdf/2304.10149) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
| VQ-Rec | [Hou Y, He Z, McAuley J, et al. Learning vector-quantized item representation for transferable sequential recommenders[C]//Proceedings of the ACM Web Conference 2023. 2023: 1162-1171.](https://dl.acm.org/doi/abs/10.1145/3543507.3583434?casa_token=ZOrcB58exVUAAAAA:o7Uh_-GmRjeDzMIjPK8FDenJ2UekLc5kB95C73BlMpmXtSRLEHZFnLR7SxSRChItIgfLskwfiWkAQw) | ACM | 2023 | [Python](https://github.com/rucaibox/vq-rec) | BERT |
| RankGPT  | [Sun, W., Yan, L., Ma, X., Ren, P., Yin, D., & Ren, Z. (2023). Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. *arXiv preprint arXiv:2304.09542*.](https://arxiv.org/pdf/2304.09542) | arxiv     | 2023     | [Python](https://github.com/sunnweiwei/RankGPT)              | ChatGPT/4             |
| GeneRec  | [Wang, W., Lin, X., Feng, F., He, X., & Chua, T. S. (2023). Generative Recommendation: Towards Next-generation Recommender Paradigm. *arXiv preprint arXiv:2304.03516*.](https://arxiv.org/pdf/2304.03516) | arxiv     | 2023     | [Python](https://github.com/Linxyhaha/GeneRec)               |         N/A              |
| NIR      | [Wang, L., & Lim, E. P. (2023). Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. *arXiv preprint arXiv:2304.03153*.](https://arxiv.org/pdf/2304.03153) | arxiv     | 2023     | [Python](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec) | GPT-3.5                 |
| Chat-REC | [Gao, Y., Sheng, T., Xiang, Y., Xiong, Y., Wang, H., & Zhang, J. (2023). Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. *arXiv preprint arXiv:2303.14524*.](https://arxiv.org/pdf/2303.14524) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
|    N/A   | [Sileo, D., Vossen, W., & Raymaekers, R. (2022, April). Zero-Shot Recommendation as Language Modeling. In *Advances in Information Retrieval: 44th European Conference on IR Research, ECIR 2022.*.](https://arxiv.org/pdf/2112.04184) | ECIR      | 2022     | [Python](https://colab.research.google.com/drive/1f1mlZ-FGaLGdo5rPzxf3vemKllbh2esT?usp=sharing) | GPT-2                 |
| UniCRS   | [Wang, X., Zhou, K., Wen, J. R., & Zhao, W. X. (2022, August). Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 1929-1937).](https://arxiv.org/pdf/2206.09363) | KDD       | 2022     | [Python](https://github.com/RUCAIBox/UniCRS)                 | GPT-2/ DialoGPT /BART |




### Supervised Fine-Tuning

| **Name** | **Paper**                                                    | **Venue**        | **Year** | **Code**                                                     | LLM      |
| -------- | ------------------------------------------------------------ | ---------------- | -------- | ------------------------------------------------------------ | -------- |
| GLRec   |  [Wu, L., Qiu, Z., Zheng, Z., Zhu, H., & Chen, E. (2023). Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations.](https://arxiv.org/abs/2307.05722) | arxiv | 2023 | N/A |  N/A|
| GIRL   |  [Zheng, Z., Qiu, Z., Hu, X., Wu, L., Zhu, H., & Xiong, H. (2023). Generative Job Recommendations with Large Language Model.](https://arxiv.org/abs/2307.02157) | arxiv | 2023 | N/A |  N/A|
| Amazon-M2 |  [Jin, Wei et al. Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation. ArXiv abs/2307.09688 (2023)](https://arxiv.org/pdf/2307.09688.pdf)| arxiv | 2023 | [Project](https://kddcup23.github.io/) |  mT5|
| GenRec   |  [Ji, J., Li, Z., Xu, S., Hua, W., Ge, Y., Tan, J., & Zhang, Y. (2023). GenRec: Large Language Model for Generative Recommendation. arXiv e-prints, arXiv-2307.](https://arxiv.org/pdf/2307.00457.pdf)| arxiv | 2023 | [Python](https://github.com/rutgerswiselab/GenRec) |  LLaMA|
| RecLLM | [Friedman, L., Ahuja, S., Allen, D., Tan, T., Sidahmed, H., Long, C., ... & Tiwari, M. (2023). Leveraging Large Language Models in Conversational Recommender Systems. arXiv preprint arXiv:2305.07961.](https://arxiv.org/pdf/2305.07961) | arxiv | 2023 | N/A | LaMDA(video) |
| PBNR   |  [Li, X., Zhang, Y., & Malthouse, E. C. (2023). PBNR: Prompt-based News Recommender System. arXiv preprint arXiv:2304.07862.](https://arxiv.org/abs/2304.07862) | arxiv | 2023 | N/A |  T5|
|  GPTRec |  [Petrov, A. V., & Macdonald, C. (2023). Generative Sequential Recommendation with GPTRec. arXiv preprint arXiv:2306.11114.](https://arxiv.org/abs/2306.11114) | Gen-IR@SIGIR | 2023 |  N/A | GPT-2 |
| CTRL | [Li X, Chen B, Hou L, et al. CTRL: Connect Tabular and Language Model for CTR Prediction[J]. arXiv preprint arXiv:2306.02841, 2023.](https://arxiv.org/abs/2306.02841) | arxiv | 2023 | N/A | RoBERTa/GLM |
| UniTRec | [Mao, Z., Wang, H., Du, Y., & Wong, K. F. (2023). UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation. arXiv preprint arXiv:2305.15756.](https://arxiv.org/abs/2305.15756)| ACL | 2023| [Python](https://github.com/Veason-silverbullet/UniTRec)  | BART|
| ICPC | [Christakopoulou, K., Lalama, A., Adams, C., Qu, I., Amir, Y., Chucri, S., ... & Chen, M. (2023). Large Language Models for User Interest Journeys. arXiv preprint arXiv:2305.15498.](https://arxiv.org/abs/2305.15498) | arxiv| 2023|N/A| LaMDA|
| TransRec| [Fu, J., Yuan, F., Song, Y., Yuan, Z., Cheng, M., Cheng, S., ... & Pan, Y. (2023). Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights. arXiv preprint arXiv:2305.15036.](https://arxiv.org/abs/2305.15036)| arxiv| 2023 | N/A | RoBERTa|
| N/A | [Li, R., Deng, W., Cheng, Y., Yuan, Z., Zhang, J., & Yuan, F. (2023). Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights. arXiv preprint arXiv:2305.11700.](https://arxiv.org/pdf/2305.11700)| arxiv |2023 |N/A|OPT|
| PALR | [Chen, Z. (2023). PALR: Personalization Aware LLMs for Recommendation. arXiv preprint arXiv:2305.07622.](https://arxiv.org/pdf/2305.07622) | arxiv | 2023 | N/A | LLaMa |
| InstructRec  | [Zhang, J., Xie, R., Hou, Y., Zhao, W. X., Lin, L., & Wen, J. R. (2023). Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001.](https://arxiv.org/pdf/2305.07001) | arxiv            | 2023     | N/A | FLAN-T5-3B |
|  N/A | [Kang, W. C., Ni, J., Mehta, N., Sathiamoorthy, M., Hong, L., Chi, E., & Cheng, D. Z. (2023). Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction. arXiv preprint arXiv:2305.06474.](http://export.arxiv.org/pdf/2305.06474) | arxiv            | 2023     | N/A | FLAN/ChatGPT |
| LSH | [Rahmani, S., Naghshzan, A., & Guerrouj, L. (2023). Improving Code Example Recommendations on Informal Documentation Using BERT and Query-Aware LSH: A Comparative Study. arXiv preprint arXiv:2305.03017.](https://arxiv.org/abs/2305.03017v1) | arxiv | 2023 | N/A |BERT|
| TALLRec  | [Bao, K., Zhang, J., Zhang, Y., Wang, W., Feng, F., & He, X. (2023). TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. arXiv preprint arXiv:2305.00447.](https://arxiv.org/pdf/2305.00447) | arxiv            | 2023     | [Python](https://paperswithcode.com/paper/graph-convolutional-matrix-completion) | Llama-7B |
| GPT4Rec | [Li, J., Zhang, W., Wang, T., Xiong, G., Lu, A., & Medioni, G. (2023). GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation. arXiv preprint arXiv:2304.03879.](https://arxiv.org/abs/2304.03879) | arxiv | 2023 | N/A                                                          | GPT-2 |
| IDvs.MoRec| [Yuan, Z., Yuan, F., Song, Y., Li, Y., Fu, J., Yang, F., ... & Ni, Y. (2023). Where to go next for recommender systems? id-vs. modality-based recommender models revisited. arXiv preprint arXiv:2303.13835.](https://arxiv.org/abs/2303.13835) | SIGIR | 2023 | [Python](https://github.com/westlake-repl/IDvs.MoRec)| BERT|
| GReaT | [Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2022). Language models are realistic tabular data generators. arXiv preprint arXiv:2210.06280.](https://arxiv.org/abs/2210.06280) | ICLR| 2023| [Python](https://github.com/kathrinse/be_great) | GPT-2|
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


### Common Datasets

| Name           | Scene         | Tasks        | Information                                                                                                           | URL                                                   |
|----------------|---------------|--------------|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| Amazon Review  | Commerce      | Seq Rec/CF Rec | This is a large crawl of product reviews from Amazon. Ratings: 82.83 million, Users: 20.98 million, Items: 9.35 million, Timespan: May 1996 - July 2014 | [link](http://jmcauley.ucsd.edu/data/amazon/)        |
| Amazon-M2      | Commerce      | Seq Rec/CF Rec | A large dataset of anonymized user sessions with their interacted products collected from multiple language sources at Amazon. It includes 3,606,249 train sessions, 361,659 test sessions, and 1,410,675 products. | [link](https://arxiv.org/abs/2307.09688)             |
| Steam          | Game          | Seq Rec/CF Rec | Reviews represent a great opportunity to break down the satisfaction and dissatisfaction factors around games. Reviews: 7,793,069, Users: 2,567,538, Items: 15,474, Bundles: 615 | [link](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) |
| MovieLens      | Movie         | General       | The dataset consists of 4 sub-datasets, which describe users' ratings to movies and free-text tagging activities from MovieLens, a movie recommendation service. | [link](https://grouplens.org/datasets/movielens/)    |
| Yelp           | Commerce      | General       | There are 6,990,280 reviews, 150,346 businesses, 200,100 pictures, 11 metropolitan areas, 908,915 tips by 1,987,897 users. Over 1.2 million business attributes like hours, parking, availability, etc. | [link](https://www.yelp.com/dataset)                 |
| Douban         | Movie, Music, Book | Seq Rec/CF Rec | This dataset includes three domains, i.e., movie, music, and book, and different kinds of raw information, i.e., ratings, reviews, item details, user profiles, tags (labels), and date. | [link](https://paperswithcode.com/dataset/douban)    |
| MIND           | News          | General       | MIND contains about 160k English news articles and more than 15 million impression logs generated by 1 million users. Every news contains textual content including title, abstract, body, category, and entities. | [link](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf) |
| U-NEED         | Commerce      | Conversation Rec | U-NEED consists of 7,698 fine-grained annotated pre-sales dialogues, 333,879 user behaviors, and 332,148 product knowledge tuples. | [link](https://github.com/LeeeeoLiu/U-NEED)          |



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

<br/>

If our work has been of assistance to you, please feel free to cite our survey. Thank you.
### Citation
```
@article{llm4recsurvey,
  author       = {Likang Wu and
                  Zhi Zheng and
                  Zhaopeng Qiu and
                  Hao Wang and
                  Hongchao Gu and
                  Tingjia Shen and
                  Chuan Qin and
                  Chen Zhu and
                  Hengshu Zhu and
                  Qi Liu and
                  Hui Xiong and
                  Enhong Chen},
  title        = {A Survey on Large Language Models for Recommendation},
  journal      = {CoRR},
  volume       = {abs/2305.19860},
  year         = {2023}
}
```
