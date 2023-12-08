# LLM for Recommendation Systems

A list of awesome papers and resources of recommender system on large language model (LLM).

🎉 ***News: Our LLM4Rec survey has been released.***
[A Survey on Large Language Models for Recommendation](https://arxiv.org/abs/2305.19860)

***The related work and projects will be updated soon and continuously.***

<div align="center">
	<img src="https://github.com/WLiK/LLM4Rec-Awesome-Papers/blob/main/llm4rec_paradigms.png" alt="Editor" width="700">
</div>

If our work has been of assistance to you, please feel free to cite our survey. Thank you.
```
@article{llm4recsurvey,
  author       = {Likang Wu and Zhi Zheng and Zhaopeng Qiu and Hao Wang and Hongchao Gu and Tingjia Shen and Chuan Qin and Chen Zhu and Hengshu Zhu and Qi Liu and Hui Xiong and Enhong Chen},
  title        = {A Survey on Large Language Models for Recommendation},
  journal      = {CoRR},
  volume       = {abs/2305.19860},
  year         = {2023}
}
```

# Table of Contents

- [The papers and related projects](#The-papers-and-related-projects)
  - [No Tuning](#No-Tuning)
  - [Supervised Fine-Tuning](#Supervised-Fine-Tuning)
  - [Related Survey](#Related-Survey)
  - [Related Tutorial](#Related-Tutorial)
  - [Common Datasets](#Common-Datasets)
- [Single card (RTX 3090) debuggable generative language models that support Chinese corpus](#Single-card-(RTX-3090)-debuggable-generative-language-models-that-support-Chinese-corpus)


## The papers and related projects

### No Tuning

Note: The tuning here only indicates whether the LLM model has been tuned.

 | **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      LLM                   |
 | -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
 | LLMRec | [LLMRec: Large Language Models with Graph Augmentation for Recommendation](https://arxiv.org/abs/2311.00423) | WSDM | 2023 | N/A | GPT |
 |RLMRec | [Representation Learning with Large Language Models for Recommendation](https://arxiv.org/abs/2310.15950) | arXiv | 2023 | [Python](https://github.com/HKUDS/RLMRec) | GPT-3.5 |
 |KP4SR | [Knowledge Prompt-tuning for Sequential Recommendation](https://dl.acm.org/doi/abs/10.1145/3581783.3612252) | ACM | 2023 | N/A | GPT-3.5 |
 |RecInterpreter | [Large Language Model Can Interpret Latent Space of Sequential Recommender](https://arxiv.org/abs/2310.20487) | arXiv | 2023 | [Python](https://github.com/YangZhengyi98/RecInterpreter) | LLaMA-7b |
 | N/A | [Large Language Models as Zero-Shot Conversational Recommenders](https://arxiv.org/abs/2308.10053) | CIKM | 2023 | [Python](https://github.com/aaronheee/llms-as-zero-shot-conversational-recsys) | GPT-3.5-turbo ,GPT-4,BAIZE,Vicuna |
 | Agent4Rec | [On Generative Agents in Recommendation](https://arxiv.org/pdf/2310.10108.pdf) | arxiv | 2023 | [Python](https://github.com/LehengTHU/Agent4Rec) | GPT4 |
 | N/A | [Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging](https://arxiv.org/abs/2309.01026) | arxiv | 2023 | N/A | BLIP-2+GPT4 |
 | InteRecAgent | [Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations](https://arxiv.org/abs/2308.16505) | arxiv | 2023 | N/A | GPT4 |
 | GPT4SM | [Are GPT Embeddings Useful for Ads and Recommendation?](https://link.springer.com/chapter/10.1007/978-3-031-40292-0_13) | KSEM | 2023 | [Python](https://github.com/Wenjun-Peng/GPT4SM) | GPT |
 | LLMRG  | [Enhancing Recommender Systems with Large Language Model Reasoning Graphs](https://arxiv.org/abs/2308.10835) | arxiv | 2023 | N/A | GPT-3.5/GPT4|
 | RAH | [RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models](https://arxiv.org/abs/2308.09904) | arxiv | 2023 | N/A | GPT4|
 | LLM-Rec | [LLM-Rec: Personalized Recommendation via Prompting Large Language Models](https://arxiv.org/pdf/2307.15780.pdf) | arxiv | 2023 | N/A | GPT-3 |
 | N/A| [Beyond Labels: Leveraging Deep Learning and LLMs for Content Metadata](https://dl.acm.org/doi/10.1145/3604915.3608883) | RecSys | 2023 | N/A | GPT4 |
 | N/A| [Retrieval-augmented Recommender System: Enhancing Recommender Systems with Large Language Models](https://dl.acm.org/doi/10.1145/3604915.3608889) | RecSys | 2023 | N/A | ChatGPT |
 | N/A| [LLM Based Generation of Item-Description for Recommendation System](https://dl.acm.org/doi/10.1145/3604915.3610647) | RecSys | 2023 | N/A | Alpaca |
 | N/A | [Large Language Models are Competitive Near Cold-start Recommenders for Language-and Item-based Preferences](https://arxiv.org/abs/2307.14225) | RecSys | 2023 | N/A | PaLM |
 | MINT | [Large Language Model Augmented Narrative Driven Recommendations](https://arxiv.org/abs/2306.02250) | Recsys | 2023 | N/A | 175B InstructGPT |
 | KAR | [Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models](https://arxiv.org/abs/2306.10933) | arxiv | 2023| [Python](https://gitee.com/mindspore/models/tree/master/research/recommend/KAR)| ChatGLM|
 | RecAgent| [RecAgent: A Novel Simulation Paradigm for Recommender Systems](https://arxiv.org/pdf/2306.02552)| arxiv | 2023 | [Python](https://github.com/RUC-GSAI/YuLan-Rec)| ChatGPT|
 | AnyPredict | [AnyPredict: Foundation Model for Tabular Prediction](https://arxiv.org/abs/2305.12081) | arxiv | 2023 | N/A | ChatGPT,BioBERT |
 | iEvaLM | [Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models](https://arxiv.org/pdf/2305.13112) | arxiv | 2023 | [Python](https://github.com/rucaibox/ievalm-crs) | ChatGPT |
 | N/A | [Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/pdf/2305.08845) | arxiv | 2023 | [Python](https://github.com/RUCAIBox/LLMRank) | ChatGPT |
 | FaiRLLM   | [Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation](https://arxiv.org/pdf/2305.07609) | Recsys | 2023     | [Python](https://github.com/jizhi-zhang/FaiRLLM)             | ChatGPT             |
 | GENRE  | [A First Look at LLM-Powered Generative News Recommendation](https://arxiv.org/pdf/2305.06566) | arxiv     | 2023     | [Python](https://github.com/Jyonn/GENRE-requests)                | ChatGPT            |
 | N/A | [Sparks of Artificial General Recommender (AGR): Early Experiments with ChatGPT](https://arxiv.org/abs/2305.04518) | arxiv | 2023 | N/A | ChatGPT| 
 |    N/A   | [Uncovering ChatGPT's Capabilities in Recommender Systems](https://arxiv.org/pdf/2305.02182) | arxiv     | 2023     | [Python](https://github.com/rainym00d/LLM4RS)                | ChatGPT            |
 |    N/A   | [Is ChatGPT a Good Recommender? A Preliminary Study](https://arxiv.org/pdf/2304.10149) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
 | VQ-Rec | [Learning vector-quantized item representation for transferable sequential recommenders](https://dl.acm.org/doi/abs/10.1145/3543507.3583434?casa_token=ZOrcB58exVUAAAAA:o7Uh_-GmRjeDzMIjPK8FDenJ2UekLc5kB95C73BlMpmXtSRLEHZFnLR7SxSRChItIgfLskwfiWkAQw) | ACM | 2023 | [Python](https://github.com/rucaibox/vq-rec) | BERT |
 | RankGPT  | [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent](https://arxiv.org/pdf/2304.09542) | arxiv     | 2023     | [Python](https://github.com/sunnweiwei/RankGPT)              | ChatGPT/4             |
 | GeneRec  | [Generative Recommendation: Towards Next-generation Recommender Paradigm](https://arxiv.org/pdf/2304.03516) | arxiv     | 2023     | [Python](https://github.com/Linxyhaha/GeneRec)               |         N/A              |
 | NIR      | [Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://arxiv.org/pdf/2304.03153) | arxiv     | 2023     | [Python](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec) | GPT-3.5                 |
 | Chat-REC | [Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://arxiv.org/pdf/2303.14524) | arxiv     | 2023     | N/A                                                           | ChatGPT         |
 |    N/A   | [Zero-Shot Recommendation as Language Modeling](https://arxiv.org/pdf/2112.04184) | ECIR      | 2022     | [Python](https://colab.research.google.com/drive/1f1mlZ-FGaLGdo5rPzxf3vemKllbh2esT?usp=sharing) | GPT-2                 |
 | UniCRS   | [Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning](https://arxiv.org/pdf/2206.09363) | KDD       | 2022     | [Python](https://github.com/RUCAIBox/UniCRS)                 | GPT-2/ DialoGPT /BART |
 | LLMRec   |  [LLMRec: Large Language Models with Graph Augmentation for Recommendation](https://arxiv.org/pdf/2311.00423.pdf) | WSDM | 2024     | [Python](https://github.com/HKUDS/LLMRec)                 | ChatGPT |
 | K-LaMP   |  [K-LaMP: Knowledge-Guided Language Model Pre-training for Sequential Recommendation](https://arxiv.org/pdf/2311.06318.pdf) | arXiv | 2023     | N/A                 | GPT-4 |




### Supervised Fine-Tuning

| **Name** | **Paper**                                                    | **Venue**        | **Year** | **Code**                                                     | LLM      |
| -------- | ------------------------------------------------------------ | ---------------- | -------- | ------------------------------------------------------------ | -------- |
| LLaRA | [LLaRA: Aligning Large Language Models with Sequential Recommenders](https://arxiv.org/abs/2312.02443) | arxiv | 2023 | [Python](https://github.com/ljy0ustc/LLaRA) | Llama-2 |
| E4SRec | [E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation](https://arxiv.org/abs/2312.02443) | arxiv | 2023 | [Python](https://github.com/HestiaSky/E4SRec/) | Llama-2 |
| LlamaRec | [LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking](https://arxiv.org/abs/2311.02089) | arxiv | 2023 | [Python](https://github.com/Yueeeeeeee/LlamaRec) | Llama-2 |
| CLLM4Rec | [Collaborative Large Language Model for Recommender Systems](https://arxiv.org/abs/2311.01343) | arxiv | 2023 | [Python](https://github.com/yaochenzhu/llm4rec) | GPT2 |
| TransRec | [A Multi-facet Paradigm to Bridge Large Language Model and Recommendation](https://arxiv.org/abs/2310.06491) | arxiv | 2023 | N/A | BART-large and LLaMA-7B |
 | RecMind| [RecMind: Large Language Model Powered Agent For Recommendation](https://arxiv.org/abs/2308.14296) | arXiv | 2023 | N/A | ChatGPT,P5 |
| RecSysLLM | [Leveraging Large Language Models for Pre-trained Recommender Systems](https://arxiv.org/abs/2308.10837) | arxiv | 2023 | N/A |GLM-10B|
 | N/A| [Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM](https://dl.acm.org/doi/fullHtml/10.1145/3604915.3608874) | RecSys | 2023 | N/A | ChatGLM-6B,P5 |
 | N/A| [Prompt Distillation for Efficient LLM-based Recommendation](https://dl.acm.org/doi/pdf/10.1145/3583780.3615017) | RecSys | 2023 | N/A | T5,P5 |
| BIGRec |  [A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems](https://arxiv.org/abs/2308.08434) | arxiv | 2023 | [Python](https://github.com/SAI990323/Grounding4Rec) |  LLaMA|
| LLMCRS |  [A Large Language Model Enhanced Conversational Recommender System](https://arxiv.org/abs/2308.06212) | arxiv | 2023 | N/A |  Flan-T5/LLaMA|
| GLRec   |  [Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations](https://arxiv.org/abs/2307.05722) | arxiv | 2023 | N/A |  BELLE|
| GIRL   |  [Generative Job Recommendations with Large Language Model](https://arxiv.org/abs/2307.02157) | arxiv | 2023 | N/A |  BELLE|
| Amazon-M2 |  [Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation](https://arxiv.org/pdf/2307.09688.pdf)| arxiv | 2023 | [Project](https://kddcup23.github.io/) |  mT5|
| GenRec   |  [GenRec: Large Language Model for Generative Recommendation](https://arxiv.org/pdf/2307.00457.pdf)| arxiv | 2023 | [Python](https://github.com/rutgerswiselab/GenRec) |  LLaMA|
| RecLLM | [Leveraging Large Language Models in Conversational Recommender Systems](https://arxiv.org/pdf/2305.07961) | arxiv | 2023 | N/A | LaMDA(video) |
 | ONCE| [ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models](https://arxiv.org/abs/2305.06566) | arXiv | 2023 | [Python](https://github.com/Jyonn/ONCE) | ChatGPT,Llama |
| DPLLM | [Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models](https://arxiv.org/abs/2305.05973) | arxiv | 2023 | N/A | T5 |
| PBNR   |  [PBNR: Prompt-based News Recommender System](https://arxiv.org/abs/2304.07862) | arxiv | 2023 | N/A |  T5|
|  GPTRec |  [Generative Sequential Recommendation with GPTRec](https://arxiv.org/abs/2306.11114) | Gen-IR@SIGIR | 2023 |  N/A | GPT-2 |
| CTRL | [CTRL: Connect Tabular and Language Model for CTR Prediction](https://arxiv.org/abs/2306.02841) | arxiv | 2023 | N/A | RoBERTa/GLM |
| UniTRec | [UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation](https://arxiv.org/abs/2305.15756)| ACL | 2023| [Python](https://github.com/Veason-silverbullet/UniTRec)  | BART|
| ICPC | [Large Language Models for User Interest Journeys](https://arxiv.org/abs/2305.15498) | arxiv| 2023|N/A| LaMDA|
| TransRec| [Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights](https://arxiv.org/abs/2305.15036)| arxiv| 2023 | N/A | RoBERTa|
| N/A | [Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights](https://arxiv.org/pdf/2305.11700)| arxiv |2023 |N/A|OPT|
| PALR | [PALR: Personalization Aware LLMs for Recommendation](https://arxiv.org/pdf/2305.07622) | arxiv | 2023 | N/A | LLaMa |
| InstructRec  | [Recommendation as instruction following: A large language model empowered recommendation approach](https://arxiv.org/pdf/2305.07001) | arxiv            | 2023     | N/A | FLAN-T5-3B |
|  N/A | [Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction](http://export.arxiv.org/pdf/2305.06474) | arxiv            | 2023     | N/A | FLAN/ChatGPT |
| LSH | [Improving Code Example Recommendations on Informal Documentation Using BERT and Query-Aware LSH: A Comparative Study](https://arxiv.org/abs/2305.03017v1) | arxiv | 2023 | N/A |BERT|
| TALLRec  | [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation](https://arxiv.org/pdf/2305.00447) | arxiv            | 2023     | [Python](https://paperswithcode.com/paper/graph-convolutional-matrix-completion) | Llama-7B |
| GPT4Rec | [GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://arxiv.org/abs/2304.03879) | arxiv | 2023 | N/A                                                          | GPT-2 |
| IDvs.MoRec| [Where to go next for recommender systems? id-vs. modality-based recommender models revisited](https://arxiv.org/abs/2303.13835) | SIGIR | 2023 | [Python](https://github.com/westlake-repl/IDvs.MoRec)| BERT|
| GReaT | [Language models are realistic tabular data generators](https://arxiv.org/abs/2210.06280) | ICLR| 2023| [Python](https://github.com/kathrinse/be_great) | GPT-2|
| M6-Rec   | [M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems](https://arxiv.org/pdf/2205.08084) | arxiv            | 2022     | N/A                                                           | M6       |
| N/A | [Towards understanding and mitigating unintended biases in language model-driven conversational recommendation](https://www.sciencedirect.com/science/article/pii/S0306457322002400/pdfft?md5=dd8f44cd9e65dd103177b6799a371b27&pid=1-s2.0-S0306457322002400-main.pdf) | Inf Process Manag | 2023 | [Python](https://github.com/TinaBBB/Unintended-Bias-LMRec) | BERT |
| P5       | [Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)](https://arxiv.org/pdf/2203.13366) | RecSys           | 2022     | [Python](https://github.com/jeykigung/P5)                    | T5       |
| PEPLER   | [Personalized prompt learning for explainable recommendation](https://arxiv.org/pdf/2202.07371) | TOIS             | 2023     | [Python](https://github.com/lileipisces/PEPLER)              | GPT-2    |
|    N/A   | [Language models as recommender systems: Evaluations and limitations](https://openreview.net/pdf?id=hFx3fY7-m9b) | NeurIPS workshop | 2021     | N/A                                                           | BERT/GPT-2    |

### Related Survey

| **Paper**                                                    | **Venue** | **Year** |
| ------------------------------------------------------------ | --------- | -------- |
| [Large Language Models for Generative Recommendation: A Survey and Visionary Discussions](https://arxiv.org/abs/2309.01157) | arxiv | 2023 |
| [Robust Recommender System: A Survey and Future Directions](https://arxiv.org/abs/2309.02057) | arxiv | 2023 |
| [Large Language Models for Generative Recommendation: A Survey and Visionary Discussions](https://arxiv.org/abs/2309.01157) | arxiv | 2023 |
| [A Survey on Multi-Behavior Sequential Recommendation](https://arxiv.org/abs/2308.15701) | arxiv | 2023 |
| [When large language models meet personalization: Perspectives of challenges and opportunities](https://arxiv.org/abs/2307.16376) | arxiv | 2023 |
| [Recommender systems in the era of large language models (llms)](https://arxiv.org/abs/2307.02046) | arxiv | 2023 |
| [A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News](https://arxiv.org/abs/2306.10702)| arxiv | 2023 |
| [How Can Recommender Systems Benefit from Large Language Models: A Survey](https://arxiv.org/abs/2306.05817)| arxiv| 2023 |
| [Pre-train, prompt and recommendation: A comprehensive survey of language modelling paradigm adaptations in recommender systems](https://arxiv.org/pdf/2302.03735) | arxiv     | 2023     |

### Related Tutorial
| **Name** | **Venue** | **Year** |
| ------------------------------------------------------------ | --------- | -------- |
| [Large Language Models for Recommendation: Progresses and Future Directions](https://dl.acm.org/doi/abs/10.1145/3624918.3629550) | SIGIR-AP | 2023|
| [Tutorial on Large Language Models for Recommendation](https://dl.acm.org/doi/10.1145/3604915.3609494)| RecSys|2023|

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
| PixelRec | Short Video | Seq Rec/CF Rec | PixelRec is a large dataset of cover images collected from a short video recommender system, comprising approximately 200 million user image interactions, 30 million users, and 400,000 video cover images. The texts and other aggregated attributes of videos are also included. | [link](https://github.com/westlake-repl/PixelRec) |



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

