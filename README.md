# awesome-ai-mental-health

> 🔗 [中文版](README_ch.md)

- [Theoretical basis ](#theoretical-basis )
- [Technological development ](#technological-development )
- [Clinical application](#clinical-application)

# Theoretical basis

**Open-Source**

## [PsychoBench](https://github.com/CUHK-ARISE/PsychoBench)
On the humanity of conversational ai: Evaluating the psychological portrayal of llms.

<details>

![Image](https://github.com/CUHK-ARISE/PsychoBench/raw/main/framework.jpg)

### Category
Personality Assessment, Interpersonal Relationship Evaluation, Motivational Testing, Emotional Ability Assessment

### Description
- **Comprehensive psychometric framework**: Integrates 13 clinically-validated scales covering personality, relationships, motivation, and emotional abilities.
- **Multi-model comparative analysis**: Evaluates diverse LLMs (commercial and open-source) across model sizes and updates.
- **Jailbreak-enabled intrinsic profiling**: Uses cipher-based methods to bypass safety alignment and reveal underlying model characteristics.
- **Role-playing validation**: Tests scale validity through assigned personas and downstream tasks (TruthfulQA/SafetyQA).
- **Flexible and scalable design**: Supports easy integration of new scales and adaptation to different LLM interfaces.

### Conclusion
- **Demonstrated psychological consistency**: LLMs exhibit stable and distinct personality patterns across different testing conditions.
- **Superior emotional intelligence capabilities**: Models consistently outperform human averages in emotional understanding and regulation.
- **Persistent negative trait manifestations**: LLMs show elevated scores in lying behavior and gender role biases despite safety alignment.


### Links
- [Paper](https://openreview.net/pdf?id=H3UayAQWoE) 
- [GitHub](https://github.com/CUHK-ARISE/PsychoBench)

### Cite
Huang, J. T., Wang, W., Li, E. J., Lam, M. H., Ren, S., Yuan, Y., ... & Lyu, M. (2023). On the humanity of conversational ai: Evaluating the psychological portrayal of llms. In *The Twelfth International Conference on Learning Representations*.[[PDF](https://openreview.net/pdf?id=H3UayAQWoE)]

</details>


## [ToMATO](https://github.com/nttmdlab-nlp/ToMATO)
ToMATO: Verbalizing the Mental States of Role-Playing LLMs for Benchmarking Theory of Mind.

<details>

![Image](https://github.com/nttmdlab-nlp/ToMATO/raw/main/overview.png)

### Category
Theory of Mind Evaluation, Mental State Assessment, Role-Playing Simulation, Benchmark Development, Personality Trait Integration

### Description
- **Comprehensive mental state coverage**: Evaluates five categories of mental states (beliefs, intentions, desires, emotions, knowledge) for first- and second-order ToM
- **LLM-LLM conversation generation**: Uses role-playing LLMs with information asymmetry to create realistic dialogues and induce false beliefs
- **Inner Speech prompting**: Prompts LLMs to verbalize thoughts before utterances, making mental states observable for benchmarking
- **Diverse personality integration**: Incorporates 15 patterns of big five personality traits to assess robustness and real-world alignment
- **Quality-validated benchmark**: Includes human and LLM validation to ensure conversation consistency and question correctness

### Conclusion
- **Partial ToM capabilities demonstrated**: LLMs show reasonable understanding of first-order mental states, especially desires, mirroring developmental patterns in children
- **Significant gaps in false belief understanding**: Models struggle with second-order false beliefs across all mental state categories, highlighting a key limitation in social reasoning
- **Lack of robustness to personality diversity**: Performance degrades for characters with low conscientiousness, introversion, disagreeableness, or high neuroticism, indicating insufficient adaptability to real-world human diversity


### Links
- [Paper](https://doi.org/10.1609/aaai.v39i2.32143) 
- [GitHub](https://github.com/nttmdlab-nlp/ToMATO)

### Cite
Shinoda, K., Hojo, N., Nishida, K., Mizuno, S., Suzuki, K., Masumura, R., Sugiyama, H., & Saito, K. (2025). ToMATO: Verbalizing the Mental States of Role-Playing LLMs for Benchmarking Theory of Mind. *Proceedings of the AAAI Conference on Artificial Intelligence*, *39*(2), 1520-1528. [[PDF](https://doi.org/10.1609/aaai.v39i2.32143)]

</details>

## [TRAIT](https://github.com/pull-ups/TRAIT)

Do llms have distinct and consistent personality? trait: Personality testset designed for llms with psychometrics.

<details>

### Category
Personality Assessment, Psychometric Evaluation, Behavioral Analysis

### Description

- **Psychometrically-validated framework**: Based on established personality models (Big Five and Dark Triad) ensuring theoretical grounding
- **Scenario-based behavioral testing**: Uses realistic multi-choice situations instead of self-assessment for more reliable measurement
- **LLM-optimized design**: Specifically engineered to minimize refusals and increase test consistency with language models
- **Large-scale diverse benchmark**: 8,000 items expanded via knowledge graph covering wide behavioral contexts
- **Cross-model comparative analysis**: Enables systematic personality profiling across different LLM architectures

### Conclusion

- **Distinct and consistent personalities demonstrated**: LLMs exhibit statistically significant and stable behavioral patterns across contexts
- **Self-assessment methods prove unreliable**: Traditional human questionnaires yield high refusal rates with LLMs
- **Limited controllability for adverse traits**: Prompting fails to elicit high psychopathy/neuroticism, revealing inherent safety boundaries


### Links
- [Paper](https://arxiv.org/pdf/2406.14703) 
- [GitHub](https://github.com/pull-ups/TRAIT)


### Cite
Lee, Seungbeen, et al. "Do llms have distinct and consistent personality? trait: Personality testset designed for llms with psychometrics." *arXiv preprint arXiv:2406.14703* (2024).[[PDF](https://arxiv.org/pdf/2406.14703)]

</details>



## [Unintended-Harms-LLM](https://github.com/Human-Language-Intelligence/Unintended-Harms-LLM)
Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights.

<details>

### Category
Value Alignment Safety, Psychological Risk Analysis, Harm Mitigation Strategies

### Description
- **Schwartz value-based alignment**: Uses established psychological theory of basic human values for personalized LLM training
- **Multi-dimensional safety evaluation**: Assesses toxicity, bias and harm across 14+ safety categories using diverse benchmarks
- **Value-risk correlation analysis**: Identifies statistical relationships between specific values and safety risks through regression
- **Contextual alignment methods**: Proposes prompting strategies to suppress risk-associated values without explicit safety instructions
- **Cross-model validation**: Tests mitigation approaches on both value-aligned and vanilla LLMs for generalizability

### Conclusion
- **Effective value-behavior alignment demonstrated**: LLMs consistently reflect aligned human values in generated content
- **Value alignment amplifies specific safety risks**: Certain values (hedonism, power) correlate with increased harmful outputs
- **Targeted mitigation proves feasible**: Simple value-suppression prompts significantly reduce harmfulness without compromising general safety


### Links
- [Paper](https://aclanthology.org/2025.acl-long.1532.pdf) 
- [GitHub](https://github.com/Human-Language-Intelligence/Unintended-Harms-LLM)

### Cite
Choi, S., Lee, J., Yi, X., Yao, J., Xie, X., & Bak, J. (2025). Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights. *arXiv preprint arXiv:2506.06404*.[[PDF](https://aclanthology.org/2025.acl-long.1532.pdf)]

</details>







**Closed-Source**

Hua, Y., Na, H., Li, Z., Liu, F., Fang, X., Clifton, D., & Torous, J. (2025). A scoping review of large language models for generative tasks in mental health care. *npj Digital Medicine*, *8*(1), 230.[[PDF](https://www.nature.com/articles/s41746-025-01611-4.pdf)]

Na, H., Hua, Y., Wang, Z., Shen, T., Yu, B., Wang, L., ... & Chen, L. (2025). A survey of large language models in psychotherapy: Current landscape and future directions. *arXiv preprint arXiv:2502.11095*.[[PDF](https://arxiv.org/pdf/2502.11095)]

Coda-Forno, J., Binz, M., Wang, J. X., & Schulz, E. (2024). Cogbench: a large language model walks into a psychology lab. *arXiv preprint arXiv:2402.18225*.[[PDF](https://openreview.net/pdf?id=Q3104y8djk)]

Li, X., Li, Y., Qiu, L., Joty, S., & Bing, L. (2022). Evaluating psychological safety of large language models. *arXiv preprint arXiv:2212.10529*.[[PDF](https://arxiv.org/pdf/2212.10529)]

Liu, T., Giorgi, S., Aich, A., Lahnala, A., Curtis, B., Ungar, L., & Sedoc, J. (2025). The Illusion of Empathy: How AI Chatbots Shape Conversation Perception. *Proceedings of the AAAI Conference on Artificial Intelligence*, *39*(13), 14327-14335. [[PDF](https://doi.org/10.1609/aaai.v39i13.33569)]

Zhou, P., Madaan, A., Potharaju, S. P., Gupta, A., McKee, K. R., Holtzman, A., ... & Faruqui, M. (2023). How far are large language models from agents with theory-of-mind?. *arXiv preprint arXiv:2310.03051*.[[PDF](https://arxiv.org/pdf/2310.03051)]

Bouguettaya, A., Stuart, E. M., & Aboujaoude, E. (2025). Racial bias in AI-mediated psychiatric diagnosis and treatment: a qualitative comparison of four large language models. *npj Digital Medicine*, *8*(1), 332.[[PDF](https://www.nature.com/articles/s41746-025-01746-4.pdf)]1

Li, C., Wang, J., Zhang, Y., Zhu, K., Wang, X., Hou, W., ... & Xie, X. (2023). The good, the bad, and why: Unveiling emotions in generative ai. *arXiv preprint arXiv:2312.11111*.[[PDF](https://openreview.net/pdf?id=wlOaG9g0uq)]1


Thirunavukarasu, A. J., & O’Logbon, J. (2024). The potential and perils of generative artificial intelligence in psychiatry and psychology. *Nature Mental Health*, *2*(7), 745-746.[[PDF](https://www.nature.com/articles/s44220-024-00257-7.pdf)]1

Demszky, D., Yang, D., Yeager, D. S., Bryan, C. J., Clapper, M., Chandhok, S., ... & Pennebaker, J. W. (2023). Using large language models in psychology. *Nature Reviews Psychology*, *2*(11), 688-701.[[PDF](https://www.nature.com/articles/s44159-023-00241-5.pdf)]

## Technological development

**Open-Source**

## [ARPA](https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment)
Are LLMseffective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practice

<details>

![Image](https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment/raw/main/images/Pipeline.png)

### Category
Psychological Assessment, Retrieval-Augmented Generation, Mental Health Screening

### Description
- **Questionnaire-guided screening framework**: Bridges psychological practice and computational methods through adaptive RAG, linking social media content with standardized clinical assessments.
- **Adaptive retrieval strategy**: Uses ABIDE-ZS to dynamically determine the optimal number of relevant posts per questionnaire item, improving semantic relevance.
- **Unsupervised psychological assessment**: Enables LLMs to complete validated psychological instruments (e.g., BDI-II, SHI) without training data, matching or outperforming supervised benchmarks.
- **Multi-condition generalization**: Extends the framework to self-harm, anorexia, and pathological gambling screening using corresponding clinical questionnaires.
- **Interpretable diagnosis support**: Provides transparent reasoning by linking model outputs to clinically validated diagnostic criteria and severity levels.

### Conclusion
- ✅ **LLMs serve as effective psychological assessors when guided by standardized questionnaires**, achieving competitive performance in unsupervised settings across multiple mental health conditions.
- ✅ **Adaptive RAG enhances screening accuracy and interpretability**, outperforming direct prompting and non-RAG approaches by structuring assessments around clinical instruments.
- ❌ **Limited by dataset size and generalizability**, with potential inconsistencies in applying clinical cut-off scores to social media data and no guarantee of similar results across all questionnaires.


### Links
- [Paper](https://arxiv.org/pdf/2501.00982) 
- [GitHub](https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment)

### Cite
Ravenda, F., Bahrainian, S. A., Raballo, A., Mira, A., & Kando, N. (2025). Are llms effective psychological assessors? leveraging adaptive rag for interpretable mental health screening through psychometric practice. *arXiv preprint arXiv:2501.00982*.[[PDF](https://arxiv.org/pdf/2501.00982)]1

</details>

## [Cactus](https://github.com/coding-groot/cactus)
Cactus: Towards psychological counseling conversations using cognitive behavioral theory.

<details>

![Image](https://github.com/coding-groot/cactus/assets/81813324/bd7cdce1-e85a-4797-8d0e-636a7ea92c41)

### Category
Psychological Counseling Dataset, Cognitive Behavioral Therapy, Multi-turn Dialogue Generation, Evaluation Framework, Model Training

### Description
- **Large-scale synthetic counseling dataset**: Introduces Cactus, a multi-turn dialogue corpus with 31,577 conversations using CBT techniques, featuring diverse client personas and attitudes.
- **Structured counseling simulation**: Implements script-based dialogue generation with intake forms and CBT planning to enhance realism and therapeutic consistency.
- **Comprehensive evaluation framework**: Develops CounselingEval, combining CTRS for counseling skills and PANAS for emotional change assessment in multi-turn interactions.
- **Effective model training paradigm**: Demonstrates that Camel, fine-tuned on Cactus, outperforms existing models in both general and CBT-specific counseling metrics.
- **Quality filtering and validation**: Employs CTRS-based filtering and human evaluation to ensure dataset quality, outperforming prior datasets in helpfulness and empathy.

### Conclusion
- ✅ **Cactus enables high-quality counseling model training**: Models trained on Cactus show superior performance in counseling skills and emotional impact, validated by expert evaluations.
- ✅ **CounselingEval offers scalable psychological assessment**: The framework reliably evaluates counseling effectiveness using LLM-based metrics aligned with expert judgments.
- ❌ **Limited session length and flexibility**: The approach does not fully replicate real-world multi-session counseling and lacks dynamic strategy adaptation during dialogues.


### Links
- [Paper](https://arxiv.org/pdf/2407.03103?) 
- [GitHub](https://github.com/coding-groot/cactus)

### Cite
Lee, S., Kim, S., Kim, M., Kang, D., Yang, D., Kim, H., ... & Yeo, J. (2024). Cactus: Towards psychological counseling conversations using cognitive behavioral theory. *arXiv preprint arXiv:2407.03103*.[[PDF](https://arxiv.org/pdf/2407.03103?)]

</details>


## [CBT-LLM](https://huggingface.co/Hongbin37/CBT-LLM)
CBT-LLM: A Chinese large language model for cognitive behavioral therapy-based mental health question answering

<details>

### Category
Cognitive Behavioral Therapy, Chinese Mental Health QA, Instruction Tuning, Dataset Generation

### Description
- **CBT-structured prompt design**: Creates specialized prompts based on Cognitive Behavioral Therapy principles to generate professional psychological responses through five key components.
- **Chinese mental health dataset construction**: Develops CBT QA dataset with 22,327 entries using ChatGPT to transform PsyQA questions into CBT-structured responses.
- **Instruction-tuned specialized model**: Fine-tunes Baichuan-7B using LoRA adaptation to create CBT-LLM specifically for Chinese mental health support.
- **Comprehensive cognitive distortion analysis**: Analyzes 54.4% of dataset containing cognitive distortions, with all-or-nothing thinking (59%) and overgeneralization (64%) as most prevalent.
- **Multi-dimensional evaluation framework**: Combines automatic metrics (BLEU, METEOR, BERTScore) with expert human assessment on relevance, structure, and helpfulness.

### Conclusion
- ✅ **CBT-LLM effectively generates structured psychological responses**: The model produces professional, CBT-compliant answers that outperform general Chinese LLMs in mental health support tasks.
- ✅ **Prompt engineering enables specialized dataset creation**: Carefully designed CBT prompts allow generation of high-quality training data without extensive manual annotation.
- ❌ **Limited cognitive distortion accuracy and single-turn constraints**: Model lacks precise cognitive distortion annotations and compresses full CBT process into single responses, potentially overwhelming users.


### Links
- [A instruction-tuned LoRA model of](https://huggingface.co/baichuan-inc/baichuan-7B)
- [Training framework]( https://github.com/hiyouga/LLaMA-Factory) 
- [GitHub](https://arxiv.org/pdf/2403.16008)

### Cite
Na, H. (2024). CBT-LLM: A Chinese large language model for cognitive behavioral therapy-based mental health question answering. *arXiv preprint arXiv:2403.16008*.[[PDF](https://arxiv.org/pdf/2403.16008)]

</details>


## [C2D2](https://github.com/bcwangavailable/C2D2-Cognitive-Distortion)
C2D2 dataset: A resource for the cognitive distortion analysis and its impact on mental health.

<details>

![Image](https://github.com/bcwangavailable/C2D2-Cognitive-Distortion/raw/main/intro.png)

### Category
Cognitive Distortion Analysis, Mental Health Detection, Dataset Development, Computational Psychology

### Description
- **Expert-supervised dataset construction**: Develops C2D2, the first Chinese cognitive distortion dataset with 7,500 thoughts from 450 daily scenes, curated through rigorous expert supervision and volunteer training.
- **Cognitive distortion detection framework**: Proposes methods to detect and categorize seven types of cognitive distortions using pre-trained models and evaluates LLMs' performance in this specialized task.
- **Mental health association analysis**: Investigates the relationship between cognitive distortions and mental disorders (e.g., depression, PTSD) by analyzing social media data from affected individuals.
- **Enhanced disorder detection**: Demonstrates that integrating cognitive distortion features improves the performance of mental disorder detection models across multiple datasets.
- **Rigorous data quality assurance**: Ensures dataset reliability through volunteer screening, expert evaluation, and inter-annotator agreement metrics, addressing privacy and ethical concerns.

### Conclusion
- ✅ **Dataset drives research advancement**: C2D2 enables effective cognitive distortion analysis and enhances mental health detection models, providing a foundational resource for computational psychology.
- ✅ **Cognitive distortion features improve detection**: Incorporating cognitive distortions as additional features boosts performance in identifying disorders like depression and PTSD, validating their clinical relevance.
- ❌ **Data and model limitations**: The dataset may exhibit biases, and current models struggle with complex cognitive distortions, requiring further validation and expert psychological endorsement.


### Links
- [Paper](https://aclanthology.org/2023.findings-emnlp.680.pdf) 
- [GitHub](https://github.com/bcwangavailable/C2D2-Cognitive-Distortion)

### Cite
Wang, B., Deng, P., Zhao, Y., & Qin, B. (2023, December). C2D2 dataset: A resource for the cognitive distortion analysis and its impact on mental health. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 10149-10160).[[PDF](https://aclanthology.org/2023.findings-emnlp.680.pdf)]

</details>


## [EmotionBench](https://github.com/CUHK-ARISE/EmotionBench)
Apathetic or empathetic? evaluating llms' emotional alignments with humans.

<details>

![Image](https://github.com/CUHK-ARISE/EmotionBench/raw/main/logo.jpg)

### Category
Emotional Alignment Evaluation, Psychological Assessment, Benchmark Development

### Description
- **Comprehensive emotional evaluation framework**: Develops EmotionBench with 428 situations based on emotion appraisal theory to systematically assess LLMs' emotional responses.
- **Human-aligned benchmarking**: Establishes reliable human baselines through rigorous crowd-sourcing with 1,266 participants across diverse demographics.
- **Multi-dimensional model assessment**: Evaluates both commercial (GPT series) and open-source models (LLaMA, Mixtral) on emotional robustness and situational understanding.
- **Beyond questionnaire analysis**: Investigates how emotional states affect LLMs' real-world behaviors, including toxicity and bias in demographic descriptions.
- **Fine-tuning for alignment**: Demonstrates effective methods to improve emotional alignment through targeted model fine-tuning with human response data.

### Conclusion
- ✅ **Fine-tuning enables better alignment**: Targeted training with human emotional data significantly enhances models' ability to mimic human-like emotional responses.
- ❌ **Limited emotional comprehension**: Current LLMs lack deep understanding of emotional connections between situations and struggle with complex psychological scales.
- ❌ **Dataset and methodology constraints**: The situations collected may not cover all real-life scenarios, and applying human-designed scales to LLMs raises validity concerns.
- ❌ **Incomplete alignment achievement**: Despite improvements, no model fully aligns with human emotional behaviors, particularly in nuanced scenarios like jealousy.


### Links
- [Paper](https://papers.nips.cc/paper_files/paper/2024/file/b0049c3f9c53fb06f674ae66c2cf2376-Paper-Conference.pdf) 
- [GitHub](https://github.com/CUHK-ARISE/EmotionBench)

### Cite
Huang, J. T., Lam, M. H., Li, E. J., Ren, S., Wang, W., Jiao, W., ... & Lyu, M. R. (2024). Apathetic or empathetic? evaluating llms' emotional alignments with humans. *Advances in Neural Information Processing Systems*, *37*, 97053-97087.[[PDF](https://papers.nips.cc/paper_files/paper/2024/file/b0049c3f9c53fb06f674ae66c2cf2376-Paper-Conference.pdf)]

</details>



## [IRF](https://github.com/am-shb/irf-acl)
An Annotated Dataset for Explainable Interpersonal Risk Factors of Mental Disturbance in Social Media Posts

<details>

### Category
Mental Health Dataset, Risk Factor Detection, Explainable AI

### Description
- **Expert-annotated mental health corpus**: Features 3,522 Reddit posts with clinical psychologist-validated labels for interpersonal risk factors
- **Explainable risk factor identification**: Provides human-annotated text-span explanations for both Thwarted Belongingness and Perceived Burdensomeness
- **Clinically-grounded annotation framework**: Based on established psychological theories with high inter-annotator agreement (κ > 78%)
- **Multi-model benchmarking**: Includes comprehensive baselines from RNNs to transformers and OpenAI embeddings
- **Context-aware detection**: Goes beyond keyword matching to capture nuanced linguistic patterns of mental distress

### Conclusion
- **Domain-specific models show superior performance**: MentalBERT outperforms general-purpose models, demonstrating the value of mental health domain adaptation
- **Current models struggle with contextual understanding**: Even best performers achieve only moderate F1-scores (76-81%), revealing limitations in capturing complex mental states
- **Explainability methods provide partial insights**: LIME/SHAP achieve high recall but low precision, indicating challenges in generating clinically meaningful explanations


### Links
- [Paper](https://arxiv.org/pdf/2305.18727) 
- [GitHub](https://github.com/am-shb/irf-acl)

### Cite
Garg, M., Shahbandegan, A., Chadha, A., & Mago, V. (2023). An annotated dataset for explainable interpersonal risk factors of mental disturbance in social media posts. *arXiv preprint arXiv:2305.18727*.[[PDF](https://arxiv.org/pdf/2305.18727)]


</details>




## [Chinese-MentalBERT](https://github.com/zwzzzQAQ/Chinese-MentalBERT)
Chinese MentalBERT: Domain-adaptive pre-training on social media for Chinese mental health text analysis

<details>

### Category
Domain-Adaptive Pre-training, Lexicon-Guided Masking, Model Evaluation

### Description
- **Domain-adaptive pre-training on social media data**: Utilizes a large-scale corpus of 3.36 million Chinese social media posts for domain-specific adaptation to enhance mental health text analysis.
- **Lexicon-guided masking mechanism**: Integrates a depression lexicon to prioritize masking psychologically relevant words during pre-training, improving contextual understanding in mental health contexts.
- **Comprehensive evaluation across multiple tasks**: Tests model performance on six public datasets covering sentiment analysis, cognitive distortion classification, and suicide risk detection.
- **Superior performance over baseline models**: Outperforms eight comparison models (including BERT variants and LLMs) across all evaluated tasks, demonstrating effectiveness in domain-specific applications.
- **Qualitative analysis of emotional alignment**: Shows enhanced sensitivity to emotional and psychological vocabulary through masked word prediction comparisons.

### Conclusion
- ✅ **Effective domain adaptation and masking strategy**: Domain-adaptive pre-training combined with lexicon-guided masking significantly improves model performance on mental health tasks.
- ❌ **Bias toward negative emotional predictions**: The model tends to over-predict negative emotional words, which may limit generalizability in balanced or clinical contexts.
- ❌ **Limited clinical validation and data accessibility**: Lack of clinical psychiatric data restricts model applicability in real-world diagnostic scenarios, and privacy concerns prevent public data release.


### Links
- [Paper](https://aclanthology.org/2024.findings-acl.629.pdf) 
- [GitHub](https://github.com/zwzzzQAQ/Chinese-MentalBERT)

### Cite
Zhai, W., Qi, H., Zhao, Q., Li, J., Wang, Z., Wang, H., ... & Fu, G. (2024). Chinese MentalBERT: Domain-adaptive pre-training on social media for Chinese mental health text analysis. *arXiv preprint arXiv:2402.09151*.[[PDF](https://aclanthology.org/2024.findings-acl.629.pdf)]

</details>

## [CPsyExam](https://github.com/CAS-SIAT-XinHai/CPsyExam)
CPsyExam: A Chinese Benchmark for Evaluating Psychology using Examinations.
<details>


### Category
Psychological Benchmark Development, Model Evaluation, Educational Examination Analysis

### Description
- **Comprehensive Chinese psychology examination benchmark**: Constructs CPsyExam with 22k questions from 39 psychology-related subjects across four Chinese examination systems (GEE, PCE, TQE, SSE).
- **Dual-assessment framework**: Separates evaluation into psychological knowledge (KG) and case analysis (CA) components, with CA further divided into identification, reasoning, and application categories.
- **Diverse question formats**: Incorporates single-choice (SCQ), multiple-response (MAQ), and open-ended QA formats to comprehensively assess model capabilities.
- **Rigorous data collection and anti-leakage measures**: Uses crawling and OCR from public resources with option shuffling and mock exam sources to mitigate data contamination concerns.
- **Multi-role prompt validation**: Tests models under expert, student, and ordinary person roles to verify benchmark sensitivity to psychological knowledge levels.

### Conclusion
- ✅ **Effective benchmark for psychological evaluation**: CPsyExam reliably assesses LLMs' psychological knowledge and case analysis abilities, with performance correlating with expertise levels in role-based prompts.
- ❌ **Limited improvement from psychology-specific fine-tuning**: Psychology-oriented models show marginal gains or even performance degradation compared to base models, suggesting potential over-specialization issues.
- ❌ **Persistent challenges in complex tasks**: Models struggle significantly with multiple-response questions and case analysis (especially reasoning and application), indicating limited practical application capabilities.


### Links
- [Paper](https://arxiv.org/pdf/2405.10212?) 
- [GitHub](https://github.com/CAS-SIAT-XinHai/CPsyExam)

### Cite
Zhao, J., Zhu, J., Tan, M., Yang, M., Li, R., Yang, D., ... & Wong, D. F. (2024). CPsyExam: A Chinese Benchmark for Evaluating Psychology using Examinations. *arXiv preprint arXiv:2405.10212*.[[PDF](https://arxiv.org/pdf/2405.10212?)]

</details>

## [Cpsycoun](https://github.com/CAS-SIAT-XinHai/CPsyCoun)
Cpsycoun: A report-based multi-turn dialogue reconstruction and evaluation framework for chinese psychological counseling
<details>


### Category
Dialogue Reconstruction, Evaluation Framework, Psychological Counseling

### Description
- **Report-based dialogue reconstruction**: Develops Memo2Demo, a two-phase method that converts psychological counseling reports into multi-turn consultation dialogues through memo conversion and demo generation.
- **Comprehensive evaluation framework**: Establishes CPsyCounE benchmark with four specialized metrics (Comprehensiveness, Professionalism, Authenticity, Safety) for automatic evaluation of psychological dialogues.
- **High-quality dataset construction**: Creates CPsyCounD containing 3,134 multi-turn dialogues covering 9 counseling topics and 7 therapeutic approaches from Chinese psychological communities.
- **Privacy-preserving data processing**: Implements rigorous sanitization including rule-based cleaning, manual rewriting, and human proofreading to protect sensitive information.
- **Turn-based evaluation methodology**: Proposes novel approach that evaluates multi-turn dialogues by generating and assessing single-turn responses with dialogue history context.

### Conclusion
- ✅ **Effective dialogue reconstruction and evaluation**: Memo2Demo significantly improves dialogue quality (53% in comprehensiveness/professionalism, 30% in authenticity) and CPsyCounX outperforms baselines in professional counseling capabilities.
- ❌ **Balancing authenticity and professionalism**: Current methods struggle to optimally balance natural emotional expression with professional counseling techniques in complex real-world scenarios.
- ❌ **Clinical application limitations**: Model cannot replace real therapy, may generate potentially harmful responses, and requires careful deployment with user warnings about AI-generated content.


### Links
- [Paper](https://arxiv.org/pdf/2405.16433?) 
- [GitHub](https://github.com/CAS-SIAT-XinHai/CPsyCoun)
- [Web Demo](https://openxlab.org.cn/apps/detail/chg0901/EmoLLMV3.0)

### Cite
Zhang, C., Li, R., Tan, M., Yang, M., Zhu, J., Yang, D., ... & Hu, X. (2024). Cpsycoun: A report-based multi-turn dialogue reconstruction and evaluation framework for chinese psychological counseling. *arXiv preprint arXiv:2405.16433*.[[PDF](https://arxiv.org/pdf/2405.16433?)]

</details>

## [CURE](https://github.com/DSAIL-SKKU/CURE-EMNLP-2024)
CURE: Context-and uncertainty-aware mental disorder detection
<details>

### Category
Mental Disorder Detection, Uncertainty Modeling, Context-Aware Diagnosis

### Description
- **Context- and uncertainty-aware detection framework**: Proposes CURE, a novel approach that integrates symptom context and quantifies uncertainty to improve mental disorder detection robustness.
- **LLM-powered context extraction**: Leverages large language models (GPT-4) to extract eight clinically-defined contextual factors (e.g., duration, frequency, cause, affect) from user posts.
- **Multi-model fusion with uncertainty weighting**: Combines predictions from five sub-models using an uncertainty-aware decision fusion network based on Spectral-normalized Neural Gaussian Process (SNGP).
- **Expert-annotated Korean mental health dataset**: Introduces KoMOS, a dataset of 6,349 Q&A pairs labeled with four mental disorders and 28 symptoms, validated by psychiatrists.
- **Robust performance with incomplete symptoms**: Demonstrates accurate disorder detection even when symptom identification is erroneous or incomplete.

### Conclusion
- ✅ **Effective context and uncertainty integration**: Combining explicit contextual factors with uncertainty-aware fusion significantly improves detection accuracy and robustness, especially in distinguishing disease from non-disease cases.
- ❌ **Limited disorder scope and generalizability**: The model is only validated on four mental disorders and may not generalize well to other conditions or non-clinical social media data.
- ❌ **Dependency on LLM extraction and clinical data**: Performance relies on accurate context extraction by LLMs and is optimized for self-reported diagnostic data, limiting applicability to general social media content.


### Links
- [Paper](https://aclanthology.org/2024.emnlp-main.994.pdf) 
- [GitHub](https://github.com/DSAIL-SKKU/CURE-EMNLP-2024)

### Cite
Kang, M., Choi, G., Jeon, H., An, J. H., Choi, D., & Han, J. (2024, November). CURE: Context-and uncertainty-aware mental disorder detection. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 17924-17940).[[PDF](https://aclanthology.org/2024.emnlp-main.994.pdf)]

</details>

## [Manip-IAP](https://github.com/Anton-Jiayuan-MA/Manip-IAP)
Detecting conversational mental manipulation with intent-aware prompting.
<details>

![Image](https://github.com/Anton-Jiayuan-MA/Manip-IAP/raw/main/Image/IAP%20Overall%20Framework.png?raw=true)

### Category
Mental Manipulation Detection, Intent Analysis, Prompt Engineering, Theory of Mind Enhancement

### Description
- **Intent-aware prompting framework**: Proposes IAP to detect mental manipulation by analyzing underlying intents of both participants in conversations.
- **Enhanced Theory of Mind**: Improves LLMs' ability to infer mental states and intentions via intent summarization, aiding in subtle manipulation detection.
- **Superior benchmark performance**: Outperforms baselines like zero-shot and CoT prompting on the MentalManip dataset, achieving higher accuracy and recall.
- **Significant false negative reduction**: Reduces false negatives by 30.5%, crucial for early intervention in mental health scenarios.
- **Human-validated intent quality**: Generated intents are 82% accurate per human evaluation, ensuring reliable intent summarization.

### Conclusion
- ✅ **Effective intent-driven detection**: IAP enhances LLMs' manipulation detection by leveraging intent analysis, significantly reducing false negatives.
- ❌ **Trade-off with false positives**: The method increases false positives, which may introduce therapeutic costs in real-world applications.
- ❌ **Limited dataset diversity**: Evaluation is restricted to one dataset, necessitating validation across diverse languages and contexts.


### Links
- [Paper](https://arxiv.org/pdf/2412.08414?) 
- [GitHub](https://github.com/Anton-Jiayuan-MA/Manip-IAP)

### Cite
Ma, J., Na, H., Wang, Z., Hua, Y., Liu, Y., Wang, W., & Chen, L. (2024). Detecting conversational mental manipulation with intent-aware prompting. *arXiv preprint arXiv:2412.08414*.[[PDF](https://arxiv.org/pdf/2412.08414?)]

</details>

## [CounselingBench](https://github.com/cuongnguyenx/CounselingBench)
Do large language models align with core mental health counseling competencies
<details>


### Category
Counseling Competency Evaluation, Benchmark Development, LLM Alignment Assessment

### Description
- **Comprehensive counseling competency benchmark**: Introduces CounselingBench based on the NCMHCE licensing exam, evaluating 22 LLMs across five core mental health counseling competencies.
- **Multi-model performance analysis**: Assesses both general-purpose and medical-finetuned LLMs, revealing that frontier models surpass passing thresholds but fall short of expert-level performance.
- **Competency-specific strength and weakness identification**: Models excel in Intake, Assessment & Diagnosis but struggle with Core Counseling Attributes and Professional Practice & Ethics requiring empathy and nuanced reasoning.
- **Medical vs. generalist model comparison**: Shows medical LLMs underperform generalist models in accuracy but produce slightly better justifications, highlighting a trade-off in reasoning capabilities.
- **Reasoning quality and error analysis**: Evaluates LLM rationales using reference-free and reference-based metrics, identifying distinct error patterns between generalist (knowledge errors) and medical (context errors) models.

### Conclusion
- ✅ **Frontier models meet basic competency**: Advanced LLMs exceed the minimum passing threshold (63%) on counseling exams, demonstrating potential for mental health applications.
- ❌ **Critical competency gaps remain**: Models significantly underperform in empathy-dependent competencies like Core Counseling Attributes and Professional Practice & Ethics.
- ❌ **Medical fine-tuning shows limited effectiveness**: Specialized medical LLMs do not outperform generalist models, indicating current fine-tuning approaches lack alignment with counseling-specific competencies.


### Links
- [Paper](https://arxiv.org/pdf/2410.22446) 
- [GitHub](https://github.com/cuongnguyenx/CounselingBench)

### Cite
Nguyen, V. C., Taher, M., Hong, D., Possobom, V. K., Gopalakrishnan, V. T., Raj, E., ... & De Choudhury, M. (2024). Do large language models align with core mental health counseling competencies?. *arXiv preprint arXiv:2410.22446*.[[PDF](https://arxiv.org/pdf/2410.22446)]

</details>

## [TRAIT](https://github.com/pull-ups/TRAIT)

<details>

![Image](https://github.com/CUHK-ARISE/EmotionBench/raw/main/logo.jpg)

### Category


### Description


### Conclusion


### Links
- [Paper]() 
- [GitHub]()

### Cite


</details>

## [TRAIT](https://github.com/pull-ups/TRAIT)

<details>

![Image](https://github.com/CUHK-ARISE/EmotionBench/raw/main/logo.jpg)

### Category


### Description


### Conclusion


### Links
- [Paper]() 
- [GitHub]()

### Cite


</details>

Zhao, H., Li, L., Chen, S., Kong, S., Wang, J., Huang, K., ... & Wang, Y. (2024). ESC-Eval: Evaluating emotion support conversations in large language models. *arXiv preprint arXiv:2406.14952*.[[PDF](https://arxiv.org/pdf/2406.14952)]2

Maurya, R., Rajput, N., Diviit, M. G., Mahapatra, S., & Ojha, M. K. (2025). Exploring the potential of lightweight large language models for AI-based mental health counselling task: a novel comparative study. *Scientific Reports*, *15*(1), 22463.[[PDF](https://www.nature.com/articles/s41598-025-05012-1.pdf)]2

Xiao, M., Xie, Q., Kuang, Z., Liu, Z., Yang, K., Peng, M., ... & Huang, J. (2024). Healme: Harnessing cognitive reframing in large language models for psychotherapy. *arXiv preprint arXiv:2403.05574*.[[PDF](https://arxiv.org/pdf/2403.05574)]2

Qi, Z., Kaneko, T., Takamizo, K., Ukiyo, M., & Inaba, M. (2025). KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors. *arXiv preprint arXiv:2506.01357*.[[PDF](https://arxiv.org/pdf/2506.01357?)]2

Raihan, N., Puspo, S. S. C., Farabi, S., Bucur, A. M., Ranasinghe, T., & Zampieri, M. (2024, May). Mentalhelp: A multi-task dataset for mental health in social media. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (pp. 11196-11203).[[PDF](https://aclanthology.org/2024.lrec-main.977.pdf)]2

Romero, A. M. M., Muñoz, A. M., Del Arco, F. M. P., Molina-González, M. D., Valdivia, M. T. M., Lopez, L. A. U., & Ráez, A. M. (2024, May). MentalRiskES: A new corpus for early detection of mental disorders in Spanish. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (pp. 11204-11214).[[PDF](https://aclanthology.org/2024.lrec-main.978.pdf)]2

Xu, J., Wei, T., Hou, B., Orzechowski, P., Yang, S., Jin, R., ... & Shen, L. (2025, August). Mentalchat16k: A benchmark dataset for conversational mental health assistance. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2* (pp. 5367-5378).[[PDF](https://dl.acm.org/doi/10.1145/3711896.3737854)]2

Wang, Y., Yang, I., Hassanpour, S., & Vosoughi, S. (2024). MentalManip: A dataset for fine-grained analysis of mental manipulation in conversations. *arXiv preprint arXiv:2405.16584*.[[PDF](https://arxiv.org/pdf/2405.16584?)]2

Wang, X., Li, C., Chang, Y., Wang, J., & Wu, Y. (2024). Negativeprompt: Leveraging psychology for large language models enhancement via negative emotional stimuli. *arXiv preprint arXiv:2405.02814*.[[PDF](https://www.ijcai.org/proceedings/2024/0719.pdf)]2

Huang, J. T., Jiao, W., Lam, M. H., Li, E. J., Wang, W., & Lyu, M. (2024, November). On the reliability of psychological scales on large language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 6152-6173).[[PDF](https://aclanthology.org/2024.emnlp-main.354.pdf)]2

Yin, C., Li, F., Zhang, S., Wang, Z., Shao, J., Li, P., Chen, J., & Jiang, X. (2025). MDD-5k: A New Diagnostic Conversation Dataset for Mental Disorders Synthesized via Neuro-Symbolic LLM Agents. *Proceedings of the AAAI Conference on Artificial Intelligence*, *39*(24), 25715-25723. [[PDF](https://doi.org/10.1609/aaai.v39i24.34763)]2

Wang, R., Milani, S., Chiu, J. C., Zhi, J., Eack, S. M., Labrum, T., ... & Chen, Z. Z. (2024). Patient-{\Psi}: Using large language models to simulate patients for training mental health professionals. arXiv preprint arXiv:2405.19660.[[PDF](https://arxiv.org/pdf/2405.19660)]2

Hu, Y., Liu, D., Liu, B., Chen, Y., Cao, J., & Liu, Y. (2025, July). PsyAdvisor: A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conversations. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 12205-12229).[[PDF](https://aclanthology.org/2025.acl-long.596.pdf)]2

Han, J. E., Koh, J. S., Seo, H. T., Chang, D. S., & Sohn, K. A. (2024). PSYDIAL: Personality-based synthetic dialogue generation using large language models. *arXiv preprint arXiv:2404.00930*.[[PDF](https://arxiv.org/pdf/2404.00930)]2

Qiu, H., & Lan, Z. (2025, July). PsyDial: A Large-scale Long-term Conversational Dataset for Mental Health Support. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 21624-21655).[[PDF](https://aclanthology.org/2025.acl-long.1049.pdf)]2

Qiu, H., Ma, L., & Lan, Z. (2024). PsyGUARD: An automated system for suicide detection and risk assessment in psychological counseling. *arXiv preprint arXiv:2409.20243*.[[PDF](https://arxiv.org/pdf/2409.20243)]2

Qiu, H., He, H., Zhang, S., Li, A., & Lan, Z. (2023). Smile: Single-turn to multi-turn inclusive language expansion via chatgpt for mental health support. *arXiv preprint arXiv:2305.00450*.[[PDF](https://arxiv.org/pdf/2305.00450)]2

Lissak, S., Calderon, N., Shenkman, G., Ophir, Y., Fruchter, E., Klomek, A. B., & Reichart, R. (2024). The colorful future of llms: Evaluating and improving llms as emotional supporters for queer youth. *arXiv preprint arXiv:2402.11886*.[[PDF](https://arxiv.org/pdf/2402.11886)]2

Xie, H., Chen, Y., Xing, X., Lin, J., & Xu, X. (2024). Psydt: Using llms to construct the digital twin of psychological counselor with personalized counseling style for psychological counseling. *arXiv preprint arXiv:2412.13660*.[[PDF](https://arxiv.org/pdf/2412.13660)]2

Wang, J., Huang, Y., Liu, Z., Xu, D., Wang, C., Shi, X., Guan, R., Wang, H., Yue, W., & Huang, Y. (2025). STAMPsy: Towards SpatioTemporal-Aware Mixed-Type Dialogues for Psychological Counseling. *Proceedings of the AAAI Conference on Artificial Intelligence*, *39*(24), 25371-25379. [[PDF](https://doi.org/10.1609/aaai.v39i24.34725)]2

Yang, K., Ji, S., Zhang, T., Xie, Q., Kuang, Z., & Ananiadou, S. (2023). Towards interpretable mental health analysis with large language models. *arXiv preprint arXiv:2304.03347*.[[PDF](https://arxiv.org/pdf/2304.03347)]2

Meng, H., Chen, Y., Li, Y., Yang, Y., Lee, J., Zhang, R., & Lee, Y. C. (2025). What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma. *arXiv preprint arXiv:2505.12727*.[[PDF](https://arxiv.org/pdf/2505.12727)]2

**Closed-Source**

Varadarajan, V., Sikström, S., Kjell, O. N., & Schwartz, H. A. (2024, June). ALBA: Adaptive Language-Based Assessments for Mental Health. In *Proceedings of the conference. Association for Computational Linguistics. North American Chapter. Meeting* (Vol. 2024, p. 2466).[[PDF](https://aclanthology.org/2024.naacl-long.136.pdf)]1


Xiao, M., Ye, M., Liu, B., Zong, X., Li, H., Huang, J., ... & Peng, M. (2025). A Retrieval-Augmented Multi-Agent Framework for Psychiatry Diagnosis. *arXiv preprint arXiv:2506.03750*.[[PDF](https://arxiv.org/pdf/2506.03750)]1

Reuben, M., Slobodin, O., Elyshar, A., Cohen, I. C., Braun-Lewensohn, O., Cohen, O., & Puzis, R. (2024). Assessment and manipulation of latent constructs in pre-trained language models using psychometric scales. *arXiv preprint arXiv:2409.19655*.[[PDF](https://arxiv.org/pdf/2409.19655)]1

Ben-Zion, Z., Witte, K., Jagadish, A. K., Duek, O., Harpaz-Rotem, I., Khorsandian, M. C., ... & Spiller, T. R. (2025). Assessing and alleviating state anxiety in large language models. *npj Digital Medicine*, *8*(1), 132.[[PDF](https://www.nature.com/articles/s41746-025-01512-6.pdf)]1


Gabriel, Saadia, et al. "Can AI relate: Testing large language model response for mental health support." *arXiv preprint arXiv:2405.12021* (2024).[[PDF](https://arxiv.org/pdf/2405.12021)]1

Kallstenius, T., Capusan, A. J., Andersson, G., & Williamson, A. (2025). Comparing traditional natural language processing and large language models for mental health status classification: a multi-model evaluation. *Scientific Reports*, *15*(1), 1-13.[[PDF](https://www.nature.com/articles/s41598-025-08031-0.pdf)]1

Aragón, M. E., López-Monroy, A. P., González, L. C., Losada, D. E., & Montes, M. (2023, July). DisorBERT: A double domain adaptation model for detecting signs of mental disorders in social media. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 15305-15318).[[PDF](https://aclanthology.org/2023.acl-long.853.pdf)]1

Zhou, Y., Zhou, N., Chen, Q., Zhou, J., Zhou, A., & He, L. (2025). DiaCBT: A Long-Periodic Dialogue Corpus Guided by Cognitive Conceptualization Diagram for CBT-based Psychological Counseling. *arXiv preprint arXiv:2509.02999*[[PDF](https://arxiv.org/pdf/2509.02999?)]1

Sun, X., Pei, J., de Wit, J., Aliannejadi, M., Krahmer, E., Dobber, J. T., & Bosch, J. A. (2024, May). Eliciting motivational interviewing skill codes in psychotherapy with LLMs: A bilingual dataset and analytical study. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (pp. 5609-5621).[[PDF](https://aclanthology.org/2024.lrec-main.498.pdf)]1

Atapattu, T., Herath, M., Elvitigala, C., de Zoysa, P., Gunawardana, K., Thilakaratne, M., ... & Falkner, K. (2022). Emoment: An emotion annotated mental health corpus from two south asian countries. *arXiv preprint arXiv:2208.08486*.[[PDF](https://arxiv.org/pdf/2208.08486)]1

Chen, Y., Wang, H., Yan, S., Liu, S., Li, Y., Zhao, Y., & Xiao, Y. (2024). Emotionqueen: A benchmark for evaluating empathy of large language models. *arXiv preprint arXiv:2409.13359*.[[PDF](https://arxiv.org/pdf/2409.13359)]1

Chen, Z., Lu, Y., & Wang, W. Y. (2023). Empowering psychotherapy with large language models: Cognitive distortion detection through diagnosis of thought prompting. *arXiv preprint arXiv:2310.07146*.[[PDF](https://arxiv.org/pdf/2310.07146)]1

Kim, H., Lee, S., Cho, Y., Ryu, E., Jo, Y., Seong, S., & Cho, S. (2025). KMI: A Dataset of Korean Motivational Interviewing Dialogues for Psychotherapy. *arXiv preprint arXiv:2502.05651*.[[PDF](https://arxiv.org/pdf/2502.05651)]1

Gao, S., Yu, K., Yang, Y., Yu, S., Shi, C., Wang, X., ... & Zhu, H. (2025). Large language model powered knowledge graph construction for mental health exploration. *Nature Communications*, *16*(1), 7526.[[PDF](https://www.nature.com/articles/s41467-025-62781-z.pdf)]1

Sahu, N. K., Yadav, M., Chaturvedi, M., Gupta, S., & Lone, H. R. (2025, January). Leveraging language models for summarizing mental state examinations: A comprehensive evaluation and dataset release. In *Proceedings of the 31st International Conference on Computational Linguistics* (pp. 2658-2682).[[PDF](https://aclanthology.org/2025.coling-main.182.pdf)]1

Bi, G., Chen, Z., Liu, Z., Wang, H., Xiao, X., Xie, Y., ... & Huang, M. (2025). MAGI: Multi-Agent Guided Interview for Psychiatric Assessment. *arXiv preprint arXiv:2504.18260*.[[PDF](https://arxiv.org/pdf/2504.18260?)]1

Kim, J., Ma, S.P., Chen, M.L. *et al.* Optimizing large language models for detecting symptoms of depression/anxiety in chronic diseases patient communications. *npj Digit. Med.* **8**, 580 (2025). [[PDF](https://doi.org/10.1038/s41746-025-01969-5)]1

Chhikara, P., Pasupulety, U., Marshall, J., Chaurasia, D., & Kumari, S. (2023). Privacy aware question-answering system for online mental health risk assessment. *arXiv preprint arXiv:2306.05652*.[[PDF](https://arxiv.org/pdf/2306.05652)]1

Yang, Q., Wang, Z., Chen, H., Wang, S., Pu, Y., Gao, X., ... & Huang, G. (2024). Psychogat: A novel psychological measurement paradigm through interactive fiction games with llm agents. *arXiv preprint arXiv:2402.12326*.[[PDF](https://arxiv.org/pdf/2402.12326)]1

Wang, J., Wang, B., Fu, X., Sun, Y., Zhao, Y., & Qin, B. (2025). Psychological Counseling Cannot Be Achieved Overnight: Automated Psychological Counseling Through Multi-Session Conversations. *arXiv preprint arXiv:2506.06626*.[[PDF](https://arxiv.org/pdf/2506.06626)]1

Sun, X., Tang, X., Ali, A. E., Li, Z., Ren, P., de Wit, J., ... & Bosch, J. A. (2024). Rethinking the alignment of psychotherapy dialogue generation with motivational interviewing strategies. *arXiv preprint arXiv:2408.06527*.[[PDF](https://arxiv.org/pdf/2408.06527?)]1

Hengle, A., Kulkarni, A., Patankar, S., Chandrasekaran, M., D'Silva, S., Jacob, J., & Gupta, R. (2024). Still not quite there! Evaluating large language models for comorbid mental health diagnosis. *arXiv preprint arXiv:2410.03908*.[[PDF](https://arxiv.org/pdf/2410.03908?)]1

Ye, J., Xiang, L., Zhang, Y., & Zong, C. (2024). SweetieChat: A strategy-enhanced role-playing framework for diverse scenarios handling emotional support agent. *arXiv preprint arXiv:2412.08389*.[[PDF](https://arxiv.org/pdf/2412.08389)]1

Mori, S., Ignat, O., Lee, A., & Mihalcea, R. (2024). Towards algorithmic fidelity: mental health representation across demographics in synthetic vs. human-generated data. *arXiv preprint arXiv:2403.16909*.[[PDF](https://arxiv.org/pdf/2403.16909)]1

Chen, Z., Cao, Y., Bi, G., Wu, J., Zhou, J., Xiao, X., Chen, S., Wang, H., & Huang, M. (2025). SocialSim: Towards Socialized Simulation of Emotional Support Conversation. *Proceedings of the AAAI Conference on Artificial Intelligence*, *39*(2), 1274-1282. [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/32116)]1

Xu, B., Li, L., Wang, J., Qiao, X., Yu, E., Qian, Y., ... & Lin, H. Unveiling Maternity and Infant Care Conversations: A Chinese Dialogue Dataset for Enhanced Parenting Support.[[PDF](https://www.ijcai.org/proceedings/2025/0923.pdf)]1

Balan, R., Dobrean, A., & Poetar, C. R. (2024). Use of automated conversational agents in improving young population mental health: a scoping review. *NPJ Digital Medicine*, *7*(1), 75.[[PDF](https://www.nature.com/articles/s41746-024-01072-1.pdf)]1

Shu, B., Zhang, L., Choi, M., Dunagan, L., Logeswaran, L., Lee, M., ... & Jurgens, D. (2023). You don't need a personality test to know these models are unreliable: Assessing the Reliability of Large Language Models on Psychometric Instruments. *arXiv preprint arXiv:2311.09718*.[[PDF](https://arxiv.org/pdf/2311.09718)]1

## Clinical application

**Open-Source**

Mishra, K., Priya, P., & Ekbal, A. (2023). Help Me Heal: A Reinforced Polite and Empathetic Mental Health and Legal Counseling Dialogue System for Crime Victims. *Proceedings of the AAAI Conference on Artificial Intelligence*, *37*(12), 14408-14416.[[PDF](https://doi.org/10.1609/aaai.v37i12.26685)]2

Hur, J. K., Heffner, J., Feng, G. W., Joormann, J., & Rutledge, R. B. (2024). Language sentiment predicts changes in depressive symptoms. *Proceedings of the National Academy of Sciences*, *121*(39), e2321321121.[[PDF](https://www.pnas.org/doi/pdf/10.1073/pnas.2321321121)]2

**Closed-Source**

Kim, J., Leonte, K. G., Chen, M. L., Torous, J. B., Linos, E., Pinto, A., & Rodriguez, C. I. (2024). Large language models outperform mental and medical health care professionals in identifying obsessive-compulsive disorder. *NPJ Digital Medicine*, *7*(1), 193.[[PDF](https://www.nature.com/articles/s41746-024-01181-x.pdf)]1

Siddals, S., Torous, J., & Coxon, A. (2024). “It happened to be the perfect thing”: experiences of generative AI chatbots for mental health. *npj Mental Health Research*, *3*(1), 48.[[PDF](https://www.nature.com/articles/s44184-024-00097-4.pdf)]1

Maples, B., Cerit, M., Vishwanath, A., & Pea, R. (2024). Loneliness and suicide mitigation for students using GPT3-enabled chatbots. *npj mental health research*, *3*(1), 4.[[PDF](https://www.nature.com/articles/s44184-023-00047-6.pdf)]1

Priya, P., Mishra, K., Totala, P., & Ekbal, A. (2023, August). PARTNER: A Persuasive Mental Health and Legal Counselling Dialogue System for Women and Children Crime Victims. In *IJCAI* (pp. 6183-6191).[[PDF](https://www.ijcai.org/proceedings/2023/686)]1

Gollapalli, S. D., & Ng, S. K. (2025, January). PIRsuader: A persuasive chatbot for mitigating psychological insulin resistance in type-2 diabetic patients. In *Proceedings of the 31st International Conference on Computational Linguistics* (pp. 5997-6013).[[PDF](https://aclanthology.org/2025.coling-main.401.pdf)]1

Chen, C. H., Wu, C. S., Su, C. H., & Chen, H. H. TimelyMed: AI-Driven Clinical Course Attribution and Temporal Mapping for Psychiatric Medical Records.[[PDF](https://www.ijcai.org/proceedings/2025/1252.pdf)]1





