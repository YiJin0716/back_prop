# An Interpretable Coarse-to-Fine Joint Model for Whole-CT Pulmonary Nodule Analysis

## 面向全 CT 肺结节分析的可解释粗到细联合模型

> Draft status: literature-informed abstract, 14 September 2026. The optimized model has not yet been trained, so the bracketed results below are intentional placeholders, not claims. This draft concerns LIDC-IDRI reader ratings; its scan-level target is **not** pathology-confirmed cancer or prospective cancer risk.

## Defensible research gap

This wording is based on a targeted audit of primary literature through 14 September 2026; it is not a formal systematic review.

Low-dose CT screening reduces lung-cancer mortality, and deep learning has made substantial progress in whole-volume cancer prediction, pulmonary-nodule detection, segmentation, and malignancy assessment.[1–7] Interpretable nodule models have also combined segmentation with semantic attributes or malignancy prediction, including PN-SAMP, two-stage multitask U-Net, PK-MTL, MT-Net, LNMSNet, and DPMT-Net.[8–13] More recent systems go further: 3D-NoduleNet links 3D detection, ROI segmentation, and nodule classification; Seg-CADe-CADx combines segmentation-guided detection, 2.5D refinement, radiological attributes, late-fused radiomics, and scan-level pooling; and VITALIS jointly models whole-volume nodule detection, malignancy, nodule count, and patient survival risk from multimodal imaging and structured text.[14,15,22]

Nodule interpretation is also intrinsically subjective. LIDC-IDRI deliberately retained four radiologists’ independent final contours and characteristic ratings without forcing consensus, and a subsequent analysis of 1,880 nodules found substantial inter-reader disagreement across diagnostic traits, including malignancy.[2,25] A single scan-level risk score therefore has limited value as a stand-alone decision aid: it collapses disagreement and does not reveal which nodule, boundary, or imaging characteristics support the prediction. Returning per-nodule masks and quantitative and semantic features is intended to provide inspectable evidence with which to assess that score.

Accordingly, this work should **not** claim to be the first joint detection–segmentation–classification model, the first segmentation–attribute–malignancy model, the first whole-volume detection–patient-outcome model, the first noisy-OR scan aggregator, or the first radiomics/deep-learning fusion model. The defensible gap is narrower and stronger:

> To our knowledge, among the studies surveyed through 14 September 2026, no single CT-only system combines both principal advantages of the proposed model: (1) a loss-connected training path in which nodule- and scan-level objectives back-propagate through explicit radiomic and semantic prediction to soft instance masks and the upstream trainable modules, and (2) identity-preserving, interpretable per-nodule outputs—non-downsampled 1-mm 3D masks, 64 quantitative descriptors, and the seven targeted ordinal LIDC ratings—alongside transparent multi-nodule scan aggregation.

PN-SAMP is especially important prior art: its semantic loss already propagates through the predicted segmentation, but the model starts from a prelocalized 64³ nodule patch and has neither autonomous whole-CT detection nor explicit radiomics or scan aggregation.[8] VITALIS spans whole-volume detection and patient endpoints, but its reported detector is box-based and does not expose 1-mm nodule instance masks or an explicit per-nodule radiomic–ordinal evidence chain.[22] Seg-CADe-CADx functionally covers many neighboring components, yet hard thresholding, connected components/NMS, handcrafted radiomics, and separately optimized stages interrupt the downstream-to-mask gradient path.[15] Other close pipelines retain at least one such limitation.[5,7,9–14,23,24] VISTA3D and DETR provide suitable foundations for dense 3D screening and one-to-one set prediction, respectively, but neither supplies the complete chain by itself.[16,17] The 64-feature Raicu/BRISC lineage provides explicit 2D shape, size, intensity, Haralick, Gabor, and Markov descriptors linked to LIDC semantics; conventional radiomics systems likewise provide auditable descriptors, but normally assume a predefined ROI and are not themselves a mask-learning mechanism.[18–21]

## English abstract

Low-dose computed tomography (LDCT) screening reduces lung-cancer mortality, but each examination may contain multiple nodules requiring localization, delineation, and characterization. Deep learning has advanced whole-scan prediction and nodule detection, segmentation, and classification, while semantic attributes and radiomics offer interpretable evidence.

However, these capabilities remain fragmented. Interpretable multitask models commonly assume a prelocalized nodule, whereas whole-volume systems usually lack voxel-level instances or an explicit radiomic–ordinal evidence chain; hard candidate selection and late fusion also interrupt downstream-to-mask gradients. Moreover, nodule assessment is subjective: radiologists can assign different contours, semantic traits, and malignancy ratings to the same lesion. A single opaque risk score is therefore insufficient as a stand-alone decision aid because it hides which nodule and which imaging evidence drove the prediction.

We address both limitations with an identity-preserving, coarse-to-fine 3D framework mapping an entire 1-mm isotropic CT to nodule instances and a reader-derived scan-level malignancy surrogate. Sliding-window VISTA3D supplies a dense prior; a 24-query 3D detection transformer predicts objects, boxes, coarse masks, and query embeddings; and a query-conditioned FiLM refiner processes each non-downsampled 128³ proposal crop. The framework has two central advantages. First, it supports full-pipeline back-propagation across all differentiable modules: nodule, ordinal, and scan losses flow through noisy-OR aggregation, dense proportional-odds heads, a surrogate-gradient Raicu/BRISC radiomics bottleneck, the fine soft mask, and the trainable DETR/VISTA pathway. Because matching and crop coordinates are discrete, proposal geometry is trained by explicit object, box, and mask losses. Second, prediction does not end at one risk score: every detected nodule yields an inspectable 1-mm 3D mask, 64 quantitative descriptors, seven LIDC ratings, and a malignancy probability.

After prespecified exclusion of physical nodules whose mean reader malignancy rating was exactly 3, the held-out LIDC-IDRI fold contained 166 evaluable scans, 423 physical nodules, and 1,199 retained reader annotations. The model achieved **[proposal sensitivity at X false positives/scan]**, **[1-mm Dice/surface Dice/HD95]**, **[ordinal-rating MAE]**, **[nodule AUROC/AUPRC]**, and **[scan AUROC/Brier score]**. Scan-level discrimination used the 137 scans with an unambiguous retained target (83 positive); negative scans that still contained an excluded indeterminate nodule were masked. Paired ablations showed **[effect sizes and 95% CIs]** for 1-mm refinement and for allowing diagnostic losses to back-propagate through the radiomic–semantic mask pathway.

By combining loss-connected training with interpretable per-nodule by-products, the framework turns a scan score into an auditable chain of masks, descriptors, and reader-aligned semantic predictions designed to support rather than obscure clinical judgment.

## 中文摘要

低剂量计算机断层扫描（LDCT）筛查能降低肺癌死亡率，但每套 CT 可包含多个需要定位、勾画和表征的结节。深度学习已推进整体扫描预测以及结节检测、分割和分类，而语义属性与 radiomics 可提供可解释证据。

然而，这些能力仍然割裂。市面上已存在的可解释的多任务模型常常没有结节识别功能，这意味着它需要结节位置或结节mask作为输入；并且它们只是简单地把几个模型拼接到一起，各个模型之前不存在监督及联动。与之相对的，市面上的整幅 CT 模型通常缺少体素级mask和semantic feature等输出；hard mask还会中断“下游诊断—segmentation”的梯度。这些mask与features几乎与最后的predicted risk同样重要，因为结节判断具有主观性：不同医生可能对同一结节给出不同的边界、语义属性和恶性度评分。因此，单一且不透明的 risk score 不足以独立支持判断，因为它隐藏了哪个结节mask以及哪些影像证据驱动了预测。

为同时解决两个问题，我们提出一个whole CT的完整框架，它能够解决上述两个模型的不足之处：我们的模型首先将whole CT送入segmentation model得到整个CT的nodule mask，然后利用多通道的深度学习模型将whole CT mask拆分成各个nodule的segmentation。随后对每一个nodule进行feature extraction，得到它们的radiomics feature和semantic feature用于之后的malignancy prediction。我们的模型能够做到几乎全流程的梯度back prop，这意味着下游任务能够为上游任务提供监督，这将使我们的模型表现优于市面上简单拼接的可解释多任务模型。同时我们的模型在可解释性上能够击败那些纯粹的black box deep learning model，因为我们的模型能够输出每个nodule的mask以及features，这对临床诊断非常重要。

## Five-part structure

1. **And / Once upon a time:** paragraph 1 establishes why LDCT nodule analysis matters and what existing methods already do well.
2. **But / Villain:** paragraph 2 identifies fragmentation, broken gradient paths, inter-reader subjectivity, and the insufficiency of an opaque risk score.
3. **Therefore / Hero:** paragraph 3 presents the model and foregrounds its two advantages: loss-connected training and interpretable per-nodule outputs.
4. **Therefore 2 / Villain defeated:** paragraph 4 is a results-ready template with prespecified metrics; it must remain bracketed until experiments finish.
5. **Therefore 3 / Happily ever after:** paragraph 5 states the broader value of an auditable nodule-to-scan evidence chain.

## Terminology and claim guardrails

- **“Scan-level malignancy surrogate,” not “patient cancer risk.”** LIDC-IDRI contains radiologists’ subjective malignancy ratings and nodule outlines, not a pathology-confirmed prospective outcome for every case.[2]
- **“Full-pipeline back-propagation” means loss-connected training across all differentiable modules, not that every operation has a derivative.** Nodule, ordinal, and scan losses can update the soft fine mask, refiner, shared DETR pathway, and unfrozen VISTA3D parameters. Hungarian assignment, integer ROI routing, the exact hard-mask radiomic forward pass, and some inference decisions remain discrete; explicit object, box, and mask losses train the affected proposal geometry.
- **“64 paper-compatible 2D descriptors,” not “64 volumetric IBSI features.”** The current extractor operates on a representative axial section of the 3D fine mask and follows the Raicu/BRISC feature families.[18,19] IBSI compliance should not be claimed unless phantom benchmarks are run and passed.[21]
- **Inspectable evidence does not by itself model reader disagreement.** The current targets aggregate readers at the physical-nodule level, and the ordinal heads return one distribution per trait. Masks and features expose evidence for review, but explicit reader-specific or disagreement-aware modeling would be a separate extension.
- **“To our knowledge, the surveyed literature does not combine …,” not an unqualified “first.”** The combination claim should be rerun immediately before manuscript submission because this is an active research area.
- The bracketed performance sentence must be populated only from the frozen held-out test analysis; thresholds and model selection must be fixed on training/validation data.

## Sources

1. National Lung Screening Trial Research Team. “Reduced Lung-Cancer Mortality with Low-Dose Computed Tomographic Screening.” *New England Journal of Medicine* 365, 395–409 (2011). https://doi.org/10.1056/NEJMoa1102873
2. Armato SG III et al. “The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans.” *Medical Physics* 38, 915–931 (2011). https://doi.org/10.1118/1.3528204
3. Ardila D et al. “End-to-end Lung Cancer Screening with Three-dimensional Deep Learning on Low-dose Chest Computed Tomography.” *Nature Medicine* 25, 954–961 (2019). https://doi.org/10.1038/s41591-019-0447-x
4. Mikhael PG et al. “Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest Computed Tomography.” *Journal of Clinical Oncology* 41, 2191–2200 (2023). https://doi.org/10.1200/JCO.22.01345
5. Liao F et al. “Evaluate the Malignancy of Pulmonary Nodules Using the 3-D Deep Leaky Noisy-OR Network.” *IEEE Transactions on Neural Networks and Learning Systems* 30, 3484–3495 (2019). https://doi.org/10.1109/TNNLS.2019.2892409
6. Tang H et al. “NoduleNet: Decoupled False Positive Reduction for Pulmonary Nodule Detection and Segmentation.” *MICCAI* (2019). https://doi.org/10.1007/978-3-030-32226-7_30
7. Liu C and Chan S-C. “A Joint Detection and Recognition Approach to Lung Cancer Diagnosis From CT Images With Label Uncertainty.” *IEEE Access* 8, 228905–228921 (2020). https://doi.org/10.1109/ACCESS.2020.3044941
8. Wu B et al. “Joint Learning for Pulmonary Nodule Segmentation, Attributes and Malignancy Prediction.” *IEEE ISBI* (2018). https://arxiv.org/abs/1802.03584
9. Ni Y et al. “Two-stage Multitask U-Net Construction for Pulmonary Nodule Segmentation and Malignancy Risk Prediction.” *Quantitative Imaging in Medicine and Surgery* 12, 292–309 (2022). https://doi.org/10.21037/qims-21-19
10. Xue P et al. “Prior Knowledge-based Multi-task Learning Network for Pulmonary Nodule Classification.” *Computerized Medical Imaging and Graphics* 121, 102511 (2025). https://doi.org/10.1016/j.compmedimag.2025.102511
11. Tang T and Zhang R. “A Multi-Task Model for Pulmonary Nodule Segmentation and Classification.” *Journal of Imaging* 10, 234 (2024). https://doi.org/10.3390/jimaging10090234
12. Liu Y et al. “LNMSNet: A Multi-task Deep Learning Network for Pulmonary Nodules Segmentation and Malignancy Classification.” *Frontiers in Medicine* (2026). https://doi.org/10.3389/fmed.2026.1773338
13. Sutradhar D et al. “DPMT-Net: A Dual-prompt Attention Multi-task 3D Network for Joint Nodule Segmentation and Malignancy Prediction in Lung CT Scans.” *Complex & Intelligent Systems* 12, 200 (2026). https://doi.org/10.1007/s40747-026-02337-w
14. Yu H, Dai Q, and Wu Y. “3D-NoduleNet: A Comprehensive Framework for Benign and Malignant Pulmonary Nodule Classification.” *Applied Soft Computing* 191, 114645 (2026). https://doi.org/10.1016/j.asoc.2026.114645
15. Subramanyam GK et al. “Segmentation-Guided Hybrid Deep Learning for Pulmonary Nodule Detection and Risk Prediction from Multi-Cohort CT Images.” *Diseases* 14, 21 (2026). https://doi.org/10.3390/diseases14010021
16. He Y et al. “VISTA3D: A Unified Segmentation Foundation Model for 3D Medical Imaging.” *CVPR* (2025). https://openaccess.thecvf.com/content/CVPR2025/html/He_VISTA3D_A_Unified_Segmentation_Foundation_Model_For_3D_Medical_Imaging_CVPR_2025_paper.html
17. Carion N et al. “End-to-End Object Detection with Transformers.” *ECCV* (2020). https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/123460205.html
18. Raicu DS et al. “Modelling Semantics from Image Data: Opportunities from LIDC.” *International Journal of Biomedical Engineering and Technology* 3, 83–113 (2010). https://doi.org/10.1504/IJBET.2010.029653
19. Lam MO et al. “BRISC—An Open Source Pulmonary Nodule Image Retrieval Framework.” *Journal of Digital Imaging* 20, 63–71 (2007). https://doi.org/10.1007/s10278-007-9059-y
20. van Griethuysen JJM et al. “Computational Radiomics System to Decode the Radiographic Phenotype.” *Cancer Research* 77, e104–e107 (2017). https://doi.org/10.1158/0008-5472.CAN-17-0339
21. Zwanenburg A et al. “The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping.” *Radiology* 295, 328–338 (2020). https://doi.org/10.1148/radiol.2020191145
22. Zhao D et al. “Graphicalized Vision-Language Modeling for Comprehensive Lung Nodule Analysis and Risk Stratification.” *npj Digital Medicine* 9, 442 (2026). https://doi.org/10.1038/s41746-026-02602-9
23. Ozdemir O et al. “A 3D Probabilistic Deep Learning System for Detection and Diagnosis of Lung Cancer Using Low-Dose CT Scans.” *IEEE Transactions on Medical Imaging* 39, 1419–1429 (2020). https://doi.org/10.1109/TMI.2019.2947595
24. Zheng S et al. “Interpretative Computer-aided Lung Cancer Diagnosis: From Radiology Analysis to Malignancy Evaluation.” *Computer Methods and Programs in Biomedicine* 210, 106363 (2021). https://doi.org/10.1016/j.cmpb.2021.106363
25. Lin H et al. “Measuring Interobserver Disagreement in Rating Diagnostic Characteristics of Pulmonary Nodule Using the Lung Imaging Database Consortium and Image Database Resource Initiative.” *Academic Radiology* 24, 401–410 (2017). https://doi.org/10.1016/j.acra.2016.11.022
