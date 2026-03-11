# Introduction

The rapid and accurate identification of bacterial pathogens is a cornerstone of clinical microbiology and infectious disease management. Gram staining, a century-old diagnostic technique, remains the first and most crucial step in differentiating bacteria based on their cell wall properties (Gram-positive vs. Gram-negative) and morphology (cocci vs. bacilli). This initial classification is critical for guiding empirical antimicrobial therapy, particularly in life-threatening situations like sepsis, where every hour of delay in administering the correct antibiotic can significantly increase mortality rates [1]. In countries like India, the burden of infectious diseases is disproportionately high, and access to trained microbiologists can be limited in remote or under-resourced settings, making rapid, automated diagnostic tools a national health priority [2].

Despite its importance, the manual interpretation of Gram-stained slides presents several challenges. The process is labor-intensive, time-consuming, and highly dependent on the subjective judgment of experienced microbiologists. This can lead to inter-observer variability and diagnostic delays, undermining its utility for rapid patient care. The need for an objective, standardized, and efficient alternative has driven interest in automated image analysis.

In recent years, deep learning (DL) has emerged as a powerful tool for medical image analysis. Several studies have successfully applied Convolutional Neural Networks (CNNs) to related diagnostic problems. For instance, multiple research groups have developed models to differentiate bacterial from fungal keratitis using slit-lamp images of the eye, often achieving diagnostic accuracies superior to those of trained ophthalmologists [3, 4, 5]. However, these studies focus on a specific clinical syndrome and imaging modality. A more fundamental approach involves the direct classification of Gram stain images. Early work by Smith et al. demonstrated the potential of CNNs to classify blood culture Gram stains into three categories with high accuracy [6]. More recent studies have explored enhanced CNN architectures and even "virtual staining" techniques using DL [7, 8].

While these studies have established the feasibility of automated Gram stain analysis, they also highlight existing challenges. Many rely on traditional CNN architectures, and performance can be inconsistent across different bacterial classes, often due to the subtle variations in stain uptake and morphology. Furthermore, some studies report exceptionally high accuracy but may lack the robust validation needed to ensure the model generalizes well to new, unseen data.

This study aims to address these gaps by developing a highly robust and accurate model for the automated four-class classification of Gram-stained bacteria: Gram-Positive Cocci (GPC), Gram-Positive Bacilli (GPB), Gram-Negative Cocci (GNC), and Gram-Negative Bacilli (GNB). Our primary contribution is the systematic evaluation of modern deep learning architectures for this task. The novelty of our approach is twofold. First, we conducted a comprehensive comparison of 24 different architectures, including both conventional CNNs and newer Vision Transformer (ViT) models. Second, we identify and validate the superiority of the Vision Outlooker (VOLO) architecture, a ViT variant, which has not been extensively applied to this specific problem. We demonstrate through rigorous 5-fold cross-validation that our final VOLO-based model achieves state-of-the-art performance with high consistency and reliability.

The successful development and deployment of such a system could revolutionize microbiology workflows. By providing a rapid, accurate, and objective classification of bacteria from digital slide images, our model has the potential to decrease diagnostic turnaround times, reduce human error, and ultimately improve patient outcomes by enabling earlier, more targeted antibiotic therapy.

# References

[1] Kumar, A., Roberts, D., Wood, K. E., Light, B., Parrillo, J. E., Sharma, S., ... & Cheang, M. (2006). Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. _Critical care medicine_, 34(6), 1589-1596.

[2] Laxminarayan, R., & Chaudhury, R. R. (2016). Antibiotic Resistance in India: Drivers and Opportunities for Action. _PLoS Medicine_, 13(3), e1001974.

[3] Hung, N., Shih, Y. L., Chen, Y. T., Weng, C. C., Chen, W. L., & Chen, H. C. (2021). Using Slit-Lamp Images for Deep Learning-Based Identification of Bacterial and Fungal Keratitis: Model Development and Validation with Different Convolutional Neural Networks. _Diagnostics_, 11(7), 1246.

[4] Redd, T. K., Prajna, L., Niziol, L. M., Lalitha, P., Krishnan, T., Srinivasan, M., ... & Woodward, M. A. (2022). Image-Based Differentiation of Bacterial and Fungal Keratitis Using Deep Convolutional Neural Networks. _Ophthalmology Science_, 2(2), 100119.

[5] Zhang, Z., Zhao, Y., Lin, Y., Zhou, X., Chen, Z., Chen, W., ... & Liu, Z. (2022). Deep learning-based classification of infectious keratitis on slit-lamp images. _Therapeutic Advances in Chronic Disease_, 13, 20406223221136071.

[6] Smith, K. P., Kang, A. D., Kirby, J. E. (2018). Automated Interpretation of Blood Culture Gram Stains by Use of a Deep Convolutional Neural Network. _Journal of Clinical Microbiology_, 56(7), e00323-18.

[7] Nguyen, T., Chtar, M., Sinton, D. & Alland, D. (2025). Virtual Gram staining of label-free bacteria using darkfield microscopy and deep learning. _bioRxiv_.

[8] Al-Hjaily, M., Al-Nahari, A., Al-Subaihi, A., & Al-Raeei, M. (2026). A Deep Learning Approach for Gram-Stained Bacterial Classification Using an Enhanced Hybrid Deep Convolutional Neural Network. _Journal of Imaging_, 9(3), 58.
