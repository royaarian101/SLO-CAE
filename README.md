# SLO-CAE
**First-Time Utilization of SLO Images for the Classification of Individuals with Multiple Sclerosis and Healthy Individuals**

# Introduction
For the first time, we employed SLO images to distinguish between eyes affected by Multiple Sclerosis (MS) and healthy controls (HC), achieving promising results through the combined use of a custom-designed Convolutional Autoencoder (CAE) and a Multi-Layer Perceptron (MLP). The proposed CAE's effectiveness in detecting more informative representations can be partly attributed to the connections between two blocks of the encoder and decoder. This design facilitates information propagation between the encoder and decoder components, helping to prevent the loss of critical information during the up-sampling process, as a higher-resolution feature map is constructed and subsequently processed by the decoder's convolutional layers.


# Dara Prepration
To create a dictionary for your dataset, each key should represent an individual patient, with the corresponding value being a nested dictionary containing the number of images for that patient and their corresponding numpy arrays. Save this dictionary as a pickle file named "subjects_slo_data.pkl." In addition, create a label dictionary where the keys match those in the image dictionary, and the values represent each patient's label. Save this dictionary as a pickle file named "labels_slo_data.pkl."


For example:

****Patient 1 has 4 images with label = 1:

images[0] is a dictionary with size (4):

np.shape(images[0][0]) = (128 × 128 × 1)

np.shape(images[0][1]) = (128 × 128 × 1)

np.shape(images[0][2]) = (128 × 128 × 1)

np.shape(images[0][3]) = (128 × 128 × 1)


labels_train[0] = 1










****Patient 2 has 2 images with label = 0:


images[1] is a dictionary with size (2):

np.shape(images[1][0]) = (128 × 128 × 1)

np.shape(images[1][1]) = (128 × 128 × 1)

labels_train[1] = 0


Ensure that all images are resized to a square dimension of (128 × 128 × 1). To extract features using the proposed Conventional Autoencoder (CAE), run the file "feature_extraction.py." This code will save the extracted features for further processing. To complete the classification using the proposed method, execute the code "mlp_classification."





# Citing
**Please ensure to include the following citations when utilizing any part of the code:**

[1] Arian, R., Aghababaei, A., Soltanipour, A., Khodabandeh, Z., Rakhshani, S., Iyer, S. B., Ashtari, F., Rabbani, H., & Kafieh, R. (2024). SLO-net: Enhancing multiple sclerosis diagnosis beyond optical coherence tomography using infrared reflectance scanning laser ophthalmoscopy images. Translational Vision Science & Technology, 13(7), 13. https://doi.org/10.1167/tvst.13.7.13

[2] Aghababaei A, Arian R, Soltanipour A, Ashtari F, Rabbani H, Kafieh R. Discrimination of Multiple Sclerosis using Scanning Laser Ophthalmoscopy Images with Autoencoder-Based Feature Extraction. Multiple Sclerosis and Related Disorders. 2024 Aug 1;88:105743–3.
