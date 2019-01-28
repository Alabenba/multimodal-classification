import image_feature_extraction
import image_calculate_mean_vectors_for_the_same_recipe_images
import text_prepare_texts_for_doc2vec
import text_doc2vec_v2
import text_doc_embeddings_visualisation
import gaited_multimodal_unit

image_feature_extraction.extract_features(image_dir='data/images', models_filename='data/v8_vgg16_model_1.h5')
image_calculate_mean_vectors_for_the_same_recipe_images.calculate_means(features_filename='data/images/extracted_features.pkl', 
    output_filename='data/images/mean_feature_vectors.pkl')

text_prepare_texts_for_doc2vec.prepare_texts(path_to_raw_texts='data/texts/raw_texts', 
    preprocessed_texts_file='data/texts/preprocessed_texts_for_doc2vec.pkl')
text_doc2vec_v2.doc2vec(models_folder_name='data/texts/models', path_to_preprocessed_texts='data/texts/preprocessed_texts_for_doc2vec.pkl')
text_doc_embeddings_visualisation.embed(path_to_preprocessed_texts='data/texts/preprocessed_texts_for_doc2vec.pkl',
    path_to_saved_model='data/texts/models/doc2vec_recipes_checkpoint.ckpt',
    path_to_save_docs_embeddings_pkl='data/texts/docs_extracted_features.pkl')

gaited_multimodal_unit.classify(visual_m_path='data/images/mean_feature_vectors.pkl',
    textual_m_path='data/texts/docs_extracted_features.pkl')