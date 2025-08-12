Dataset Augmentation with Pegasus Paraphrasing
This script enhances a sentiment analysis dataset by automatically generating paraphrased versions of each positive and negative review using the Pegasus transformer model (tuner007/pegasus_paraphrase).

Key Steps:

Load existing positive and negative reviews from CSV files.

Use Pegasus to generate multiple paraphrases per review.

Augment the original dataset with these new paraphrased entries.

Label and save the final datasets for model training.

Goal: Improve model performance by increasing dataset size and diversity through automatic data augmentation.