# Project Name : DOF1.0 -  Empowering Datasets with Language Modeling 
###  Study Name  : Exploring the landscape for generative models for specialized data generation
The project focuses on creating a generative AI model to generate realistic flow-based network traffic for improving network-based intrusion detection systems. By leveraging language modeling techniques, this project aims to overcome the limitations of traditional methods and generate high-dimensional datasets that are difficult to collect. The model was trained on 30,000 packets of network traffic.
## Repository Structure

```plaintext
├── .gitignore
├── Models/
│   ├── RNN_Model.ipynb
│   ├── Transformer_Model.py
│   └── Wavenet_Model.ipynb
├── README.md
├── Tests/
│   ├── Poly_Kernel_Testing.py
│   ├── driver_transformer_testing.py
│   └── polynomial_testing.ipynb
├── data_processing/
│   ├── data_preprocessing_main.ipynb
│   └── make_dataset_with_no_dots.ipynb
├── datasets/
│   ├── dataset _Main.csv
│   ├── mapped_words_main.txt
│   └── mapped_words_with-dots.txt
├── litreture/
│   ├── A Neural Probabilistic Language Model.pdf
│   ├── Batchnormalization.pdf
│   ├── He_Delving_Deep_into_ICCV_2015_paper.pdf
│   ├── RethinkingBatch.pdf
│   └── attention.pdf
├── outputs/
│   ├── output_text_RNN.txt
│   └── output_text_transformer.txt
└── reports/
    └── main_report.pdf

