### **AI Models for Legal Text Processing**

Explore our specialized repository designed for training state-of-the-art AI models on legal text data. We focus on key tasks such as text classification, text generation, and text summarization using BERT, GPT-2, and PEGASUS.

#### **Our Cutting-Edge Models Include:**

- **BERT**: Specialized in sequence classification and sentiment analysis.
- **GPT-2**: Designed for advanced text generation.
- **PEGASUS**: Expert at concise text summarization.

#### **Training Datasets:**

The models are fine-tuned on diverse legal datasets:
- **ILDC Dataset**: Used for BERT in text classification.
- **Legal Contracts Dataset**: The basis for GPT-2's text generation capabilities.
- **Indian Legal Documents**: Aids PEGASUS in summarization tasks.

#### **GPT-2 Text Generation Process:**

- Make sure to install necessary packages through ```pip install -r requirements.txt```

1. **Dataset Acquisition**:
   Download the legal contracts dataset with:
   ```bash
   python data.py --percent <percentage of data>
   ```

2. **Fine-Tuning the Model**:
   Enhance GPT-2's performance using:
   ```bash
   python Fine_Tuning_GPT2.py --dataset <path to dataset>
   ```

3. **Generating Text**:
   Utilize the trained GPT-2 model for text generation:
   ```bash
   python Inference_Text_Generation.py --model-path <trained GPT2 model path>
   ```
#### **Pegasus Text Summarization Process:**
   ```bash
   cd Code/Summarization/
   python3 train.py
   ```
#### **Text Classification:**
    ```bash
      cd Code/Sentiment/
      python3 Bert_Tuned.py
    ```
#### **Running Streamlit app:**
```bash
   streamlit run streamlit.py --server.port 8888 --server.fileWatcherType none
```

#### **Experimentation and Customization:**

The scripts support experimentation with various hyperparameters, such as learning rate, batch size, and number of epochs. For detailed instructions and more options, refer to the code documentation.
