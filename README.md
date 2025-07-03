### **AI Models for Legal Text Processing**

## Table of contents
- Scope of project
- Demo GIF
- Model information
- GPT-2 Text Generation Process
- Pegasus Text Summarization Process
- Text Classification
- Running Streamlit app
- Experimentation and Customization
  
## Scope of project
Legal documents are very tough to interpret as they are very long and have a lot of important information. Humans find it quite challenging to quickly go through legal documents and understand key aspects without missing vital information. Most legaldocuments have a lot of information that is more relevant such as dates, deadlines, names of people, etc. Attorneys, judges, lawyers, and others in the justice system are constantly surrounded by large amounts of the legal text, which can be difficult to manage across many cases. They face the problem of organizing briefs, judgments, and acts. Due to the huge amount of legal information available on the internet, as well as other sources, the research community needs to do more extensive research on the area of legal text processing, which can help us make sense of the vast amount of available data. This information growth has compelled the requirement to develop systems that can help legal professionals, as well as ordinary citizens, get relevant legal information with very little effort.We have worked on different tasks such as Text summarization (both extractive and abstractive), Sentiment analysis to Predict the Judgment, and Text generation of legal documents. <br>

Explore our specialized repository designed for training state-of-the-art AI models on legal text data. We focus on key tasks such as text classification, text generation, and text summarization using BERT, GPT-2, and PEGASUS.
## Demo GIF
![](/Demo/Text_Summarization.gif)

## Model information
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
After running the app one can use the test files (.txt format) in the Test files folder to upload files in the app and check results. 

#### **Experimentation and Customization:**

The scripts support experimentation with various hyperparameters, such as learning rate, batch size, and number of epochs. For detailed instructions and more options, refer to the code documentation.
