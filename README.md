## Amharic Entity Extraction for E-Commerce
This project focuses on building a pipeline for real-time entity extraction from Amharic text messages in Ethiopian e-commerce Telegram channels. The goal is to identify key entities such as product names, prices, and locations, enabling business intelligence and insights for e-commerce use cases.
# Key Features
1. Real-Time Data Ingestion:

Connect to Ethiopian-based Telegram e-commerce channels.
Collect and store messages, images, and metadata in a structured format.
2. Amharic-Specific Preprocessing:

Tokenization, normalization, and cleaning of Amharic text.
Handling linguistic features unique to Amharic, such as diacritics and compound words.
3. Named Entity Recognition (NER):

Fine-tune existing language models for Amharic-specific entity extraction.
Extract entities such as:
Product Names or Types
Prices or Monetary Values
Locations
Materials or Ingredients
4. Model Comparison:

Evaluate and compare the performance of different NER models on labeled Amharic datasets.
Analyze interpretability and accuracy for business intelligence applications.

By completing this project,
Develop a pipeline for ingesting and preprocessing Amharic text data.
Fine-tune large language models for Amharic-specific NER tasks.
Gain insights into how extracted entities can enhance decision-making in e-commerce.
Perform performance analysis of different models and interpretability techniques.
# Technologies Used
Programming Languages: Python
Libraries: Telethon, TensorFlow, PyTorch, Hugging Face Transformers, AmharicNLP
Database: MySQL/PostgreSQL for structured data storage
Tools: GitHub for CI/CD, Docker for containerization
Preprocessing: Regex, custom Amharic tokenizers
Machine Learning: NER fine-tuning, evaluation, and comparison
# How to Use
1. Clone the repository 
git clone https://github.com/melatdest/EthioMart_Telegram-based-e-commerce-activities.git
cd amharic-entity-extraction
2. Set up the environment:
Install dependencies from requirements.txt.
3. Configure your Telegram API credentials in a .env file.
Run the data ingestion script to fetch Telegram messages
# contributing
Contributions are welcome! If you have suggestions, improvements, or new features to propose, please feel free to submit a pull request or open an issue.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.






