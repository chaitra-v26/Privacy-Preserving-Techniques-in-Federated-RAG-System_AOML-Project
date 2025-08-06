# Privacy Preserving Techniques in a Federated RAG System

## 📋 Project Overview

This project explores the integration of privacy-preserving techniques within Federated Retrieval-Augmented Generation (RAG) systems. By combining federated learning with RAG architectures, we aim to maintain data privacy while enabling efficient information retrieval across distributed nodes.

## 🎯 Objectives

- **Synergistic Integration**: Combine the strengths of federated learning and RAG systems
- **Privacy Protection**: Implement cutting-edge privacy-preserving techniques including differential privacy, secure multi-party computation, and homomorphic encryption
- **Collaborative Innovation**: Enable multiple entities to contribute to model improvement without centralized data aggregation
- **User-Centric Design**: Provide end users with unparalleled control and transparency over their data

## 👥 Team Members

| Name | Section | SRN |
|------|---------|-----|
| Abhishek Bhat | A | PES1UG22AM005 |
| Anagha S Bharadwaj | A | PES1UG22AM020 |
| C Hemachandra | A | PES1UG22AM044 |
| Chaitra V | A | PES1UG22AM045 |

**Supervisor**: Prof. Bhaskar Jyoti Das (bhaskarjyoti01@gmail.com)
**Course**: Algorithms and Optimizations For Machine Learning (CSE-AIML)  
**Semester**: 5th Semester CSE-AIML 2024 (August-Dec 2024)  
**Institution**: PES University, Bangalore

## 🔧 Tech Stack

### Programming Languages
- **Python** - Primary programming language for implementation

### Machine Learning & Deep Learning
- **TensorFlow/Keras** - Deep learning framework
- **scikit-learn** - Machine learning algorithms and utilities
- **VGG16** - Feature extraction for image processing
- **FAISS** - Efficient similarity search and clustering

### Privacy & Security
- **AES (Advanced Encryption Standard)** - Symmetric encryption
- **Elliptic Curve Cryptography (ECC)** - Asymmetric encryption
- **Homomorphic Encryption** - Privacy-preserving computations
- **Differential Privacy** - Statistical disclosure control
- **Fernet Encryption** - Symmetric encryption for secure data handling

### Data Processing & NLP
- **NLTK** - Natural Language Processing toolkit
- **TF-IDF Vectorization** - Text feature extraction
- **CountVectorizer** - Text preprocessing and vectorization
- **Cosine Similarity** - Similarity computation

### Datasets
- **CIFAR-10** (752.88 MB) - 60,000 color images across 10 classes
- **CIFAR-100** (186.3 MB) - 60,000 images across 100 classes
- **MNIST** (54.95 MB) - 70,000 grayscale handwritten digit images
- **Emotion Detection Dataset** (3.77 MB) - Text sentiment analysis
- **Amazon Product Reviews** (300.9 MB) - E-commerce review data
- **Regularization-Images-Man**

### Additional Libraries
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib/Seaborn** - Data visualization
- **Threading** - Concurrent processing

## 🏗️ System Architecture

### Core Components

1. **PrivacyPreservingDistributedIndex**: Manages encrypted knowledge distribution across nodes
2. **KnowledgeBase**: Implements TF-IDF vectorization with encrypted storage
3. **PrivacyPreservingQueryOrchestrator**: Handles distributed search operations
4. **Federated Nodes**: Independent processing units maintaining local encrypted indices

## 📁 Project Structure

```
Privacy-Preserving-Techniques-in-Federated-RAG-System_AOML-Project/
├── CODE BASE/
│   ├── Image_AES.ipynb                           # AES + Differential 
│   ├── Image_ecc&fedavg.ipynb                   # ECC + Federated Averaging
│   ├── Text_Federated_Learning_Technique.ipynb  # End-to-End Encryption for 
│   ├── Text_SMPC_Principles.ipynb              # Secure Multi-Party 
│   └── Text_simple_harmonic_encryption_and_differential_privacy.ipynb
├── AOML_DATAVORTEX_005_020_044_045_PROJECT_PPT.pptx
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
# Install required Python packages
pip install tensorflow
pip install scikit-learn
pip install nltk
pip install numpy pandas matplotlib seaborn
pip install faiss-cpu
pip install cryptography
pip install jupyter notebook
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/chaitra-v26/Privacy-Preserving-Techniques-in-Federated-RAG-System_AOML-Project.git
cd Privacy-Preserving-Techniques-in-Federated-RAG-System_AOML-Project
```

2. **Download datasets**
   - **MNIST**: [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
   - **CIFAR-10**: [Kaggle CIFAR-10](https://www.kaggle.com/c/cifar-10/)
   - **CIFAR-100**: [Kaggle CIFAR-100](https://www.kaggle.com/datasets/fedesoriano/cifar100)
   - **Emotion Detection**: [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)
   - **Amazon Product Reviews**: [Kaggle Amazon Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
   - **regularization-images-man**: [Kaggle Regularization-Images-Man](https://www.kaggle.com/datasets/timothyalexisvass/regularization-images-man)


3. **Set up environment**
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # If requirements.txt is available
```

## 🔄 How to Run

### 1. Image-based RAG with AES and Differential Privacy
```bash
jupyter notebook "CODE BASE/Image_AES.ipynb"
```
- Implements AES encryption for image queries
- Applies differential privacy for secure similarity search
- Uses CIFAR-10/CIFAR-100 datasets

### 2. Federated Learning with ECC
```bash
jupyter notebook "CODE BASE/Image_ecc&fedavg.ipynb"
```
- Demonstrates elliptic curve cryptography
- Implements federated averaging with privacy preservation
- Shows secure communication between federated nodes

### 3. Text-based RAG with Homomorphic Encryption
```bash
jupyter notebook "CODE BASE/Text_simple_harmonic_encryption_and_differential_privacy.ipynb"
```
- Implements homomorphic encryption for text processing
- Applies Laplace mechanism for differential privacy
- Uses emotion detection and Amazon review datasets

### 4. Secure Multi-Party Computation for Text
```bash
jupyter notebook "CODE BASE/Text_SMPC_Principles.ipynb"
```
- Demonstrates SMPC principles in federated search
- Implements secure query handling across distributed nodes
- Uses Fernet encryption for data protection

### 5. End-to-End Encryption for Text Retrieval
```bash
jupyter notebook "CODE BASE/Text_Federated_Learning_Technique.ipynb"
```
- Shows complete federated learning workflow
- Implements end-to-end encryption for text queries
- Demonstrates privacy-preserving similarity computation

## 📊 Key Features

### Privacy Techniques Implemented

1. **Advanced Encryption Standard (AES)**: Symmetric encryption for secure data storage
2. **Elliptic Curve Cryptography (ECC)**: Efficient public-key cryptography
3. **Homomorphic Encryption**: Computation on encrypted data
4. **Differential Privacy**: Statistical privacy through controlled noise injection
5. **Secure Multi-Party Computation (SMPC)**: Collaborative computation without data sharing

### System Capabilities

- **Multi-modal Support**: Handles both image and text queries
- **Distributed Processing**: Efficient federated architecture
- **Privacy Guarantees**: Multiple layers of privacy protection
- **Scalable Design**: Supports multiple federated nodes
- **Real-time Retrieval**: Fast similarity search with encrypted data

## 📈 Results

The implementation successfully demonstrates:

- **Image RAG**: Secure image retrieval with <0.1% accuracy loss
- **Text RAG**: Privacy-preserving text search with maintained relevance
- **Federated Training**: Convergent model training across distributed nodes
- **Encryption Overhead**: Minimal computational overhead (~5-10%)
- **Privacy Budget**: Configurable privacy-utility trade-off

## 🔮 Future Work

- **Enhanced Encryption**: Improve AES, homomorphic encryption, and ECC efficiency
- **Advanced Architectures**: Integration with transformer models
- **Multi-modal Datasets**: Support for combined text, image, and metadata
- **Scalability Improvements**: Better distributed indexing techniques
- **Real-world Applications**: Healthcare and finance domain implementations
- **Regulatory Compliance**: GDPR and HIPAA compliance validation
- **Explainable AI**: Integration of XAI for transparent retrieval
- **Energy Efficiency**: Sustainable federated learning protocols

## 📚 References

1. [A Frequency Estimation Algorithm under Local Differential Privacy](https://ieeexplore.ieee.org/document/9377325)
2. [Research on Data Mining Technology of Internet Privacy Protection](https://ieeexplore.ieee.org/document/10653389)
3. [A Multi-Stage Partial Homomorphic Encryption Scheme](https://ieeexplore.ieee.org/document/10212153)
4. [A Study on Partially Homomorphic Encryption](https://ieeexplore.ieee.org/document/10035630)
5. [Analysis and Comparison of Various Fully Homomorphic Encryption Techniques](https://ieeexplore.ieee.org/document/8940577)

---

**Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning)**  
**PES University, Bangalore, India**
