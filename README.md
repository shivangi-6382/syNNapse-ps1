# Image Similarity Retrieval using Deep Embeddings

An end-to-end deep learning system for **visual similarity search**, built using **metric learning**, **cosine similarity**, FAISS optimization, and FastAPI deployment.

This project focuses on learning meaningful visual representations instead of performing traditional image classification.

---

##  Problem Overview

The objective of this project is to design an **image similarity retrieval pipeline** that:

- Converts images into dense vector embeddings  
- Retrieves visually similar images instead of predicting class labels  
- Ensures semantically similar instances lie close in embedding space  

Unlike classification systems that output discrete labels, this system performs:

> Representation Learning + Nearest Neighbor Search  

The system supports **open-set recognition**, meaning it can retrieve visually similar images even for unseen categories.

---

##  Detailed Problem Approach

The central idea is to learn a structured embedding space where:

- Similar images → Close in vector space  
- Dissimilar images → Far apart  

To achieve this:

1. Use a pretrained CNN backbone for feature extraction  
2. Add a projection head to learn task-specific embeddings  
3. Train using **Triplet Loss** for metric learning  
4. Normalize embeddings for cosine-based similarity search  
5. Store embeddings for fast retrieval  

This transforms the task into a **metric learning problem**, directly optimizing embedding distances instead of classification accuracy.

---

## End-to-End Pipeline

Complete workflow:

1. Feature extraction using pretrained ResNet-50  
2. Embedding learning via Triplet Loss  
3. Saving trained model  
4. Precomputing and storing gallery embeddings  
5. Retrieval using cosine similarity / FAISS  
6. Deployment through FastAPI  

---

## Model Architecture
<img width="736" height="524" alt="image" src="https://github.com/user-attachments/assets/e4b7bb76-d8af-4fe4-9dba-ffdbde020c3d" />


### Backbone
- ResNet-50 (ImageNet pretrained)
- Convolutional layers frozen
- Final classification layer removed

### Embedding Head
- Linear Layer → 512-dimensional vector  
- ReLU Activation  
- Batch Normalization  
- L2 Normalization  

### Final Output
- 512-dimensional L2-normalized embedding
  

### Why L2 Normalization?

Because cosine similarity becomes equivalent to dot product when vectors are normalized.

---

## Training Strategies Explored

### Option A — Full Triplet Training (Slowest)

**Approach**
- Train entire ResNet-50 + embedding head end-to-end  

**Observations**
- Very slow on CPU  
- High resource usage  
- Several hours of training  
- Not practical without GPU  

---

### Option B — Light Fine-Tuning (Selected Approach)

**Key Insight**

The task is embedding + similarity retrieval, not classification.

**Strategy**
- Freeze pretrained backbone  
- Train only embedding projection layer  

**Benefits**
- Preserves strong pretrained visual features  
- Faster convergence  
- Stable training  
- Works efficiently on CPU  

**Performance**
- Training time: ~30–60 minutes on CPU  
- Stable convergence  
- Strong retrieval accuracy
  <img width="730" height="548" alt="image" src="https://github.com/user-attachments/assets/0666a0f4-3900-4f2c-a08c-b0d9664943a2" />


---

## Similarity Metric

We use **Cosine Similarity** because:

- Works best with normalized embeddings  
- Measures angular similarity  
- Standard for metric learning  
- Computationally efficient  

**Formula:**
Cosine Similarity = (A · B) / (||A|| ||B||)

With normalized embeddings, this simplifies to a dot product.

---

## Retrieval Pipeline (Module B)

### Step 1: Query Encoding

- Input image passed through trained encoder  
- A 512-dimensional L2-normalized embedding is generated  

---

### Step 2: Embedding Database

All gallery images are pre-encoded and stored as:

- Embedding matrix (N × 512)  
- Corresponding image paths  
- Optional labels  

This ensures fast retrieval without re-encoding database images.

---

### Step 3: Similarity Computation

Two approaches implemented:

#### (A) Cosine Similarity — Baseline

- Implemented using Scikit-learn  
- Direct similarity computation:
 
- Suitable for small-to-medium datasets  

---

#### (B) FAISS Indexing — Optimized

For scalability:

- Embeddings stored in FAISS index  
- Inner product search used  
- Equivalent to cosine similarity with normalized vectors  
- Enables large-scale approximate nearest neighbor search  

FAISS significantly reduces retrieval latency compared to brute-force cosine computation.

---

### Step 4: Ranking

- Similarity scores sorted in descending order  
- Top-K highest scoring images selected  

---

### Step 5: Output

System returns:

- Top-K most visually similar images  
- Corresponding similarity scores  

---

## Evaluation Protocol

Performance evaluated using:

- Top-1 Accuracy  
- Top-5 Accuracy  
- Top-10 Accuracy  

These metrics measure whether the correct instance appears within Top-K retrieved results.


<img width="336" height="116" alt="image" src="https://github.com/user-attachments/assets/2321b59e-1518-4d42-a235-1b238f947edd" />


---

## Sample Retrieval Visualization 
<img width="1389" height="518" alt="image" src="https://github.com/user-attachments/assets/7a6625f6-9bbc-471d-a111-4d7216d7c3b2" />


Qualitative visualization is the most important evaluation proof for similarity systems.

For each query image:

- Query image displayed  
- Top-K retrieved results shown  
- Correct matches highlighted  

This demonstrates:

- Embedding quality  
- Semantic clustering ability  
- Real-world retrieval effectiveness  

---

## Why This Approach Was Chosen

### Problem Alignment

The requirement was:

- Visual encoding  
- Similarity-based retrieval  
- Open-set recognition  

Metric learning directly optimizes embedding distances, making it more suitable than classification models.

---

### Efficiency vs Performance Trade-off

Freezing the backbone:

- Reduces computational cost  
- Prevents overfitting  
- Retains pretrained visual knowledge  

Provides strong balance between:

- Accuracy  
- Training time  
- Deployment feasibility  

---

### Scalability

Using FAISS:

- Enables large-scale retrieval  
- Reduces latency  
- Makes system production-ready  

---

## Technology Stack

### Core Deep Learning
- PyTorch — model training & inference  
- Torchvision — pretrained ResNet-50 and transforms  
- NumPy — numerical operations  

---

### Data Processing & Visualization
- Pandas — metadata handling  
- Pillow (PIL) — image loading  
- Matplotlib — retrieval visualization  
- tqdm — progress tracking  

---

### Similarity Search & Deployment
- Scikit-learn — cosine similarity computation  
- FAISS — scalable nearest neighbor search  
- FastAPI — REST API deployment  
- Uvicorn — ASGI server  

---

### Development Environment
- Python 3.13  
- Jupyter Notebook — experimentation & training  
- PyCharm — modular project structure  
- Git & GitHub — version control  

---

# Sample Output
<img width="730" height="326" alt="image" src="https://github.com/user-attachments/assets/dd00a32b-2d43-4482-bf72-22d17ed7bc18" />
---

# Final System Summary

This project successfully implements:

- Deep embedding learning using Triplet Loss  
- 512-dimensional L2-normalized feature vectors  
- Cosine similarity-based retrieval  
- FAISS-optimized scalable search  
 

It builds a scalable **visual similarity search engine** suitable for:

- E-commerce product matching  
- Visual recommendation systems  
- Duplicate detection  
- Inventory similarity matching  
