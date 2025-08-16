# **Transformer from Scratch – README**

## **Overview**  
This project implements a **Transformer model** from scratch using **PyTorch**.  
It follows the original Transformer architecture introduced in the paper *“Attention is All You Need”*.  
The model is modular, with clear building blocks such as embedding, positional encoding, multi-head attention, feed-forward layers, encoder, decoder, and projection layers.  

---

## **Workflow**  
The implementation follows this step-by-step pipeline:  

### **1. Import Libraries**  
- Uses PyTorch for deep learning (`torch`, `torch.nn`)  
- Utilities: `math`, `DataLoader`, `Dataset`  

### **2. Input Embedding**  
- Converts token indices into dense vectors  
- Scaled by √d_model to stabilize training  

### **3. Positional Encoding**  
- Adds sinusoidal position information to embeddings so the model understands token order  

### **4. Multi-Head Attention (Self-Attention)**  
- Splits embedding into multiple heads  
- Each head learns different relations between tokens  
- Supports masking for autoregressive decoding  

### **5. Layer Normalization**  
- Normalizes activations across features for stable training  

### **6. Feed Forward Network**  
- Two fully connected layers with ReLU activation  
- Expands then reduces dimensionality (`d_model → d_ff → d_model`)  

### **7. Residual Connection**  
- Adds input back after sublayer computation to prevent gradient vanishing  

### **8. Encoder**  
- Stack of N encoder blocks  
- Each block = self-attention + feed forward + residual + normalization  

### **9. Decoder**  
- Stack of N decoder blocks  
- Each block = masked self-attention + cross-attention (with encoder output) + feed forward  

### **10. Projection Layer**  
- Maps decoder output to target vocabulary  
- Uses log softmax for prediction  

### **11. Transformer (Full Model)**  
- Encoder processes source input  
- Decoder generates target sequence step by step  
- Final projection predicts the next token  

---

### **Key Parameters**

d_model → Size of embeddings (default 512)

N → Number of encoder/decoder layers (default 6)

h → Number of attention heads (default 8)

d_ff → Feed-forward hidden size (default 2048)

dropout → Dropout rate (default 0.1)

## **Architecture Flow (Mermaid)**  

```mermaid
flowchart TD
    %% Input and Embedding
    A[Import Libraries] --> B[Input Sequence]
    B --> C[Input Embedding]
    C --> D[Positional Encoding]
    
    %% Encoder Stack
    subgraph Encoder["Encoder Stack (x N layers)"]
        D --> E1[Multi-Head Self-Attention]
        E1 --> E2[Add & Norm]
        E2 --> E3[Feed Forward Network]
        E3 --> E4[Add & Norm]
    end
    E4 --> F[Encoder Output]
    
    %% Decoder Stack
    subgraph Decoder["Decoder Stack (x N layers)"]
        G[Target Sequence] --> H[Output Embedding]
        H --> I[Positional Encoding]
        I --> J1[Masked Multi-Head Self-Attention]
        J1 --> J2[Add & Norm]
        J2 --> J3[Encoder-Decoder Attention]
        F --> J3
        J3 --> J4[Add & Norm]
        J4 --> J5[Feed Forward Network]
        J5 --> J6[Add & Norm]
    end
    J6 --> K[Decoder Output]
    
    %% Final Output
    K --> L[Linear Projection]
    L --> M[Softmax]
    M --> N[Output Probabilities]


