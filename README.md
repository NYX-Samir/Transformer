Transformer from Scratch – README
Overview

This project implements a Transformer model from scratch using PyTorch. It follows the original Transformer architecture introduced in the paper “Attention is All You Need”. The model is modular, with clear building blocks such as embedding, positional encoding, multi-head attention, feed-forward layers, encoder, decoder, and projection layers.

Workflow

The implementation follows this step-by-step pipeline:

Import Libraries

Uses PyTorch for deep learning (torch, torch.nn)

Utilities: math, DataLoader, Dataset

Input Embedding

Converts token indices into dense vectors.

Scaled by √d_model to stabilize training.

Positional Encoding

Adds sinusoidal position information to embeddings so the model understands token order.

Multi-Head Attention (Self-Attention)

Splits embedding into multiple heads.

Each head learns different relations between tokens.

Supports masking for autoregressive decoding.

Layer Normalization

Normalizes activations across features for stable training.

Feed Forward Network

Two fully connected layers with ReLU activation.

Expands then reduces dimensionality (d_model → d_ff → d_model).

Residual Connection

Adds input back after sublayer computation to prevent gradient vanishing.

Encoder

Stack of N encoder blocks.

Each block = self-attention + feed forward + residual + normalization.

Decoder

Stack of N decoder blocks.

Each block = masked self-attention + cross-attention (with encoder output) + feed forward.

Projection Layer

Maps decoder output to target vocabulary.

Uses log softmax for prediction.

Transformer (Full Model)

Encoder processes source input.

Decoder generates target sequence step by step.

Final projection predicts the next token.

flowchart TD

    subgraph Step1
        A[1. Import Libraries]
    end

    subgraph Step2
        B[2. Input Embedding]
    end

    subgraph Step3
        C[3. Positional Encoding]
    end

    subgraph Step4
        D[4. Multi-Head Attention (Self Attention)]
    end

    subgraph Step5
        E[5. Add & Normalize]
    end

    subgraph Step6
        F[6. Feed Forward]
    end

    subgraph Step7
        G[7. Residual Connection]
    end

    subgraph Step8
        H[8. Encoder]
    end

    subgraph Step9
        I[9. Decoder]
    end

    subgraph Step10
        J[10. Projection Layer]
    end

    subgraph Step11
        K[11. Build Transformer]
    end

    %% Connections
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

Key Parameters

d_model → Size of embeddings (default 512)

N → Number of encoder/decoder layers (default 6)

h → Number of attention heads (default 8)

d_ff → Feed-forward hidden size (default 2048)

dropout → Dropout rate (default 0.1)


