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

classDiagram
    class InputEmbedding {
        +d_model
        +vocab_size
        +forward(x)
    }

    class PositionEncoding {
        +d_model
        +seq_len
        +dropout
        +forward(x)
    }

    class MultiHeadAttention {
        +d_model
        +h
        +d_k
        +forward(q, k, v, mask)
        +attention(query, key, value, mask, dropout)
    }

    class LayerNormalization {
        +eps
        +alpha
        +bias
        +forward(x)
    }

    class FeedForward {
        +linear_1
        +linear_2
        +forward(x)
    }

    class ResidualConnection {
        +dropout
        +norm
        +forward(x, sublayer)
    }

    class EncoderBlock {
        +self_attention
        +feed_forward
        +residualConnection
        +forward(x, src_mask)
    }

    class Encoder {
        +layers
        +norm
        +forward(x, mask)
    }

    class DecoderBlock {
        +self_attention
        +cross_attention
        +feed_forward
        +residualConnection
        +forward(x, encoder_output, src_mask, tgt_mask)
    }

    class Decoder {
        +layers
        +norm
        +forward(x, encoder_output, src_mask, tgt_mask)
    }

    class ProjectionLayer {
        +proj
        +forward(x)
    }

    class Transformer {
        +encoder
        +decoder
        +src_embed
        +tgt_embed
        +src_pos
        +tgt_pos
        +projection_layer
        +encode(src, src_mask)
        +decode(tgt, memory, src_mask, tgt_mask)
        +project(x)
    }

    %% Relationships
    Transformer --> Encoder
    Transformer --> Decoder
    Transformer --> InputEmbedding : src_embed
    Transformer --> InputEmbedding : tgt_embed
    Transformer --> PositionEncoding : src_pos
    Transformer --> PositionEncoding : tgt_pos
    Transformer --> ProjectionLayer

    Encoder --> EncoderBlock
    EncoderBlock --> MultiHeadAttention
    EncoderBlock --> FeedForward
    EncoderBlock --> ResidualConnection

    Decoder --> DecoderBlock
    DecoderBlock --> MultiHeadAttention : self_attention
    DecoderBlock --> MultiHeadAttention : cross_attention
    DecoderBlock --> FeedForward
    DecoderBlock --> ResidualConnection


Key Parameters

d_model → Size of embeddings (default 512)

N → Number of encoder/decoder layers (default 6)

h → Number of attention heads (default 8)

d_ff → Feed-forward hidden size (default 2048)

dropout → Dropout rate (default 0.1)
