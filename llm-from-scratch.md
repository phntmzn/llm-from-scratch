Below is a step-by-step guide to **training a small GPT-style language model from scratch in Python using PyTorch**. We'll follow the same concepts in your notes: architecture, dataset preparation, tokenization, initialization, autoregressive learning, loss calculation, backpropagation, optimization, epochs, and overfitting prevention.

We'll build a working mini-LLM that learns to predict the next character in a sequence. This is a practical starting point before moving to word-level or subword-tokenized models.

# Step 1: Understand the training pipeline

The complete process looks like this:

```text
Raw Text Dataset
      |
      v
Text Preprocessing
      |
      v
Tokenization
      |
      v
Training and Validation Data
      |
      v
Define GPT Architecture
      |
      v
Initialize Model Weights
      |
      v
Forward Pass: Predict Next Token
      |
      v
Calculate Cross-Entropy Loss
      |
      v
Backpropagation
      |
      v
Optimizer Updates Weights
      |
      v
Repeat Across Batches and Epochs
      |
      v
Evaluate and Generate Text
```

Each step builds on the previous one.

---

# Step 2: Create the project structure

Create a project named `python-llm`.

```text
python-llm/
│
├── data/
│   ├── raw/
│   │   └── sample.txt
│   │
│   └── processed/
│
├── configs/
│   └── model.yaml
│
├── model/
│   ├── __init__.py
│   ├── attention.py
│   ├── transformer.py
│   └── gpt.py
│
├── tokenizer/
│   ├── __init__.py
│   └── tokenizer.py
│
├── training/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── inference/
│   ├── __init__.py
│   └── generate.py
│
├── checkpoints/
│
├── requirements.txt
└── README.md
```

For this first implementation, we'll keep the model simple and put most of the training logic into one script. Once it works, you can distribute the components into the modules above.

## Install the dependencies

Create `requirements.txt`:

```text
torch
pyyaml
```

Open your terminal in the project directory and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS, this creates an isolated Python environment for your project.

Verify that PyTorch is installed:

```bash
python -c "import torch; print(torch.__version__)"
```

---

# Step 3: Prepare the training dataset

An LLM needs text from which to learn patterns.

For our experiment, we'll use a small text corpus. A real LLM requires a much larger, carefully curated dataset.

Create `data/raw/sample.txt`:

```text
Language models learn patterns from text.

A neural network learns by adjusting its parameters.
Training improves the predictions of a language model.
A transformer uses attention to process sequences.

Python provides tools for building neural networks.
PyTorch supports automatic differentiation and optimization.

A language model predicts the next token from previous tokens.
The training process minimizes prediction error.
Attention helps a model learn relationships between tokens.

Machine learning systems improve through repeated training.
A dataset contains examples used to optimize model parameters.
```

You can add more original text to this file as you experiment.

**Important:** A tiny dataset is useful for learning the mechanics of training, but it won't produce a capable general-purpose language model.

---

# Step 4: Tokenize the text

Neural networks operate on numbers, not raw text.

Tokenization converts text into numerical IDs.

For example, a character-level tokenizer might represent:

```text
Text:  hello

Tokens: h e l l o

IDs:    7 4 11 11 14
```

The IDs above are illustrative; the actual mapping depends on the vocabulary.

For simplicity, we'll implement a character-level tokenizer.

Create `tokenizer/tokenizer.py`:

```python
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))

        self.stoi = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.itos = {
            index: char
            for char, index in self.stoi.items()
        }

        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [
            self.stoi[char]
            for char in text
        ]

    def decode(self, tokens):
        return "".join(
            self.itos[token]
            for token in tokens
        )
```

### How it works

The tokenizer builds a vocabulary from the unique characters in the training text.

- `stoi` maps characters to integer IDs.
- `itos` maps integer IDs back to characters.
- `encode()` converts text into numbers.
- `decode()` converts numbers back into text.
- `vocab_size` records the number of unique characters.

Character-level tokenization is easy to understand, but modern LLMs generally use subword tokenizers that represent text more efficiently.

---

# Step 5: Convert text into training sequences

Our model will learn this task:

> Given a sequence of characters, predict the next character.

For example:

```text
Input:  languag
Target: anguage
```

Each target character is the next character corresponding to the input position.

Create `training/dataset.py`:

```python
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(
            0,
            len(self.tokens) - self.block_size
        )

    def __getitem__(self, index):
        x = self.tokens[
            index:index + self.block_size
        ]

        y = self.tokens[
            index + 1:index + self.block_size + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )
```

Suppose the token sequence is:

```text
[10, 20, 30, 40, 50]
```

With a block size of four, one training example becomes:

```text
Input:  [10, 20, 30, 40]
Target: [20, 30, 40, 50]
```

The model learns to predict the next token at every position.

This is the foundation of **autoregressive language modeling**.

---

# Step 6: Define the GPT model

A GPT-style transformer contains several important components:

1. Token embeddings.
2. Positional embeddings.
3. Causal self-attention.
4. Feed-forward neural networks.
5. Layer normalization.
6. An output projection that predicts the next token.

We'll implement a small GPT model using PyTorch.

Create `model/gpt.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=64,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, idx):
        batch_size, sequence_length = idx.shape

        positions = torch.arange(
            sequence_length,
            device=idx.device
        )

        token_embeddings = self.token_embedding(idx)

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(1024, 1024)
            ).bool(),
            persistent=False
        )

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        attention = (
            q @ k.transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        attention = attention.masked_fill(
            ~mask,
            float("-inf")
        )

        attention = F.softmax(
            attention,
            dim=-1
        )

        attention = self.attention_dropout(attention)

        output = attention @ v

        output = output.transpose(
            1, 2
        ).contiguous().view(
            batch_size,
            sequence_length,
            embedding_dim
        )

        output = self.projection(output)

        return self.residual_dropout(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                4 * embedding_dim
            ),
            nn.GELU(),
            nn.Linear(
                4 * embedding_dim,
                embedding_dim
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            dropout
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))

        x = x + self.feed_forward(self.norm2(x))

        return x
```

### What the architecture does

**Token embeddings:** Convert token IDs into vectors.

**Positional embeddings:** Give the model information about token positions.

**Causal self-attention:** Lets each position use information from preceding positions without seeing future tokens.

**Feed-forward layers:** Transform the learned representations.

**Layer normalization:** Helps stabilize training.

**Output head:** Produces a score, or logit, for every possible next token.

The model's predictions have the shape:

```text
[batch_size, sequence_length, vocab_size]
```

For example, with a batch size of 16, a sequence length of 64, and a vocabulary of 50 characters, the output shape is:

```text
[16, 64, 50]
```

These scores are used to calculate the training loss.

---

# Step 7: Initialize the model and define the training objective

Before training, we must:

1. Load the text.
2. Build the tokenizer.
3. Convert text into token IDs.
4. Split the data into training and validation sets.
5. Initialize the model's parameters.
6. Select a loss function and optimizer.

The loss measures how far the model's predictions are from the correct next tokens.

We'll use **cross-entropy loss**, a standard objective for next-token prediction.

For example:

```text
Context: The cat sat on the

Correct next token: mat

Model prediction:
    mat   -> 0.60
    floor -> 0.20
    chair -> 0.10
    roof  -> 0.10
```

A higher probability assigned to the correct token generally produces a lower loss.

The model learns to improve these predictions by updating its parameters.

---

# Step 8: Train the model

Create `training/train.py`:

```python
import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tokenizer.tokenizer import CharTokenizer
from training.dataset import TextDataset
from model.gpt import GPTLanguageModel


# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 16
BLOCK_SIZE = 64
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

LEARNING_RATE = 3e-4
EPOCHS = 100
DROPOUT = 0.1

DATA_PATH = "data/raw/sample.txt"
CHECKPOINT_PATH = "checkpoints/model.pt"


# -----------------------------
# Device
# -----------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# -----------------------------
# Load and tokenize data
# -----------------------------

with open(DATA_PATH, "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = CharTokenizer(text)

tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

print(f"Characters: {len(text)}")
print(f"Vocabulary size: {tokenizer.vocab_size}")


# -----------------------------
# Split training and validation data
# -----------------------------

split_index = int(0.9 * len(tokens))

train_tokens = tokens[:split_index]

val_tokens = tokens[split_index:]

train_dataset = TextDataset(
    train_tokens,
    BLOCK_SIZE
)

val_dataset = TextDataset(
    val_tokens,
    BLOCK_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Initialize model
# -----------------------------

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)


# -----------------------------
# Loss and optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate():
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, tokenizer.vocab_size),
            y.reshape(-1)
        )

        total_loss += loss.item()
        total_batches += 1

    model.train()

    if total_batches == 0:
        return float("nan")

    return total_loss / total_batches


# -----------------------------
# Training loop
# -----------------------------

best_val_loss = float("inf")

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_batches = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        # Clear old gradients.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass.
        logits = model(x)

        # Calculate next-token prediction loss.
        loss = criterion(
            logits.reshape(-1, tokenizer.vocab_size),
            y.reshape(-1)
        )

        # Backpropagation.
        loss.backward()

        # Limit unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update model parameters.
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    scheduler.step()

    train_loss = (
        total_loss / total_batches
        if total_batches > 0
        else float("nan")
    )

    val_loss = evaluate()

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # Save the best model.
    if (
        total_batches > 0
        and val_loss == val_loss
        and val_loss < best_val_loss
    ):
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "chars": tokenizer.chars,
                "block_size": BLOCK_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            },
            CHECKPOINT_PATH
        )

print("Training complete.")

if best_val_loss < float("inf"):
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
else:
    print(
        "No checkpoint saved. "
        "Check that the dataset is long enough."
    )
```

## Understand the training loop

The core of the training process is:

```python
optimizer.zero_grad(set_to_none=True)

logits = model(x)

loss = criterion(
    logits.reshape(-1, tokenizer.vocab_size),
    y.reshape(-1)
)

loss.backward()

optimizer.step()
```

Each line has a specific purpose.

| Operation | Purpose |
|---|---|
| `optimizer.zero_grad()` | Clears gradients from the previous batch. |
| `model(x)` | Predicts the next token at each position. |
| `criterion(...)` | Calculates prediction error. |
| `loss.backward()` | Calculates gradients using backpropagation. |
| `optimizer.step()` | Updates the model's parameters. |

The optimizer uses the gradients to adjust the weights in a direction intended to reduce the loss.

This process repeats across batches.

**One important distinction:** A batch is a group of training examples processed together. An epoch is one complete pass through the training dataset, not a single batch or a single parameter update.

---

# Step 9: Understand epochs and loss

When you run the training script:

```bash
python -m training.train
```

You should see output resembling:

```text
Using device: mps
Characters: 500
Vocabulary size: 35

Epoch 001/100 | Train Loss: 3.6500 | Val Loss: 3.5000
Epoch 002/100 | Train Loss: 3.4200 | Val Loss: 3.3000
Epoch 003/100 | Train Loss: 3.1800 | Val Loss: 3.1500
...
Training complete.
Best validation loss: ...
Checkpoint saved to: checkpoints/model.pt
```

These values are illustrative, not guaranteed results. Actual losses depend on your dataset, initialization, hardware, and training settings.

### What should you monitor?

**Training loss:** Measures prediction error on the training data.

**Validation loss:** Measures prediction error on held-out data.

A healthy training process generally reduces training loss while validation loss also improves.

If training loss continues to fall but validation loss starts rising, the model may be overfitting.

A few useful techniques are:

- **Dropout:** Randomly disables some activations during training.
- **Early stopping:** Stops training when validation performance no longer improves.
- **Learning rate scheduling:** Changes the learning rate during training.
- **Gradient clipping:** Limits unusually large gradient norms.

These techniques can help, but they do not guarantee generalization. Dataset quality, model capacity, and training configuration matter too.

### A note about this implementation

The example saves the best checkpoint based on validation loss, but it doesn't stop early. It also uses a very small dataset, so its validation loss will be noisy and may not reliably indicate generalization.

For a more robust training pipeline, use a larger dataset, a fixed validation split, and a configurable patience-based early-stopping mechanism.

---

# Step 10: Generate text using the trained model

Once training finishes, you can use the saved checkpoint to generate new text.

Create `inference/generate.py`:

```python
import torch

from model.gpt import GPTLanguageModel
from tokenizer.tokenizer import CharTokenizer


CHECKPOINT_PATH = "checkpoints/model.pt"


# -----------------------------
# Device
# -----------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# -----------------------------
# Load checkpoint
# -----------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

tokenizer = CharTokenizer("".join(checkpoint["chars"]))

model = GPTLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -----------------------------
# Generate text
# -----------------------------

prompt = "Language"

unknown_chars = set(prompt) - set(tokenizer.chars)

if unknown_chars:
    raise ValueError(
        f"Prompt contains characters outside the vocabulary: "
        f"{unknown_chars}"
    )

context = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    generated = model.generate(
        context,
        max_new_tokens=200
    )

output = tokenizer.decode(
    generated[0].tolist()
)

print(output)
```

Run it:

```bash
python -m inference.generate
```

The model starts with the prompt `Language` and generates additional characters one at a time.

Because the dataset is tiny, the output may be repetitive, incoherent, or grammatically incorrect. That's normal for this experiment.

Also, this character-level model doesn't understand language in the same way a large pretrained LLM does. It learns statistical patterns in its training text.

---

# Step 11: Understand what happens mathematically

The model's learning process revolves around three important operations.

### 1. Forward propagation

The input tokens pass through the transformer:

\[
X \rightarrow \text{Embeddings}
\rightarrow \text{Attention}
\rightarrow \text{Feed-Forward}
\rightarrow \text{Logits}
\]

The output logits represent the model's scores for possible next tokens.

### 2. Cross-entropy loss

For a sequence of \(T\) target tokens, the average next-token loss is:

\[
\mathcal{L}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P_\theta(x_{t+1}\mid x_1,\ldots,x_t)
\]

Here:

- \(\mathcal{L}\) is the loss.
- \(x_t\) is a token.
- \(P_\theta\) is the probability predicted by the model.
- \(\theta\) represents the model's trainable parameters.

The model is penalized when it assigns a low probability to the correct next token.

### 3. Backpropagation and optimization

Backpropagation calculates how the loss changes with respect to the model's parameters:

\[
\nabla_\theta \mathcal{L}
\]

An optimizer such as AdamW uses these gradients to update the parameters.

Conceptually, a basic gradient-descent update is:

\[
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta\mathcal{L}
\]

where \(\eta\) is the learning rate.

AdamW uses a more sophisticated update involving estimates of gradient means and variances, along with decoupled weight decay.

Over many updates, the model attempts to improve its next-token predictions.

---

# Step 12: Connect the implementation to the original concepts

Here's how the ideas from your notes map to the code.

| Concept | Implementation |
|---|---|
| Model architecture | `GPTLanguageModel` |
| Attention layers | `CausalSelfAttention` |
| Feed-forward layers | `FeedForward` |
| Dataset preparation | `data/raw/sample.txt` |
| Tokenization | `CharTokenizer` |
| Numerical representation | `tokenizer.encode()` |
| Training examples | `TextDataset` |
| Model initialization | `GPTLanguageModel(...)` |
| Autoregressive prediction | `model(x)` with causal attention |
| Loss calculation | `nn.CrossEntropyLoss()` |
| Backpropagation | `loss.backward()` |
| Weight updates | `optimizer.step()` |
| Batches | `DataLoader` |
| Epochs | `for epoch in range(EPOCHS)` |
| Learning rate scheduling | `CosineAnnealingLR` |
| Regularization | Dropout |
| Validation | `evaluate()` |
| Model checkpointing | `torch.save()` |
| Text generation | `model.generate()` |

## Important distinction: training from scratch versus fine-tuning

This project initializes a small transformer with randomly initialized weights and trains it on your own text. That is training from scratch.

Fine-tuning is different: it starts with an already pretrained model and continues training it on a more specific dataset.

A small educational model like this can teach you how training works, but training a capable general-purpose LLM from scratch typically requires substantially more data, compute, and engineering.

---

# Step 13: Recommended improvements

Once the initial implementation runs successfully, develop it in stages.

**Stage 1 — Training fundamentals**
- Increase the size and diversity of your dataset.
- Record training and validation loss.
- Add early stopping.
- Save training checkpoints.

**Stage 2 — Better tokenization**
- Replace the character tokenizer with a subword tokenizer such as Byte Pair Encoding (BPE).
- Save the tokenizer vocabulary alongside each checkpoint.
- Handle unknown characters and special tokens appropriately.

**Stage 3 — More capable architecture**
- Increase the context length.
- Experiment with embedding dimensions, attention heads, and transformer layers.
- Add configurable model settings in `configs/model.yaml`.

**Stage 4 — Training reliability**
- Track experiments and random seeds.
- Resume training from checkpoints.
- Add automated tests for attention, tokenization, and model shapes.
- Monitor validation loss and generated samples.

**Stage 5 — Performance**
- Use larger, well-curated datasets.
- Experiment with mixed-precision training when supported.
- Profile memory use and throughput.
- Consider distributed training when the model and dataset outgrow a single device.

For modern LLM development, subword tokenization, large datasets, efficient attention, and careful evaluation become particularly important.

---

## Final result

You now have the blueprint for a small GPT-style language model that can:

1. Read a text corpus.
2. Convert characters into token IDs.
3. Build autoregressive training sequences.
4. Predict the next token using a transformer.
5. Calculate cross-entropy loss.
6. Update weights through backpropagation and AdamW.
7. Monitor training and validation loss.
8. Save the best checkpoint.
9. Generate text from a prompt.

**The central idea:** An autoregressive language model repeatedly predicts the next token, measures its error, and adjusts its parameters to improve future predictions. Training on sufficiently large and diverse data can help it learn increasingly complex patterns, but generalization depends on more than simply minimizing training loss.
