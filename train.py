import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import tiktoken
import orbax.checkpoint
from pathlib import Path
from helper import MiniGPT, load_and_preprocess_data

# Move these outside of main so JIT caches cleanly
def loss_fn(model, batch):
    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, logits

@nnx.jit
def train_step(model, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(grads)

def main():
    maxlen = 128
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Defaults changed to 1000 max_stories (entire dataset) for valid generalization
    text_dl, batches_per_epoch = load_and_preprocess_data(
        file_path='../TinyStories-1000.txt',
        batch_size=32,
        maxlen=maxlen,
        max_stories=1000, 
        shuffle=True, # Shuffle for better generation characteristics
        seed=42
    )

    rngs = nnx.Rngs(0)
    model = MiniGPT(
        maxlen=maxlen,
        vocab_size=tokenizer.n_vocab,
        embed_dim=192,
        num_heads=6,
        feed_forward_dim=512,
        num_transformer_blocks=6,
        rngs=rngs,
    )

    num_epochs = 10
    total_steps = batches_per_epoch * num_epochs
    warmup_steps = max(1, total_steps // 10)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=5e-4,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5
    )
    
    optimizer = nnx.ModelAndOptimizer(
        model,
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)
    )
    
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    prep_target_batch = jax.vmap(lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0]))))

    print(f"Total training steps: {total_steps:,}")
    print("Starting training:")
    for epoch in range(num_epochs):
        step = 0
        for batch in text_dl:
            input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
            target_batch = prep_target_batch(jnp.array(jnp.array(batch).T)).astype(jnp.int32)
            
            print(".", end="", flush=True)
            train_step(model, optimizer, metrics, (input_batch, target_batch))
            
            if (step + 1) % 2 == 0:
                current_loss = metrics.compute()['loss']
                metrics.reset()
                current_lr = lr_schedule(step)
                print(f"\nEpoch: {epoch + 1}, Step {step + 1}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}")
            
            step += 1

    # Save orbax Checkpoint
    checkpoint_path = Path.cwd() / "train_checkpoint.orbax"
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(checkpoint_path, nnx.state(model), force=True)
    
    print(f"\nTraining Complete! Model saved successfully to {checkpoint_path}")

if __name__ == "__main__":
    main()
