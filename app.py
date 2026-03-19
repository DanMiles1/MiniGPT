import gradio as gr
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import tiktoken
import orbax.checkpoint
from pathlib import Path
from helper import MiniGPT, load_and_preprocess_data, generate_story

# Global state to hold components across tabs
global_state = {
    'tokenizer': tiktoken.get_encoding("gpt2"),
    'text_dl': None,
    'batches_per_epoch': 0,
    'maxlen': 128,
    'model': None,
}

def load_data(file_path, batch_size, max_stories, maxlen):
    try:
        global_state['maxlen'] = int(maxlen)
        text_dl, batches_per_epoch = load_and_preprocess_data(
            file_path=file_path,
            batch_size=int(batch_size),
            maxlen=int(maxlen),
            max_stories=int(max_stories),
            shuffle=True,  # Enable shuffling to learn better
            seed=42
        )
        global_state['text_dl'] = text_dl
        global_state['batches_per_epoch'] = batches_per_epoch
        return f"Successfully loaded dataset! Estimated batches per epoch: {batches_per_epoch}"
    except Exception as e:
        return f"Error loading data: {str(e)}"

def init_model(embed_dim, num_heads, feed_forward_dim, num_transformer_blocks):
    try:
        rngs = nnx.Rngs(0)
        model = MiniGPT(
            maxlen=global_state['maxlen'],
            vocab_size=global_state['tokenizer'].n_vocab,
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            feed_forward_dim=int(feed_forward_dim),
            num_transformer_blocks=int(num_transformer_blocks),
            rngs=rngs,
        )
        global_state['model'] = model
        return "Model initialized successfully!"
    except Exception as e:
        return f"Error initializing model: {str(e)}"

# Define generic loss and jitted train step outside to prevent re-compiling cache
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

def train_model(num_epochs, peak_lr):
    if global_state['model'] is None:
        yield "Error: Model not initialized. Please go to the Data & Model tabs first."
        return
    if global_state['text_dl'] is None:
        yield "Error: DataLoader not initialized. Please load data first."
        return

    model = global_state['model']
    text_dl = global_state['text_dl']
    batches_per_epoch = global_state['batches_per_epoch']
    num_epochs = int(num_epochs)
    peak_lr = float(peak_lr)

    total_steps = batches_per_epoch * num_epochs
    warmup_steps = max(1, total_steps // 10)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
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

    output_log = f"Starting training for {num_epochs} epochs ({total_steps} total steps)...\n"
    yield output_log

    for epoch in range(num_epochs):
        step = 0
        for batch in text_dl:
            input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
            target_batch = prep_target_batch(jnp.array(jnp.array(batch).T)).astype(jnp.int32)
            
            train_step(model, optimizer, metrics, (input_batch, target_batch))
            
            if (step + 1) % 2 == 0:
                current_loss = metrics.compute()['loss']
                metrics.reset()
                current_lr = lr_schedule(step)
                output_log += f"Epoch: {epoch + 1}/{num_epochs}, Step {step + 1}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}\n"
                yield output_log
            
            step += 1
            
    # Save orbax Checkpoint
    checkpoint_path = Path.cwd() / "gradio_checkpoint.orbax"
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(checkpoint_path, nnx.state(model), force=True)
    
    output_log += f"\nTraining Complete! Model saved successfully to {checkpoint_path}"
    yield output_log

def generate_text_wrapper(prompt, temperature, max_new_tokens):
    if global_state['model'] is None:
        return "Error: Model not trained/initialized!"
    try:
        model = global_state['model']
        generated = generate_story(model, prompt, float(temperature), int(max_new_tokens))
        return generated
    except Exception as e:
        return f"Error during generation: {str(e)}"

with gr.Blocks(title="JAX MiniGPT Trainer") as app:
    gr.Markdown("# 🚀 JAX MiniGPT Neural Network Builder")
    
    with gr.Tabs():
        with gr.TabItem("1. Data Stage"):
            gr.Markdown("### Load and Preprocess TinyStories")
            with gr.Row():
                file_path_input = gr.Textbox(value="../TinyStories-1000.txt", label="Dataset Path")
                max_stories_input = gr.Number(value=1000, label="Max Stories limit")
            with gr.Row():
                maxlen_input = gr.Number(value=128, label="Context length (maxlen)")
                batch_size_input = gr.Number(value=32, label="Batch Size")
            load_btn = gr.Button("Load Dataset", variant="primary")
            load_out = gr.Textbox(label="Status")
            
            load_btn.click(load_data, 
                           inputs=[file_path_input, batch_size_input, max_stories_input, maxlen_input], 
                           outputs=load_out)

        with gr.TabItem("2. Model Specs"):
            gr.Markdown("### Configure the Transformer Architecture")
            with gr.Row():
                embed_dim_input = gr.Number(value=192, label="Embedding Dimension")
                num_heads_input = gr.Number(value=6, label="Number of Attention Heads")
            with gr.Row():
                ff_dim_input = gr.Number(value=512, label="Feed Forward Dimension")
                blocks_input = gr.Number(value=6, label="Number of Transformer Blocks")
            init_btn = gr.Button("Initialize Model", variant="primary")
            init_out = gr.Textbox(label="Status")

            init_btn.click(init_model,
                           inputs=[embed_dim_input, num_heads_input, ff_dim_input, blocks_input],
                           outputs=init_out)
            
        with gr.TabItem("3. Training"):
            gr.Markdown("### Train the Language Model")
            with gr.Row():
                epochs_input = gr.Number(value=10, label="Number of Epochs")
                lr_input = gr.Number(value=5e-4, label="Peak Learning Rate")
            train_btn = gr.Button("Start Training & Save Checkpoint", variant="primary")
            train_out = gr.Textbox(label="Training Logs", lines=15, max_lines=20)
            
            train_btn.click(train_model, 
                            inputs=[epochs_input, lr_input], 
                            outputs=train_out)

        with gr.TabItem("4. Generator"):
            gr.Markdown("### Test the Model")
            prompt_input = gr.Textbox(value="Once upon a time", label="Prompt Prefix")
            with gr.Row():
                temp_input = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
                tokens_input = gr.Slider(minimum=10, maximum=200, value=50, step=1, label="Max New Tokens")
            gen_btn = gr.Button("Generate Story", variant="primary")
            gen_out = gr.Textbox(label="Generated Output", lines=5)
            
            gen_btn.click(generate_text_wrapper,
                          inputs=[prompt_input, temp_input, tokens_input],
                          outputs=gen_out)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
