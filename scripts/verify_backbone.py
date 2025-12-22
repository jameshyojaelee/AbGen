import torch
from abprop.models.transformer import AbPropModel, TransformerConfig

def test_forward_pass():
    print("Initializing model with modern config...")
    config = TransformerConfig(
        use_rope=True,
        norm_type="rmsnorm",
        activation="swiglu"
    )
    model = AbPropModel(config)
    print("Model initialized.")

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    print("Running forward pass...")
    outputs = model(input_ids, attention_mask)
    
    print("Forward pass successful!")
    print(f"MLM Logits shape: {outputs['mlm_logits'].shape}")
    print(f"Regression shape: {outputs['regression'].shape}")
    
    # Test Backward pass
    print("Testing backward pass...")
    loss = outputs['loss']
    if loss is not None:
        loss.backward()
        print("Backward pass successful!")
    else:
        print("No loss computed (expected if no labels provided).")
        
    # Test with labels
    print("Running forward pass with labels...")
    mlm_labels = input_ids.clone()
    outputs_with_loss = model(input_ids, attention_mask, mlm_labels=mlm_labels)
    loss = outputs_with_loss['loss']
    print(f"Loss: {loss.item()}")
    loss.backward()
    print("Backward pass with loss successful!")

if __name__ == "__main__":
    test_forward_pass()
