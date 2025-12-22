import torch
from abprop.models.transformer import AbPropModel, TransformerConfig
from abprop.tokenizers import TOKEN_TO_ID, apply_mlm_mask

def test_mamba_pass():
    print("Initializing model with MAMBA config...")
    config = TransformerConfig(
        encoder_type="mamba",
        d_model=64, # Small for test
        num_layers=2,
        ssm_d_state=16
    )
    model = AbPropModel(config)
    print("Model initialized.")

    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    print("Running forward pass...")
    outputs = model(input_ids, attention_mask)
    
    print("Forward pass successful!")
    print(f"MLM Logits shape: {outputs['mlm_logits'].shape}")
    
    # Test Backward pass
    print("Testing backward pass...")
    masked_input_ids, mlm_labels = apply_mlm_mask(
        input_ids,
        attention_mask,
        mlm_probability=0.2,
    )
    if (mlm_labels != -100).sum() == 0:
        mlm_labels[0, 1] = input_ids[0, 1]
        masked_input_ids[0, 1] = TOKEN_TO_ID["<mask>"]
    outputs_with_loss = model(masked_input_ids, attention_mask, mlm_labels=mlm_labels)
    loss = outputs_with_loss['loss']
    print(f"Loss: {loss.item()}")
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0
    loss.backward()
    print("Backward pass with loss successful!")

if __name__ == "__main__":
    test_mamba_pass()
