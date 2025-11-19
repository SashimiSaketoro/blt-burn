use crate::model::LMTransformer;
use anyhow::{Result, Context};
use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, TensorData},
};
use safetensors::SafeTensors;
use std::path::Path;
use std::fs;

pub fn load_weights<B: Backend>(
    model: LMTransformer<B>,
    path: &Path,
    device: &B::Device,
) -> Result<LMTransformer<B>> {
    println!("Loading weights from {}", path.display());
    let file_content = fs::read(path).context("Failed to read safetensors file")?;
    let safetensors = SafeTensors::deserialize(&file_content).context("Failed to deserialize safetensors")?;

    let mut model = model; // Take ownership
    
    // Helper to load tensor data
    let load_tensor_data = |name: &str| -> Result<TensorData> {
        let tensor_view = safetensors.tensor(name)?;
        let shape = tensor_view.shape();
        let data = tensor_view.data();
        
        let dtype = tensor_view.dtype();
        let floats: Vec<f32> = match dtype {
            safetensors::Dtype::F32 => {
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            },
            _ => anyhow::bail!("Unsupported dtype: {:?}", dtype),
        };
        
        let shape_vec: Vec<usize> = shape.iter().map(|&s| s).collect();
        Ok(TensorData::new(floats, shape_vec))
    };
    
    // Helper to load 2D tensor (Linear) with optional transpose
    let load_linear = |name: &str, transpose: bool, device: &B::Device| -> Result<Tensor<B, 2>> {
        let data = load_tensor_data(name)?;
        let mut tensor = Tensor::<B, 2>::from_data(data, device);
        if transpose {
            tensor = tensor.transpose();
        }
        Ok(tensor)
    };

    // Helper to load 1D tensor (Norm)
    let load_norm = |name: &str, device: &B::Device| -> Result<Tensor<B, 1>> {
        let data = load_tensor_data(name)?;
        Ok(Tensor::<B, 1>::from_data(data, device))
    };
    
    // Load Embeddings (2D)
    if let Ok(t) = load_linear("tok_embeddings.weight", false, device) {
        model.tok_embeddings.weight = Param::from_tensor(t);
    }
    
    // Load Norm (1D)
    if let Ok(t) = load_norm("norm.weight", device) {
        model.norm.weight = Param::from_tensor(t);
    }
    
    // Load Output (2D, transpose)
    if let Ok(t) = load_linear("output.weight", true, device) {
        model.output.weight = Param::from_tensor(t);
    }
    
    // Load Layers
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}.", i);
        
        // Attention
        if let Ok(t) = load_linear(&format!("{}attention.wq.weight", prefix), true, device) {
            layer.attention.wq.weight = Param::from_tensor(t);
        }
        if let Ok(t) = load_linear(&format!("{}attention.wk.weight", prefix), true, device) {
            layer.attention.wk.weight = Param::from_tensor(t);
        }
        if let Ok(t) = load_linear(&format!("{}attention.wv.weight", prefix), true, device) {
            layer.attention.wv.weight = Param::from_tensor(t);
        }
        if let Ok(t) = load_linear(&format!("{}attention.wo.weight", prefix), true, device) {
            layer.attention.wo.weight = Param::from_tensor(t);
        }
        
        // FeedForward
        if let Ok(t) = load_linear(&format!("{}feed_forward.w1.weight", prefix), true, device) {
            layer.feed_forward.w1.weight = Param::from_tensor(t);
        }
        if let Ok(t) = load_linear(&format!("{}feed_forward.w2.weight", prefix), true, device) {
            layer.feed_forward.w2.weight = Param::from_tensor(t);
        }
        if let Ok(t) = load_linear(&format!("{}feed_forward.w3.weight", prefix), true, device) {
            layer.feed_forward.w3.weight = Param::from_tensor(t);
        }
        
        // Norms
        if let Ok(t) = load_norm(&format!("{}attention_norm.weight", prefix), device) {
            layer.attention_norm.weight = Param::from_tensor(t);
        }
        if let Ok(t) = load_norm(&format!("{}ffn_norm.weight", prefix), device) {
            layer.ffn_norm.weight = Param::from_tensor(t);
        }
    }
    
    println!("Weights loaded successfully!");
    Ok(model)
}
