use burn::{
    tensor::{activation::log_softmax, backend::Backend, Tensor, Int},
};

pub fn entropy<B: Backend>(scores: Tensor<B, 3>) -> Tensor<B, 2> {
    // scores: [bs, seq_len, vocab]
    // returns: [bs, seq_len]
    let log_probs = log_softmax(scores, 2);
    let probs = log_probs.clone().exp();
    let p_log_p = log_probs * probs;
    let entropy = p_log_p.sum_dim(2).neg();
    entropy.squeeze::<2>()
}

pub fn patch_start_mask_from_entropy_with_monotonicity<B: Backend>(
    entropies: Tensor<B, 2>,
    threshold: f64,
) -> Tensor<B, 2, Int> {
    let [bs, seq_len] = entropies.dims();
    
    if seq_len == 0 {
        return entropies.greater_elem(threshold).int();
    }

    // Create mask with first element True (1)
    // We can't easily mutate tensors in Burn, so we build it.
    
    // differences = entropies[:, 1:] - entropies[:, :-1]
    let current = entropies.clone().slice([0..bs, 1..seq_len]);
    let prev = entropies.clone().slice([0..bs, 0..seq_len-1]);
    let differences = current - prev;

    // condition = differences > t
    let condition = differences.greater_elem(threshold).int();

    // Construct full mask: [1, condition...]
    let start_mask = Tensor::ones([bs, 1], &entropies.device());
    Tensor::cat(vec![start_mask, condition], 1)
}

pub fn patch_start_ids_from_patch_start_mask<B: Backend>(
    patch_start_mask: Tensor<B, 2, Int>,
) -> Tensor<B, 2, Int> {
    let [bs, seq_len] = patch_start_mask.dims();
    
    // This is tricky in Burn without boolean indexing or nonzero.
    // We might need to iterate or use a different approach.
    // For now, since we are running inference on CPU/Metal, maybe we can pull to data?
    // But we want to keep it on device if possible.
    
    // Alternative: Use cumulative sum on the mask to get patch IDs?
    // Python code:
    // patch_ids = torch.arange(trunc_seq_len).repeat(bs, 1)
    // all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
    // patch_start_ids = all_patch_ids[patch_start_mask_padded]
    
    // Burn doesn't support boolean indexing like PyTorch yet for variable sized outputs per batch.
    // However, for BLT, we usually process batch_size=1 for patching in the demo.
    // Let's assume batch_size=1 for now to simplify, or implement a fixed size approach.
    
    // If we assume we want to return just the start indices, we can do it on CPU for now as patching is not the heavy part compared to the model.
    // But let's try to stay with Tensor ops if possible.
    
    // Actually, for the purpose of this port, getting the data back to Rust Vec is probably fine and easier to manipulate.
    // Let's implement a helper that returns Vec<Vec<usize>>.
    
    panic!("Use patch_start_indices_cpu instead");
}

pub fn patch_start_indices_cpu<B: Backend>(
    patch_start_mask: Tensor<B, 2, Int>,
) -> Vec<Vec<usize>> {
    let [bs, seq_len] = patch_start_mask.dims();
    let mask_data = patch_start_mask.into_data();
    let mask_values = mask_data.as_slice::<i32>().unwrap(); // Assuming Int maps to i32
    
    let mut batch_indices = Vec::with_capacity(bs);
    
    for b in 0..bs {
        let mut indices = Vec::new();
        for i in 0..seq_len {
            if mask_values[b * seq_len + i] == 1 {
                indices.push(i);
            }
        }
        batch_indices.push(indices);
    }
    
    batch_indices
}

pub fn patch_lengths_from_start_indices(
    start_indices: &[Vec<usize>],
    seq_len: usize,
) -> Vec<Vec<usize>> {
    start_indices.iter().map(|indices| {
        let mut lengths = Vec::new();
        for i in 0..indices.len() {
            let start = indices[i];
            let end = if i + 1 < indices.len() {
                indices[i+1]
            } else {
                seq_len
            };
            lengths.push(end - start);
        }
        lengths
    }).collect()
}
