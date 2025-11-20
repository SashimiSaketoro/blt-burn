use burn::{
    tensor::{activation::log_softmax, backend::Backend, Tensor, Int},
};

pub fn entropy<B: Backend>(scores: Tensor<B, 3>) -> Tensor<B, 2> {
    // scores: [bs, seq_len, vocab]
    // returns: [bs, seq_len]
    let [bs, seq_len, _vocab] = scores.dims();
    let log_probs = log_softmax(scores, 2);
    let probs = log_probs.clone().exp();
    let p_log_p = log_probs * probs;
    // sum_dim(2) on [bs, seq_len, vocab] -> [bs, seq_len, 1]
    // Then neg and reshape explicitly to [bs, seq_len]
    // This avoids squeeze() panicking on batch_size=1
    p_log_p.sum_dim(2).neg().reshape([bs, seq_len])
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
