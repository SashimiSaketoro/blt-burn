use blake3::Hasher;
use polars::prelude::*;

/// Hash a Polars binary column element-wise, returning a Utf8 Series of hex digests.
pub fn hash_blake3(input: &Series) -> PolarsResult<Series> {
    let ca = input.binary()?;
    let hashes: Vec<Option<String>> = ca
        .into_iter()
        .map(|opt_bytes| opt_bytes.map(hash_bytes))
        .collect();
    Ok(Series::new(input.name().clone(), hashes))
}

/// Convenience helper for hashing arbitrary byte slices (used outside of expressions).
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize().to_hex().to_string()
}
