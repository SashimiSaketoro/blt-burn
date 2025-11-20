use anyhow::Result;
use tiktoken_rs::CoreBPE;

pub const BOE_ID: usize = 0;
pub const BOS_ID: usize = 1;
pub const EOS_ID: usize = 2;
pub const BPE_ID: usize = 3;
pub const OFFSET: usize = 4;
pub const PAD_ID: usize = usize::MAX; // -1 in Python, using MAX for usize
pub const BYTE_UNITS: usize = 256;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
}

pub struct BltTokenizer {
    add_bos: bool,
    add_eos: bool,
    offsetting_special_char: usize,
    bos_id: usize,
    eos_id: usize,
    bpe_tokenizer: Option<Box<dyn Tokenizer>>, // For bpe_delim=True
}

impl BltTokenizer {
    pub fn new(add_bos: bool, add_eos: bool) -> Self {
        Self {
            add_bos,
            add_eos,
            offsetting_special_char: OFFSET,
            bos_id: BOS_ID,
            eos_id: EOS_ID,
            bpe_tokenizer: None,
        }
    }

    pub fn new_with_bpe(add_bos: bool, add_eos: bool, bpe_tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            add_bos,
            add_eos,
            offsetting_special_char: OFFSET,
            bos_id: BOS_ID,
            eos_id: EOS_ID,
            bpe_tokenizer: Some(bpe_tokenizer),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        
        if let Some(bpe) = &self.bpe_tokenizer {
            // BPE Delim mode: Encode with BPE, then convert to bytes + BPE_ID
            let bpe_tokens_ids = bpe.encode(text);
            
            for id in bpe_tokens_ids {
                // Decode single token to get its string/byte representation
                let token_str = bpe.decode(&[id]);
                
                // Insert BPE_ID (mapped to model space)
                tokens.push(BPE_ID);
                
                // Convert token bytes to model space
                let token_bytes: Vec<usize> = token_str
                    .bytes()
                    .map(|b| b as usize + self.offsetting_special_char)
                    .collect();
                tokens.extend(token_bytes);
            }
            
            // Add BOS/EOS if requested (handled below, but we might need to be careful about order)
            // The Python code adds BOS/EOS *around* the whole sequence.
            // But wait, Python text2bytes_bpe_delims returns the sequence *without* BOS/EOS, 
            // and then encode adds them.
            // So we just populate `tokens` here and let the common code add BOS/EOS.
        } else {
            // Standard Byte Encoding
            let byte_tokens: Vec<usize> = text
                .bytes()
                .map(|b| b as usize + self.offsetting_special_char)
                .collect();
            tokens.extend(byte_tokens);
        }

        if self.add_bos {
            tokens.insert(0, self.bos_id);
        }
        if self.add_eos {
            tokens.push(self.eos_id);
        }

        tokens
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| {
                if t >= self.offsetting_special_char {
                    Some((t - self.offsetting_special_char) as u8)
                } else {
                    None
                }
            })
            .collect();
        
        String::from_utf8_lossy(&bytes).to_string()
    }
}

pub struct TikTokenTokenizer {
    bpe: CoreBPE,
    _bos_id: usize,
    _eos_id: usize,
}

impl TikTokenTokenizer {
    pub fn new(model: &str) -> Result<Self> {
        let bpe = tiktoken_rs::get_bpe_from_model(model)?;
        // Tiktoken IDs vary by model, usually cl100k_base has different special tokens
        // This is a best-effort mapping based on the Python code's intent
        let bos_id = bpe.encode_with_special_tokens("<|begin_of_text|>").first().cloned().unwrap_or(0); 
        let eos_id = bpe.encode_with_special_tokens("<|end_of_text|>").first().cloned().unwrap_or(1);
        
        Ok(Self { bpe, _bos_id: bos_id, _eos_id: eos_id })
    }
}

impl Tokenizer for TikTokenTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        self.bpe.encode_with_special_tokens(text)
    }

    fn decode(&self, tokens: &[usize]) -> String {
        self.bpe.decode(tokens.to_vec()).unwrap_or_default()
    }
}

pub struct SentencePieceTokenizer {
    model: sentencepiece::SentencePieceProcessor,
    _bos_id: usize,
    _eos_id: usize,
}

impl SentencePieceTokenizer {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = sentencepiece::SentencePieceProcessor::open(model_path)?;
        let bos_id = model.bos_id().unwrap_or(1) as usize;
        let eos_id = model.eos_id().unwrap_or(2) as usize;
        
        Ok(Self { model, _bos_id: bos_id, _eos_id: eos_id })
    }
    
    pub fn vocab_size(&self) -> usize {
        self.model.len()
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        self.model.encode(text)
            .unwrap_or_default()
            .iter()
            .map(|piece| piece.id as usize)
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.model.decode_piece_ids(&ids).unwrap_or_default()
    }
}
