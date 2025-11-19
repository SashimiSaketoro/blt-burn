use blt_burn::tokenizer::BltTokenizer;

fn main() {
    let tokenizer = BltTokenizer::new(true, true);
    let text = "Hello World";
    let tokens = tokenizer.encode(text);
    
    println!("Text: {}", text);
    println!("Tokens: {:?}", tokens);
    
    // Expected: [BOS, 'H'+4, 'e'+4, ..., EOS]
    // BOS=1, EOS=2, OFFSET=4
    // 'H' = 72 -> 76
    // 'e' = 101 -> 105
    
    let decoded = tokenizer.decode(&tokens);
    println!("Decoded: {}", decoded);
    
    assert_eq!(decoded, text);
}
