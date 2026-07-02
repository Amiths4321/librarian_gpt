from bpe import BPETokenizer

if __name__ == "__main__":
    # 1. Provide a small training sample with repetitive text patterns
    sample_corpus = """
    attention is all you need. self-attention is a core component of transformers.
    transformers are the heart of artificial intelligence ecosystems.
    when a word speaks at a party, it uses queries, keys, and values to find neighbors.
    the dog bit the man and the man bit the dog back.
    """

    # 2. Instantiate and train our custom tokenizer
    tokenizer = BPETokenizer()
    
    # We want 256 base bytes + 20 custom learned tokens = 276 total vocabulary size
    TARGET_VOCAB_SIZE = 276 
    tokenizer.train(sample_corpus, vocab_size=TARGET_VOCAB_SIZE)

    print("\n" + "="*50 + "\n🔥 TOKENIZER TESTING TIMELINE\n" + "="*50)

    # 3. Test the trained tokenizer on a brand new sentence
    test_phrase = "transformers use attention!"
    print(f"Original Text: '{test_phrase}' (Length: {len(test_phrase)} characters)")
    
    encoded_ids = tokenizer.encode(test_phrase)
    print(f"Encoded Token IDs: {encoded_ids} (Length: {len(encoded_ids)} tokens)")
    
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded Back:   '{decoded_text}'")

    # 4. Save rules out to inspection file
    tokenizer.save("bpe_rules.json")
    print("\n💾 Merge rule blueprints saved successfully to 'bpe_rules.json'!")