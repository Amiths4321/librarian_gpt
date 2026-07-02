import json

class BPETokenizer:
    def __init__(self):
        # 0-255 are reserved for raw individual bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}  # maps (byte1, byte2) -> new_token_id

    def _get_stats(self, ids_list: list) -> dict:
        """Counts how many times every adjacent pair of numbers occurs."""
        counts = {}
        for pair in zip(ids_list, ids_list[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_ids(self, ids_list: list, pair: tuple, new_id: int) -> list:
        """Replaces all occurrences of a targeted pair with a new single ID."""
        new_ids = []
        i = 0
        while i < len(ids_list):
            if i < len(ids_list) - 1 and (ids_list[i], ids_list[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids_list[i])
                i += 1
        return new_ids

    def train(self, text: str, vocab_size: int):
        """Learns token combinations from your training data."""
        assert vocab_size >= 256, "Vocab size must be at least 256 for basic bytes"
        num_merges = vocab_size - 256
        
        # Convert raw text characters directly into raw byte numbers (0-255)
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)

        print(f"Training BPE... Base length: {len(ids)} bytes.")

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break  # No more pairs left to merge
                
            # Find the most frequently occurring pair
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            # Record the merge rule
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Compress the list of IDs using the new merge rule
            ids = self._merge_ids(ids, best_pair, new_id)
            print(f"  Merge {i+1}/{num_merges}: Combined {best_pair} into new ID {new_id} (found {stats[best_pair]} times)")

    def encode(self, text: str) -> list:
        """Converts raw text into a list of trained token IDs."""
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)
        
        # We must apply merges in the exact order they were learned during training
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find which available pair in our text was learned earliest
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            
            if pair not in self.merges:
                break # No more mergeable pairs exist in this vocabulary
                
            ids = self._merge_ids(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list) -> str:
        """Converts a list of token IDs back into a human-readable string."""
        # Stitch the byte chunks back together using the vocab map
        byte_chunks = [self.vocab[idx] for idx in ids]
        compiled_bytes = b"".join(byte_chunks)
        # Decode bytes safely, replacing broken fragments if any exist
        return compiled_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Saves vocab configuration rules to a JSON file."""
        # JSON keys must be strings, so convert tuple keys to string descriptors
        serialized_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(filepath, "w") as f:
            json.dump(serialized_merges, f, indent=4)