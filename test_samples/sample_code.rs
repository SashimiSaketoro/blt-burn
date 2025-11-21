use std::collections::HashMap;

/// A simple key-value store implementation
pub struct KeyValueStore<K, V> {
    data: HashMap<K, V>,
    capacity: usize,
}

impl<K, V> KeyValueStore<K, V> 
where
    K: Eq + std::hash::Hash,
{
    /// Creates a new KeyValueStore with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Inserts a key-value pair into the store
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.data.len() >= self.capacity && !self.data.contains_key(&key) {
            // Simple eviction: remove the first item
            if let Some(first_key) = self.data.keys().next().cloned() {
                self.data.remove(&first_key);
            }
        }
        self.data.insert(key, value)
    }

    /// Retrieves a value by key
    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }

    /// Returns the current size of the store
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks if the store is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut store = KeyValueStore::new(3);
        
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        
        store.insert("key1", "value1");
        store.insert("key2", "value2");
        
        assert_eq!(store.get(&"key1"), Some(&"value1"));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_capacity_limit() {
        let mut store = KeyValueStore::new(2);
        
        store.insert("key1", 1);
        store.insert("key2", 2);
        store.insert("key3", 3); // Should evict key1
        
        assert_eq!(store.get(&"key1"), None);
        assert_eq!(store.get(&"key3"), Some(&3));
        assert_eq!(store.len(), 2);
    }
}

fn main() {
    let mut cache: KeyValueStore<String, i32> = KeyValueStore::new(100);
    
    // Example usage
    cache.insert("temperature".to_string(), 25);
    cache.insert("humidity".to_string(), 60);
    
    if let Some(temp) = cache.get(&"temperature".to_string()) {
        println!("Temperature: {}°C", temp);
    }
}
