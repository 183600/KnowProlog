use know_prolog::entity_relation_extractor::EntityRelationExtractor;

fn main() {
    let extractor = EntityRelationExtractor::new();
    
    // Test the specific case mentioned in the issue
    let result = extractor.trim_person_candidate("李四被张三");
    println!("Result for '李四被张三': {:?}", result);
    
    // Should extract "李四" (and potentially "张三" depending on tokenization)
    if let Some((name, len)) = result {
        println!("Extracted name: '{}' (length: {})", name, len);
        // Check if we got a valid name
        if name.is_empty() {
            println!("ERROR: Empty name extracted");
        } else {
            println!("SUCCESS: Valid name extracted");
        }
    } else {
        println!("ERROR: No names found");
    }
}
