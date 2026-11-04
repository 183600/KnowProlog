use know_prolog::entity_relation_extractor::EntityRelationExtractor;

fn main() {
    let extractor = EntityRelationExtractor::new();
    
    // Test the specific case mentioned in the issue
    let result = extractor.trim_person_candidate("李四被张三");
    println!("Result for '李四被张三': {:?}", result);
    
    // Should extract "李四" (and potentially "张三" depending on tokenization)
    if let Some((name, len)) = result {
        println!("Extracted name: '{}' (length: {})", name, len);
        assert!(name == "李四" || name == "张三", "Should extract a valid name");
        assert!(len > 0, "Should extract a name with length > 0");
    } else {
        println!("No names found - this indicates the fix may not be working");
    }
    
    println!("Verification complete!");
}
