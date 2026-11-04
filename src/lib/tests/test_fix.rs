use know_prolog::entity_relation_extractor::{EntityRelationExtractor};

#[test]
fn test_trim_person_candidate_fix() {
    let extractor = EntityRelationExtractor::new();
    
    // Test case: "李四被张三" should extract both "李四" and "张三"
    let result = extractor.trim_person_candidate("李四被张三");
    assert!(result.is_some(), "Should find names in '李四被张三'");
    let (name, len) = result.unwrap();
    assert_eq!(name, "李四", "Should extract '李四'");
    assert_eq!(len, 2, "Length of '李四' should be 2");
    
    // Test another case: "王五被赵六" should extract both names
    let result2 = extractor.trim_person_candidate("王五被赵六");
    assert!(result2.is_some(), "Should find names in '王五被赵六'");
    let (name2, len2) = result2.unwrap();
    assert_eq!(name2, "王五", "Should extract '王五'");
    assert_eq!(len2, 2, "Length of '王五' should be 2");
    
    // Test case that should NOT be extracted (too short)
    let result3 = extractor.trim_person_candidate("A");
    assert!(result3.is_none(), "Single character should not be extracted");
    
    // Test case that should be extracted (4 characters)
    let result4 = extractor.trim_person_candidate("小明张三");
    assert!(result4.is_some(), "Should extract names in '小明张三'");
    let (name4, len4) = result4.unwrap();
    assert_eq!(name4, "小明", "Should extract '小明'");
    assert_eq!(len4, 2, "Length of '小明' should be 2");
    
    println!("All tests passed!");
}
