use know_prolog::query_planner::{QueryPlanner, QueryPlan};

fn show(q: &str) {
    let planner = QueryPlanner::new();
    match planner.parse(q) {
        Some(plan) => println!("{q:} => {:?}", plan),
        None => println!("{q:} => None"),
    }
}

fn main() {
    let _ = QueryPlan::Similar { entity: String::new() };
    show("李四有几个座位");
    show("李四有没有座位");
    show("张三有座位吗");
    show("张三有多少位朋友");
    show("苹果公司有多少员工");
    show("张三有没有姓名");
    show("李四是王五的师傅吗");
    show("张三的师傅是李四还是王五");
}
