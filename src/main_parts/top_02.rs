#[derive(Parser)]
#[command(name = "owl", about = "OWL 人工反馈台账：挖掘、裁决、报告与导入导出")]
struct OwlCli {
    #[command(subcommand)]
    command: OwlCommands,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum OwlReviewAction {
    Accept,
    Reject,
    Defer,
    Revise,
}
