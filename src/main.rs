use clap::{Parser, Subcommand};
use std::io::{BufRead, Read, Write};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

// 从库（know_prolog）中导入所需的功能
use know_prolog::{
    binding_frames::FrameProgram,
    chronicle::{Chronicle, FsyncPolicy},
    config_contract,
    contextual_reference::{
        parse_clarify_act, rewrite_at, BindingLedger, BindingScope, ClarifyAct, ClarifyRequest,
        ContextualResolver,
    },
    conversation_theory::{self, ConversationTheory},
    create_default_synonyms,
    dense_retrieval::{DenseRetriever, SearchBudget, SearchConstraints},
    detect_input_mode,
    dialogue_eval::{
        compare_baseline as compare_dialogue_baseline, evaluate_seed_with, DialogueEvalBaseline,
        EvalKnowledge,
    },
    dialogue_frame::ClarificationOutcome,
    dialogue_fuzz::{
        self, compare_baseline as compare_fuzz_baseline, FuzzBaseline, FuzzConfig, FuzzVerdict,
    },
    dialogue_manager::{DialogueManager, MoveKind},
    embedding_eval::{
        compare_baseline, evaluate, triples_from_prolog_facts, EvalBaseline, EvalConfig,
    },
    entity_registry::{extract_alias_candidates, MintVerdict, NominationKind},
    external_knowledge::{get_direct_answer, AnswerOutcome, ExternalKnowledgeService},
    external_ledger::{ExternalLedger, LedgerLoad},
    extract_query_entities_from_tagged, frontier, fulfillment,
    grammar_extractor::GrammarExtractor,
    history_trust::TrustAssessment,
    holographic_memory::{HoloAnalogyHit, HoloHit},
    identity_eval::{
        compare_baseline as compare_identity_baseline, evaluate_seed as evaluate_identity,
        IdentityEvalBaseline,
    },
    import_journal::{ImportJournal, JournalLoad},
    knowledge_fusion::{Adjudication, FusionPolicy, NamespaceKind, NamespaceRecord, TrustLevel},
    law_book::LawBook,
    lex_gate, lexical_trace,
    link_prediction::{LinkPredictor, Triple as LawTriple},
    load_config_from,
    local_semantic_adapter::LocalSemanticAdapter,
    neuro_symbolic_contrast::ContrastArbiter,
    nlp::{normalize_words, preprocess},
    nlu::NLUEngine,
    ontology_reasoning::OntologyProgram,
    ownthink_api::select_candidate_by_context,
    path_memory::{EdgeSource, PathEdge, PathPredicate, TraversalGraph, TraversalOptions},
    pattern_forge::{self, ForgeRequest, PatternForge},
    pattern_ir::{self, PatternLibrary},
    perf_ledger::PerfLedger,
    plugin::{PluginRegistry, PluginRuntime, PluginSessionState, PluginSettings},
    prolog_engine::PrologEngine,
    prolog_engine::PrologFact,
    prompt_spec::{self, LintSeverity, PromptMode},
    proof_bench::{self, ProofBenchBaseline},
    proof_journal::{JournalProjection, ProofCertificate, ProofDetail, ProofLimits},
    provenance::{
        self, parse_provenance_tsv, ConflictReport, ConflictSchema, ProvenanceLedger,
        ProvenanceLoad,
    },
    quality::{
        build_report_with_provenance, compare_baseline as compare_quality_baseline,
        render_entity_passport, QualityBaseline, QualityIndex,
    },
    query_planner::QueryPlanner,
    recovery::RecoveryMode,
    relation_acquisition::{GoalOpen, RelationAcquisition},
    relation_frame::{
        diagnose as diagnose_frames, narrate_frame, FrameExtractor, FrameGate, FRAME_DIALECT,
    },
    release::{self, CheckOptions},
    resolution_engine::SearchStatus,
    shape_dialogue::ShapeDialogue,
    shape_eval::{
        compare_baseline as compare_shape_eval_baseline, evaluate_seed as evaluate_shape_dialogue,
        ShapeEvalBaseline,
    },
    shape_gate::{RepairAction, ShapeReport, ShapeSet},
    startup_ledger::StartupLedger,
    symbolic_reasoner::SymbolicReasoner,
    temporal::{self, TemporalConstraint, TemporalIndex, UnknownTimePolicy, Validity},
    trace_graph::TraceMode,
    view_render::{
        render_coverage_report, render_proof_ascii, render_proof_mermaid, render_timeline_flow,
        render_timeline_markdown, CoverageInput, KnowledgeSubgraph, TimelineTurn,
    },
    weak_construction::{parse_feedback, ConstructionOptions, WeakConstructor},
    InputMode, KnowledgeExtractor, KnowledgeItem, OxigraphManager, ResponseFormatter, SystemConfig,
};

#[derive(Parser)]
#[command(name = "knowprolog")]
#[command(about = "KnowProlog - Chinese knowledge graph system")]
#[command(version)]
struct Cli {
    /// Override the configured Oxigraph store path (all subcommands).
    /// Interactive dialogue memory follows the overridden store when its
    /// configured path is relative, so `--store` never mixes session state.
    #[arg(long, global = true)]
    store: Option<String>,
    /// Configuration file path (defaults to config.toml in the working directory)
    #[arg(long, global = true, default_value = "config.toml")]
    config: String,
    /// Failure recovery mode for interactive dialogue: "off", "observe", or "on"
    #[arg(long, global = true, default_value = "observe")]
    recovery: String,
    /// Layered trace-graph mode for interactive dialogue: "off", "observe", or "on".
    /// When omitted, the `[trace_graph] mode` from config.toml is used.
    #[arg(long, global = true)]
    trace: Option<String>,
    #[command(subcommand)]
    command: Option<Commands>,
}

// 实体属性查询模式的全局正则
static ENTITY_ATTRIBUTE_RE: OnceLock<regex::Regex> = OnceLock::new();

#[derive(Subcommand, Debug)]
enum Commands {
    /// Import JSON-LD knowledge base from file
    ImportJsonld {
        /// Path to the JSON-LD file
        file_path: String,
    },
    /// Import an RDF file (Turtle / N-Triples / N-Quads / TriG / RDF/XML;
    /// format inferred from the extension)
    ImportRdf {
        /// Path to the RDF file (.ttl / .nt / .nq / .trig / .rdf / .xml)
        file_path: String,
    },
    /// Export the default graph as an RDF file (Turtle / N-Triples / N-Quads /
    /// TriG / RDF/XML / JSON-LD; format inferred from the extension)
    ExportRdf {
        /// Output file path (.ttl / .nt / .nq / .trig / .rdf / .xml / .jsonld);
        /// "-" writes Turtle to stdout
        file_path: String,
    },
    /// Import JSON-LD knowledge base from content
    ImportJsonldContent {
        /// JSON-LD content as string
        content: String,
    },
    /// Run an arbitrary SPARQL 1.1 query (SELECT / ASK / CONSTRUCT / DESCRIBE)
    Sparql {
        /// SPARQL query text, e.g. 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10';
        /// "-" reads the query from stdin
        query: String,
        /// Print the result rows as JSON
        #[arg(long)]
        json: bool,
    },
    /// Answer one question through the symbolic reasoner (read-only; no
    /// interactive banners). Omit the question to read one per line from stdin
    Ask {
        /// Question to answer; omit to read newline-delimited questions from stdin
        question: Option<String>,
        /// Print a JSON array of {question, status, answer}
        #[arg(long)]
        json: bool,
        /// Exit non-zero when any question is left unanswered
        #[arg(long)]
        strict: bool,
    },
    /// Trace how a question is read: tokens, POS provenance, spans, query plan
    /// and closure-marker boundary risks (offline; does not open the store)
    ReadTrace {
        /// Question to analyse (omit when using --corpus)
        #[arg(required_unless_present = "corpus", conflicts_with = "corpus")]
        question: Option<String>,
        /// UTF-8 corpus with one question per line; reports aggregated lexicon gaps
        #[arg(long)]
        corpus: Option<String>,
        /// Also run the static lexical coverage audit: flag plan slots that have
        /// no executable definition in the configured lexicon (read-only)
        #[arg(long)]
        lexicon: bool,
        /// JSON definition file (alias / compose) to admit before the audit;
        /// accepted entries resolve coverage gaps without writing the KB
        #[arg(long, requires = "lexicon")]
        definitions: Option<String>,
        /// Print the trace/report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Show the decision frontier of every routing candidate: which element
    /// blocked, what was expected/seen, and how much prefix was covered
    /// (offline; does not open the knowledge store)
    Frontier {
        /// Question to analyse (omit when using --corpus)
        #[arg(required_unless_present = "corpus", conflicts_with = "corpus")]
        question: Option<String>,
        /// UTF-8 corpus with one question per line; aggregates the closest gaps
        #[arg(long)]
        corpus: Option<String>,
        /// Coverage threshold for treating an unmatched candidate as a near miss
        #[arg(long, default_value_t = 0.75)]
        near_miss: f64,
        /// Print the frontier report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Evaluate constructed surface recall: turn every configured route into
    /// canonical/synonym/entity/suffix variants, then measure seen vs unseen
    /// recall with a regression ratchet (offline; does not open the store)
    EvalRecall {
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Coverage threshold for the near-miss slice
        #[arg(long, default_value_t = 0.75)]
        near_miss: f64,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Forge a token pattern from positive/negative examples and preview the
    /// expanded IR (offline; does not open the knowledge store)
    Forge {
        /// Human-readable rule name
        #[arg(long)]
        name: String,
        /// Query route (query_type) the forged patterns should answer
        #[arg(long)]
        route: String,
        /// Positive example (repeat for same-shape surface variants)
        #[arg(long = "positive", required = true)]
        positives: Vec<String>,
        /// Negative example that must stay excluded (repeatable)
        #[arg(long = "negative")]
        negatives: Vec<String>,
        /// Pattern priority (default 10)
        #[arg(long, default_value_t = 10)]
        priority: i32,
        /// Materialize the candidate as active patterns (preview only)
        #[arg(long)]
        activate: bool,
        /// Print the forged pattern/probes as JSON
        #[arg(long)]
        json: bool,
    },
    /// Evaluate symbolic embeddings (link prediction + analogy) with a
    /// deterministic holdout split and regression ratchet
    Eval {
        /// Test split ratio in per mille (0-1000)
        #[arg(long, default_value_t = 200)]
        test_ratio: u32,
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// RRF fusion constant
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f64,
    },
    /// Evaluate dialogue NLU (intent / entity linking / slots / clarification /
    /// answers / provenance) on the built-in gold corpus with a regression ratchet
    EvalNlu {
        /// Knowledge backend: "reasoner" (end-to-end) or "scripted" (policy layer)
        #[arg(long, default_value = "reasoner")]
        knowledge: String,
        /// Prompt asset mode: "off" (combination generator only), "parity"
        /// (dual-run comparison), or "on" (grounded templates with fallback)
        #[arg(long, default_value = "on")]
        prompt_mode: String,
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Evaluate session-level entity linking (candidates / clarification /
    /// binding memory / chain consistency) on the built-in gold corpus with a
    /// regression ratchet
    EvalLink {
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Evaluate identity acts / cards / scopes / session ledgers on the built-in
    /// gold corpus with a regression ratchet
    EvalIdentity {
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Evaluate failure recovery (silent-failure / receipt / evidence rates) on
    /// the built-in seed scenarios with a regression ratchet
    EvalRecovery {
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Run the ProofBench corpus: structured readings, proof obligations,
    /// counterfactual twins and bounded proof-distance estimates (offline;
    /// does not open the knowledge store)
    ProofBench {
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Write the auditable Prolog corpus to this path
        #[arg(long)]
        emit_prolog: Option<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Lint the prompt asset registry (executable examples + template checks)
    PromptLint {
        /// Print the lint report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Compile-check config.toml as the source of a small language: dangling
    /// references, unreachable capabilities, cycles/conflicts, out-of-range
    /// thresholds (does not touch the knowledge store)
    ConfigLint {
        /// Config file path (defaults to config.toml)
        #[arg(long)]
        file: Option<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
        /// Also fail (exit code) on warnings
        #[arg(long)]
        strict: bool,
    },
    /// Check whether declared predicate laws are actually fulfilled by the
    /// runtime: generate minimal probes from config + graph witnesses and run
    /// them through the real answer pipeline (read-only; never writes the KB)
    ConfigFulfillment {
        /// Config file path (defaults to config.toml)
        #[arg(long)]
        file: Option<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
        /// Also fail (exit code) on gaps (honest unknown / unchecked / no
        /// witness / unregistered predicate), not only on wrong answers
        #[arg(long)]
        strict: bool,
        /// Show fulfilled obligations too
        #[arg(long)]
        verbose: bool,
        /// Maximum number of obligations to generate
        #[arg(long)]
        max_obligations: Option<usize>,
    },
    /// Compile the language assets into a schema snapshot, diff it against the
    /// release lock, and run the offline release gates
    ReleaseCheck {
        /// Config file path (defaults to config.toml)
        #[arg(long, default_value = "config.toml")]
        config_file: String,
        /// Release lock path (created by --update-lock)
        #[arg(long, default_value = "release.lock.json")]
        lock: String,
        /// Write the current snapshot and evaluation baselines as the new lock
        #[arg(long)]
        update_lock: bool,
        /// Language-pack version X.Y.Z (defaults to the locked version)
        #[arg(long = "language-version")]
        language_version: Option<String>,
        /// Skip the NLU / recovery / fuzz evaluation gates
        #[arg(long)]
        skip_evals: bool,
        /// Do not fail (exit code) on gate failures
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Extract proof-carrying relation frames from text/file/corpus and gate
    /// them against the current knowledge store (read-only diagnostic)
    Frames {
        /// Text to analyse (mutually exclusive with --file)
        #[arg(long, conflicts_with = "file")]
        text: Option<String>,
        /// UTF-8 text file to analyse
        #[arg(long)]
        file: Option<String>,
        /// Maximum number of frames to report (0 = no limit)
        #[arg(long, default_value_t = 200)]
        limit: usize,
        /// Number of example frames with narration in the text report
        #[arg(long, default_value_t = 8)]
        samples: usize,
        /// Print the full report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Learn surface patterns from seed facts x corpus and propose new graph
    /// candidates with evidence families and ordinal confidence (read-only)
    BuildKg {
        /// Text to analyse (mutually exclusive with --file)
        #[arg(long, conflicts_with = "file")]
        text: Option<String>,
        /// UTF-8 text file to analyse
        #[arg(long)]
        file: Option<String>,
        /// Minimum distinct seed entity pairs before a pattern is activated
        #[arg(long, default_value_t = 2)]
        min_support: usize,
        /// Minimum context purity in per mille (0-1000)
        #[arg(long, default_value_t = 600)]
        min_purity: u32,
        /// Maximum number of candidates to report (0 = no limit)
        #[arg(long, default_value_t = 200)]
        limit: usize,
        /// Optional feedback file: lines of 「确认/否认 主语 谓词 宾语」
        #[arg(long)]
        feedback: Option<String>,
        /// Apply accepted candidates whose endpoints already exist in the graph
        #[arg(long)]
        apply: bool,
        /// Print the full report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Enumerate provenance-preserving paths between two entities, along a
    /// predicate chain, or across a shared neighbour (read-only)
    Paths {
        /// Start entity (exact name as stored)
        #[arg(long)]
        from: String,
        /// Goal entity (exact name as stored); required unless --via or
        /// --common-with is given
        #[arg(long, required_unless_present_any = ["via", "common_with"])]
        to: Option<String>,
        /// Follow a predicate chain from --from (repeatable), e.g.
        /// --via 师傅 --via 师傅
        #[arg(long = "via", value_name = "PREDICATE", conflicts_with_all = ["to", "common_with"])]
        via: Vec<String>,
        /// Find entities both --from and this entity point to
        #[arg(long = "common-with", value_name = "ENTITY", conflicts_with_all = ["to", "via"])]
        common_with: Option<String>,
        /// Maximum number of hops
        #[arg(long, default_value_t = 4)]
        max_hops: usize,
        /// Maximum number of witness paths to keep
        #[arg(long, default_value_t = 8)]
        max_paths: usize,
        /// Restrict every path step to a valid-time point/interval,
        /// e.g. 1995, 1995-06, 1995年, 从1991年到1999年
        #[arg(long = "as-of", value_name = "TIME")]
        as_of: Option<String>,
        /// Print the path bundle as JSON
        #[arg(long)]
        json: bool,
    },
    /// Retrieve ranked graph evidence (TF-IDF x structural PPR x predicate)
    /// under exact constraints, with a certificate-carrying search receipt
    Evidence {
        /// Question or keyword query
        query: String,
        /// Number of evidence rows to return
        #[arg(long, default_value_t = 8)]
        k: usize,
        /// Restrict to an exact subject
        #[arg(long)]
        subject: Option<String>,
        /// Restrict to an exact predicate
        #[arg(long)]
        predicate: Option<String>,
        /// Restrict to an exact object
        #[arg(long)]
        object: Option<String>,
        /// Restrict to entities with this declared type
        #[arg(long = "type", value_name = "TYPE")]
        entity_type: Option<String>,
        /// Maximum number of candidates to score (default: unlimited)
        #[arg(long)]
        max_scanned: Option<usize>,
        /// Print hits and the receipt as JSON
        #[arg(long)]
        json: bool,
    },
    /// Detect conflict certificates over the knowledge graph: witnessed
    /// functional-slot multi-values and disjoint types; optional fact-level
    /// provenance yields attribution and a minimal retraction plan (read-only)
    Conflicts {
        /// Fact-level provenance ledger (ContextImage JSON, section
        /// "fact_provenance"); without it every base fact is honestly
        /// reported as "来源未登记"
        #[arg(long)]
        provenance: Option<String>,
        /// Also report latent rule conflicts (rules that can clash but have
        /// no instance yet); off by default to avoid O(R²) noise
        #[arg(long)]
        latent: bool,
        /// Include assertions from fusion source namespaces (they are
        /// normally isolated from the default graph); sources become
        /// per-namespace identities instead of "未登记"
        #[arg(long)]
        namespaces: bool,
        /// Maximum certificates printed in text mode (0 = all)
        #[arg(long, default_value_t = 40)]
        limit: usize,
        /// Exit non-zero when any certificate is reported (CI gate)
        #[arg(long)]
        strict: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Inspect or extend the fact-level provenance ledger: coverage, per-fact
    /// sources, and optional TSV recording (subject/predicate/value/kind/
    /// locator[/content]); import-journal and namespace sources are merged
    /// read-only for inspection
    Provenance {
        /// TSV file of source records to append and persist
        #[arg(long)]
        record: Option<String>,
        /// Filter facts with `subject|predicate|value` (value optional)
        #[arg(long)]
        fact: Option<String>,
        /// Ledger path (defaults to <store>/provenance.json)
        #[arg(long)]
        path: Option<String>,
        /// Also include fusion source namespace assertions in coverage
        #[arg(long)]
        namespaces: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Inspect the read-only temporal index: timeline, facts valid at a time,
    /// and time-aware functional conflicts (no writes to the knowledge store)
    Temporal {
        /// Entity to inspect (omit for a whole-index summary)
        #[arg(long)]
        entity: Option<String>,
        /// Point or interval, e.g. 1868, 1868-06, 1868年, 从1991年到1999年
        #[arg(long = "at", value_name = "TIME")]
        at: Option<String>,
        /// Predicate treated as single-valued for temporal conflict checks
        /// (repeatable; defaults to capital/spouse/president-style predicates)
        #[arg(long = "functional", value_name = "PREDICATE")]
        functional: Vec<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Audit the semantic law ledger: declared predicate laws + mined rules,
    /// with functional-violation and closure-gap counters
    Laws {
        /// Print the ledger and audit as JSON
        #[arg(long)]
        json: bool,
        /// Persist the ledger to `<store>/law_book.json` (explicit adoption artifact)
        #[arg(long)]
        save: bool,
        /// Explicitly adopt a law id (repeatable); requires --save to persist
        #[arg(long = "adopt", value_name = "LAW_ID")]
        adopt: Vec<String>,
        /// Explicitly retire a law id (repeatable); requires --save to persist
        #[arg(long = "retire", value_name = "LAW_ID")]
        retire: Vec<String>,
    },
    /// Inspect the knowledge-fusion view over source namespaces (corroboration
    /// and quarantined conflicts); `--apply` materializes winners into the
    /// default graph, `--prefer PREDICATE=NAMESPACE` records an adjudication
    Fuse {
        /// Materialize the fusion view into the default graph
        #[arg(long)]
        apply: bool,
        /// Record an adjudication `PREDICATE=NAMESPACE` (repeatable)
        #[arg(long = "prefer", value_name = "PREDICATE=NAMESPACE")]
        prefer: Vec<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Import an RDF file into a named source namespace: raw assertions stay in
    /// their own named graph, the default graph is untouched until `fuse --apply`
    Namespace {
        /// Namespace id (IRI-safe slug), e.g. "alice" or "batch-2026-10"
        #[arg(long)]
        source: String,
        /// Human-readable source label (defaults to the namespace id)
        #[arg(long)]
        label: Option<String>,
        /// Trust level: verified / curated / imported / external / speculative
        #[arg(long, default_value = "imported")]
        trust: String,
        /// Namespace kind: global / import / session / hypothesis
        #[arg(long, default_value = "import")]
        kind: String,
        /// RDF file path (.ttl / .nt / .nq / .trig / .rdf / .xml)
        file_path: String,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Build the data quality passport (deterministic entity scores + gaps)
    /// with a corpus-digest regression ratchet
    EvalQuality {
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Fact-level provenance ledger (created by `provenance` tooling);
        /// when given, the provenance dimension uses per-fact source coverage
        /// instead of predicate-level source tags
        #[arg(long)]
        provenance: Option<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Validate the knowledge graph against SHACL-style shapes compiled from
    /// entity-type config, predicate profiles and the type system
    EvalShapes {
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
        /// Exit non-zero when any Violation-severity finding is reported
        #[arg(long)]
        strict: bool,
    },
    /// Evaluate shape-filling dialogue (gap / acquisition / generation / session)
    /// on the built-in scripted corpus with a regression ratchet
    EvalShapeDialogue {
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Run deterministic quality-delta dialogue fuzzing offline (no store I/O),
    /// with an optional known-gap baseline regression ratchet
    FuzzDialogue {
        /// Number of generated cases
        #[arg(long, default_value_t = 96)]
        cases: usize,
        /// Deterministic PRNG seed (same seed → byte-identical report)
        #[arg(long, default_value_t = 20261019)]
        seed: u64,
        /// Known-gap baseline JSON path (new failure signatures fail the run)
        #[arg(long)]
        baseline: Option<String>,
        /// Write the current failures as the known-gap baseline
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Audit persisted artifacts (knowledge `rustfmt --check`): canonical /
    /// non-canonical / damaged / missing, with optional atomic `--fix`
    Artifacts {
        /// Atomically rewrite non-canonical artifacts (damaged are never touched)
        #[arg(long)]
        fix: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
        /// Extra artifact paths to audit (kind inferred from file name)
        #[arg(long = "path")]
        paths: Vec<String>,
    },
    /// Inspect, lint and evaluate declarative plugin manifests
    Plugins {
        /// Override the configured plugin directory
        #[arg(long)]
        dir: Option<String>,
        /// Print the plugin report as JSON
        #[arg(long)]
        json: bool,
        /// Run every manifest's bundled eval cases
        #[arg(long)]
        eval: bool,
    },
    /// Run the session REST service (multi-turn dialogue over HTTP + SSE)
    Serve {
        /// Listen address
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Maximum resident (in-memory) sessions
        #[arg(long, default_value_t = 64)]
        max_sessions: usize,
        /// Cold-session directory: when resident capacity is full, the
        /// least-recently-used session is frozen here and revived on access
        #[arg(long)]
        session_dir: Option<String>,
        /// Prompt asset mode for sessions: "off", "parity", or "on"
        #[arg(long, default_value = "on")]
        prompt_mode: String,
        /// Failure recovery mode for sessions: "off" (text verbatim),
        /// "observe" (record receipts only), or "on" (full recovery ladder)
        #[arg(long, default_value = "observe")]
        recovery: String,
    },
    /// Replay a query-board scenario (turn-trace contract) offline; exits
    /// non-zero when any step fails
    Scenario {
        /// Scenario TOML file (facts + turns + per-turn expectations)
        file: String,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Run interactive mode (default)
    Interactive,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // 配置契约 lint 必须最先处理：它恰恰要在 config.toml 损坏时可用，
    // 因此不依赖默认配置加载，也不打开知识存储。
    if let Some(Commands::ConfigLint { file, json, strict }) = &cli.command {
        run_config_lint(Some(file.as_deref().unwrap_or(&cli.config)), *json, *strict);
        return;
    }

    // 配置兑现只读配置与图：在加载默认配置/打开交互存储之前处理，
    // 这样空图或存储路径覆盖也能得到一份可复现的报告。
    if let Some(Commands::ConfigFulfillment {
        file,
        json,
        strict,
        verbose,
        max_obligations,
    }) = &cli.command
    {
        run_config_fulfillment(
            Some(file.as_deref().unwrap_or(&cli.config)),
            cli.store.as_deref(),
            *json,
            *strict,
            *verbose,
            max_obligations.unwrap_or(fulfillment::DEFAULT_MAX_OBLIGATIONS),
        )
        .await;
        return;
    }

    // 发布检查只读配置与插件清单：在加载默认配置 / 打开存储之前处理，
    // 这样即使存储不可用也能完整跑完离线门禁。
    if let Some(Commands::ReleaseCheck {
        config_file,
        lock,
        update_lock,
        language_version,
        skip_evals,
        allow_regression,
        json,
    }) = &cli.command
    {
        run_release_check(
            if config_file == "config.toml" {
                &cli.config
            } else {
                config_file
            },
            lock,
            *update_lock,
            language_version.clone(),
            *skip_evals,
            *allow_regression,
            *json,
        );
        return;
    }

    // 质量差分模糊测试完全离线：不依赖 config.toml / 存储，先于任何 I/O 处理。
    if let Some(Commands::FuzzDialogue {
        cases,
        seed,
        baseline,
        update_baseline,
        allow_regression,
        json,
    }) = &cli.command
    {
        run_fuzz_dialogue(
            *cases,
            *seed,
            *json,
            baseline.as_deref(),
            *update_baseline,
            *allow_regression,
        );
        return;
    }

    // 查询剧本完全离线（自带事实）：先于配置加载与存储打开处理。
    if let Some(Commands::Scenario { file, json }) = &cli.command {
        run_query_scenario(file, *json);
        return;
    }

    let recovery_mode = parse_recovery_mode(&cli.recovery);

    // 启动分期台账：config_load / store_open / facts_load / warm_build / bind
    // 各相耗时只观测不断言；serve 分支打印摘要，/metrics 输出明细。
    let startup_ledger = StartupLedger::shared();
    let config_start = Instant::now();
    let mut config = match load_config_from(&cli.config).await {
        Ok(cfg) => {
            startup_ledger.record_since("config_load", config_start);
            cfg
        }
        Err(e) => {
            eprintln!("加载配置文件失败（{}）: {}", cli.config, e);
            return;
        }
    };
    // 存储路径可被 --store 覆盖（便于对独立知识库做导入/评估而不动主库）。
    // 覆盖时相对对话状态路径也要落到新存储下，否则会话记忆会跨库串台。
    let store_path = cli
        .store
        .clone()
        .unwrap_or_else(|| config.system.oxigraph_store_path.clone());
    if cli.store.is_some() {
        isolate_dialogue_state_path(&mut config.reasoning.dialogue_state_path, &store_path);
    }
    let config = Arc::new(config);
    // 分层追踪模式：命令行 `--trace` 优先，缺省回落 config.toml 的 [trace_graph]。
    let trace_mode = match cli.trace.as_deref() {
        Some(value) => parse_trace_mode(value),
        None => config.trace_graph.parse_mode(),
    };

    // 模式锻造只读配置与例子：不打开知识存储，适合脚本化预览与回归。
    if let Some(Commands::Forge {
        name,
        route,
        positives,
        negatives,
        priority,
        activate,
        json,
    }) = &cli.command
    {
        run_pattern_forge(
            &config, name, route, positives, negatives, *priority, *activate, *json,
        );
        return;
    }

    // 读法追踪只读配置与文本：不打开知识存储，适合在词表/分词变更前后做差分。
    if let Some(Commands::ReadTrace {
        question,
        corpus,
        lexicon,
        definitions,
        json,
    }) = &cli.command
    {
        run_read_trace(
            question.as_deref(),
            corpus.as_deref(),
            *lexicon,
            definitions.as_deref(),
            *json,
            &config,
        );
        return;
    }

    // 决策前沿只读配置与文本：复用运行时路由的同一匹配器，不打开存储。
    if let Some(Commands::Frontier {
        question,
        corpus,
        near_miss,
        json,
    }) = &cli.command
    {
        run_frontier(
            question.as_deref(),
            corpus.as_deref(),
            *near_miss,
            *json,
            &config,
        );
        return;
    }

    // 构念召回评估只读配置：分母由规则与同义词表生成，不打开存储。
    if let Some(Commands::EvalRecall {
        baseline,
        update_baseline,
        allow_regression,
        near_miss,
        json,
    }) = &cli.command
    {
        run_eval_recall(
            baseline.as_deref(),
            *update_baseline,
            *allow_regression,
            *near_miss,
            *json,
            &config,
        );
        return;
    }

    // 插件清点/评测同样不依赖知识存储：清单是纯声明式资产。
    if let Some(Commands::Plugins { dir, json, eval }) = &cli.command {
        run_plugins(&config.plugins, dir.as_deref(), *json, *eval);
        return;
    }

    // 无需知识存储的子命令先行处理：只跑提示词 lint / NLU 评测时，
    // 即使 Oxigraph 库路径不可用也不受影响（不创建存储、不读知识库）。
    match &cli.command {
        Some(Commands::PromptLint { json }) => {
            run_prompt_lint(*json);
            return;
        }
        Some(Commands::EvalNlu {
            knowledge,
            prompt_mode,
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_nlu(
                knowledge,
                prompt_mode,
                baseline.clone(),
                *update_baseline,
                *allow_regression,
                *json,
            );
            return;
        }
        Some(Commands::EvalLink {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_link(baseline.clone(), *update_baseline, *allow_regression, *json);
            return;
        }
        Some(Commands::EvalIdentity {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_identity(baseline.clone(), *update_baseline, *allow_regression, *json);
            return;
        }
        Some(Commands::EvalRecovery {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_recovery(baseline.clone(), *update_baseline, *allow_regression, *json);
            return;
        }
        Some(Commands::ProofBench {
            baseline,
            update_baseline,
            allow_regression,
            emit_prolog,
            json,
        }) => {
            run_proof_bench(
                baseline.clone(),
                *update_baseline,
                *allow_regression,
                emit_prolog.as_deref(),
                *json,
            );
            return;
        }
        Some(Commands::FuzzDialogue {
            cases,
            seed,
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_fuzz_dialogue(
                *cases,
                *seed,
                *json,
                baseline.as_deref(),
                *update_baseline,
                *allow_regression,
            );
            return;
        }
        _ => {}
    }

    // 会话级性能台账：存储与推理共享同一实例，报告覆盖整条链路
    let perf_ledger = PerfLedger::shared();

    // 创建Oxigraph存储
    let store_start = Instant::now();
    let oxigraph_manager = match OxigraphManager::with_perf_ledger(&store_path, perf_ledger.clone())
    {
        Ok(manager) => {
            startup_ledger.record_since("store_open", store_start);
            manager
        }
        Err(e) => {
            eprintln!("创建Oxigraph存储失败: {}", e);
            return;
        }
    };

    // 处理命令行参数
    match cli.command {
        Some(Commands::ImportJsonld { file_path }) => {
            println!("正在导入 JSON-LD 文件: {}", file_path);

            match oxigraph_manager.import_jsonld_from_file(&file_path).await {
                Ok(stats) => {
                    println!("JSON-LD 文件导入成功：{}", stats.summary());
                }
                Err(e) => {
                    eprintln!("JSON-LD 文件导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::ImportJsonldContent { content }) => {
            println!("正在导入 JSON-LD 内容...");

            match oxigraph_manager.import_jsonld(&content).await {
                Ok(stats) => {
                    println!("JSON-LD 内容导入成功：{}", stats.summary());
                }
                Err(e) => {
                    eprintln!("JSON-LD 内容导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::ImportRdf { file_path }) => {
            println!("正在导入 RDF 文件: {}", file_path);

            match oxigraph_manager.import_rdf_file(&file_path).await {
                Ok(added) => {
                    println!("RDF 文件导入成功：新增 {} 条三元组！", added);
                }
                Err(e) => {
                    eprintln!("RDF 文件导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::ExportRdf { file_path }) => {
            if file_path == "-" {
                let stdout = std::io::stdout();
                match oxigraph_manager
                    .export_rdf_to_writer(oxigraph::io::RdfFormat::Turtle, stdout.lock())
                    .await
                {
                    Ok(count) => eprintln!("已导出 {} 条三元组（Turtle → stdout）", count),
                    Err(e) => {
                        eprintln!("RDF 导出失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                match oxigraph_manager.export_rdf_file(&file_path).await {
                    Ok(count) => {
                        println!("RDF 文件导出成功：{} 条三元组 → {}", count, file_path);
                    }
                    Err(e) => {
                        eprintln!("RDF 文件导出失败: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::Sparql { query, json }) => {
            let query_text = if query == "-" {
                let mut buffer = String::new();
                if let Err(e) = std::io::stdin().read_to_string(&mut buffer) {
                    eprintln!("读取标准输入失败: {}", e);
                    std::process::exit(1);
                }
                buffer
            } else {
                query
            };
            match oxigraph_manager.query_sparql(&query_text).await {
                Ok(result) => {
                    if json {
                        match serde_json::to_string_pretty(&result) {
                            Ok(text) => println!("{}", text),
                            Err(e) => {
                                eprintln!("SPARQL 结果序列化失败: {}", e);
                                std::process::exit(1);
                            }
                        }
                    } else {
                        println!("{}", result.render_text());
                    }
                }
                Err(e) => {
                    eprintln!("SPARQL 查询失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Ask {
            question,
            json,
            strict,
        }) => {
            run_ask(
                &config,
                &oxigraph_manager,
                &store_path,
                recovery_mode,
                question,
                json,
                strict,
            )
            .await;
        }
        Some(Commands::Frames {
            text,
            file,
            limit,
            samples,
            json,
        }) => {
            // 与 Laws 相同的只读装配：先载入持久画像/律典，再从当前存储取事实。
            let registry_path = format!("{}/predicate_registry.json", store_path);
            let _ = oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let mut laws = LawBook::from_registry(&registry);
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                laws.apply_persisted(&persisted);
            }
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let source = match (text, file) {
                (Some(text), _) => text,
                (None, Some(path)) => match std::fs::read_to_string(&path) {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("读取文本失败（{}）: {}", path, e);
                        std::process::exit(1);
                    }
                },
                (None, None) => match load_frame_corpus() {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                },
            };
            let (mut frames, stats) = FrameExtractor::new().extract_with_stats(&source);
            if limit > 0 {
                frames.truncate(limit);
            }
            let gate = FrameGate::from_facts(facts.as_slice(), registry, types, laws);
            let diagnostics = diagnose_frames(&frames, &gate, samples);
            if json {
                let reports: Vec<serde_json::Value> = frames
                    .iter()
                    .map(|frame| {
                        let report = gate.evaluate(frame);
                        serde_json::json!({
                            "frame": frame,
                            "report": report,
                            "narration": narrate_frame(frame, &report),
                        })
                    })
                    .collect();
                let payload = serde_json::json!({
                    "dialect": FRAME_DIALECT,
                    "extraction": stats,
                    "gate_facts": gate.fact_count(),
                    "diagnostics": diagnostics,
                    "frames": reports,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("关系帧序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                print!("{}", diagnostics.render_text());
                println!(
                    "抽取统计：{} 句，{} 条帧，解析失败 {} 句；闸门事实 {} 条。",
                    stats.sentences,
                    stats.frames,
                    stats.parse_errors,
                    gate.fact_count()
                );
                for sample in &diagnostics.samples {
                    println!("\n{}", sample.frame.render_text());
                    println!("  证据：{}", sample.frame.evidence.sentence);
                    println!("  {}", narrate_frame(&sample.frame, &sample.report));
                }
            }
        }
        Some(Commands::BuildKg {
            text,
            file,
            min_support,
            min_purity,
            limit,
            feedback,
            apply,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            let _ = oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let mut laws = LawBook::from_registry(&registry);
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                laws.apply_persisted(&persisted);
            }
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let source = match (text, file) {
                (Some(text), _) => text,
                (None, Some(path)) => match std::fs::read_to_string(&path) {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("读取文本失败（{}）: {}", path, e);
                        std::process::exit(1);
                    }
                },
                (None, None) => match load_frame_corpus() {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                },
            };
            let options = ConstructionOptions {
                min_pattern_support: min_support,
                min_pattern_purity: min_purity.min(1000),
                max_candidates: limit,
                ..ConstructionOptions::default()
            };
            let feedback_items = match feedback {
                Some(path) => match std::fs::read_to_string(&path) {
                    Ok(text) => {
                        let (items, errors) = parse_feedback(&text);
                        for error in errors {
                            eprintln!("反馈文件 {}：{}", path, error);
                        }
                        items
                    }
                    Err(e) => {
                        eprintln!("读取反馈文件失败（{}）: {}", path, e);
                        std::process::exit(1);
                    }
                },
                None => Vec::new(),
            };
            let constructor = WeakConstructor::new(facts.as_slice(), registry, types, laws);
            let report = constructor.build_with_feedback(&source, &options, &feedback_items);
            if apply {
                let items: Vec<KnowledgeItem> = report
                    .candidates
                    .iter()
                    .filter(|candidate| {
                        candidate.decision
                            == know_prolog::weak_construction::CandidateDecision::Accepted
                            && candidate.known_subject
                            && candidate.known_object
                    })
                    .map(|candidate| KnowledgeItem::Relationship {
                        from_label: String::new(),
                        from_id: candidate.subject.clone(),
                        to_label: String::new(),
                        to_id: candidate.object.clone(),
                        rel_type: candidate.predicate.clone(),
                        properties: std::collections::HashMap::new(),
                    })
                    .collect();
                if items.is_empty() {
                    eprintln!("没有可落库的已接受候选（端点必须已在图谱中）。");
                } else {
                    match oxigraph_manager.save_knowledge_with_receipt(items).await {
                        Ok(receipt) => eprintln!("{}", receipt.summary()),
                        Err(e) => {
                            eprintln!("写入知识库失败: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
            }
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("构建报告序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", report.render_text());
            }
        }
        Some(Commands::Paths {
            from,
            to,
            via,
            common_with,
            max_hops,
            max_paths,
            as_of,
            json,
        }) => {
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let temporal_constraint = match as_of.as_deref() {
                Some(text) => match temporal::parse_time_span(text) {
                    Some(span) => TemporalConstraint::from_span(span),
                    None => {
                        eprintln!("无法解析时间「{}」（示例：1995、1995-06、1995年）", text);
                        std::process::exit(2);
                    }
                },
                None => TemporalConstraint::Any,
            };
            let mut validity_by_entity: std::collections::BTreeMap<String, Validity> =
                std::collections::BTreeMap::new();
            for fact in facts.iter() {
                if let PrologFact::Entity {
                    name, properties, ..
                } = fact
                {
                    if let Some(validity) = temporal::validity_from_properties(properties) {
                        validity_by_entity.insert(name.clone(), validity);
                        if let Some(alias) = properties.get("name") {
                            validity_by_entity.insert(alias.clone(), validity);
                        }
                    }
                }
            }
            let mut graph = TraversalGraph::new();
            for fact in facts.iter() {
                match fact {
                    PrologFact::Relation {
                        from: subject,
                        rel_type,
                        to: object,
                    } => {
                        let validity = validity_by_entity
                            .get(subject.as_str())
                            .copied()
                            .unwrap_or(Validity::Unknown);
                        graph.add_edge_record(
                            PathEdge::new(subject.as_str(), rel_type.as_str(), object.as_str())
                                .with_validity(validity),
                        );
                    }
                    PrologFact::Entity {
                        name,
                        entity_type,
                        properties,
                    } => {
                        graph.add_entity_type(name.as_str(), entity_type.as_str());
                        for (key, value) in properties {
                            if temporal::is_temporal_field(key) {
                                continue;
                            }
                            graph.add_edge_with_source(
                                name.as_str(),
                                key.as_str(),
                                value.as_str(),
                                EdgeSource::Attribute,
                            );
                        }
                    }
                }
            }
            let options = TraversalOptions {
                max_depth: max_hops.clamp(1, 8),
                max_paths: max_paths.clamp(1, 64),
                temporal: temporal_constraint,
                unknown_time: if as_of.is_some() {
                    UnknownTimePolicy::Background
                } else {
                    UnknownTimePolicy::Strict
                },
                ..TraversalOptions::default()
            };
            let bundle = if let Some(right) = common_with.as_deref() {
                graph.common_neighbors(&from, right, &options)
            } else if !via.is_empty() {
                let chain: Vec<PathPredicate> = via
                    .iter()
                    .map(|predicate| PathPredicate::forward(predicate.trim()))
                    .collect();
                let chain_options = TraversalOptions {
                    max_depth: chain.len().clamp(1, 8),
                    ..options.clone()
                };
                graph.enumerate_chain(&from, &chain, &chain_options)
            } else if let Some(goal) = to.as_deref() {
                graph.enumerate_between(&from, goal, &options)
            } else {
                eprintln!("请提供 --to、--via 或 --common-with 之一。");
                std::process::exit(2);
            };
            if json {
                match serde_json::to_string_pretty(&bundle) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("路径结果序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", bundle.render_text());
            }
        }
        Some(Commands::Evidence {
            query,
            k,
            subject,
            predicate,
            object,
            entity_type,
            max_scanned,
            json,
        }) => {
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let mut retriever = DenseRetriever::new(k, 0.85, 12);
            retriever.observe_facts(&facts);
            let constraints = SearchConstraints {
                subject: subject.map(|value| value.trim().to_string()),
                predicate: predicate.map(|value| value.trim().to_string()),
                object: object.map(|value| value.trim().to_string()),
                entity_type: entity_type.map(|value| value.trim().to_string()),
            };
            let search = retriever.search(&query, k, &constraints, SearchBudget { max_scanned });
            if json {
                let payload = serde_json::json!({
                    "query": query,
                    "constraints": constraints,
                    "hits": search.hits,
                    "receipt": search.receipt,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("证据检索结果序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", retriever.format_hits(&query, &search.hits));
                println!();
                println!("{}", search.receipt.render_text());
            }
        }
        Some(Commands::Conflicts {
            provenance,
            latent,
            namespaces,
            limit,
            strict,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let mut facts = match oxigraph_manager.load_fact_refs().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败: {}", error);
                    std::process::exit(1);
                }
            };
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let mut schema = ConflictSchema::from_registry_and_types(&registry, &types);
            for fact in &facts {
                if schema.type_predicates.contains(&fact.predicate) {
                    schema.observe_type(&types, &fact.value);
                }
            }

            let mut ledger = ProvenanceLedger::new();
            // 跨会话导入日志：给文本导入的事实补上「第几笔 + 内容摘要」身份。
            let journal_path = format!("{}/import_journal.json", store_path);
            if let JournalLoad::Loaded(journal) = ImportJournal::load(&journal_path) {
                ledger.seed_from_import_journal(&journal);
            }
            if namespaces {
                match oxigraph_manager.load_fusion_ledger().await {
                    Ok(fusion) => {
                        ledger.seed_from_fusion_ledger(&fusion);
                        for assertion in &fusion.assertions {
                            facts.push(provenance::FactRef::new(
                                &assertion.subject,
                                &assertion.predicate,
                                &assertion.object,
                            ));
                        }
                        facts.sort();
                        facts.dedup();
                        for fact in &facts {
                            if schema.type_predicates.contains(&fact.predicate) {
                                schema.observe_type(&types, &fact.value);
                            }
                        }
                    }
                    Err(error) => eprintln!("读取融合账本失败：{}", error),
                }
            }
            let default_ledger_path = format!("{}/provenance.json", store_path);
            if let ProvenanceLoad::Loaded(loaded) = ProvenanceLedger::load(&default_ledger_path) {
                ledger.merge(&loaded);
            }
            if let Some(path) = provenance.as_deref() {
                if path != default_ledger_path {
                    match ProvenanceLedger::load(path) {
                        ProvenanceLoad::Loaded(loaded) => ledger.merge(&loaded),
                        ProvenanceLoad::Missing => {
                            eprintln!("来源台账不存在：{}（按已回填来源报告）", path)
                        }
                        ProvenanceLoad::Damaged(error) => {
                            eprintln!("来源台账损坏：{}（按已回填来源报告）", error)
                        }
                    }
                }
            }
            let rules = oxigraph_manager
                .load_rule_ledger()
                .await
                .unwrap_or_else(|error| {
                    eprintln!("读取规则账本失败：{}", error);
                    know_prolog::graph_ledger::RuleLedger {
                        synced: false,
                        fingerprint: String::new(),
                        entries: Vec::new(),
                    }
                });
            let derived = oxigraph_manager
                .load_derived_ledger()
                .await
                .unwrap_or_else(|error| {
                    eprintln!("读取派生账本失败：{}", error);
                    know_prolog::graph_ledger::DerivedLedger {
                        materialized: false,
                        revision: 0,
                        fingerprint: String::new(),
                        entries: Vec::new(),
                    }
                });
            let supports = provenance::build_supports(&facts, &ledger, &rules, &derived);
            // 派生事实通常只在命名图里：把它们也纳入已见证冲突的候选结论。
            facts.extend(supports.values().map(|support| support.fact.clone()));
            facts.sort();
            facts.dedup();
            let mut certificates = provenance::detect_witnessed(&facts, &supports, &schema);
            if latent {
                let clauses: Vec<know_prolog::resolution_engine::Clause> =
                    rules.entries.iter().map(ledger_entry_to_clause).collect();
                certificates.extend(provenance::detect_latent(&clauses, &schema));
            }
            let coverage = provenance::Coverage::from_supports(&facts, &supports);
            let report = ConflictReport::new(certificates, coverage);
            if json {
                println!("{}", report.to_json());
            } else {
                let mut shown = report.clone();
                if limit > 0 && shown.certificates.len() > limit {
                    shown.certificates.truncate(limit);
                }
                println!("{}", shown.render_text());
                if shown.certificates.len() < report.certificates.len() {
                    println!(
                        "（仅显示前 {} 个证书；用 --limit 0 查看全部）",
                        shown.certificates.len()
                    );
                }
            }
            if strict && !report.certificates.is_empty() {
                std::process::exit(1);
            }
        }
        Some(Commands::Provenance {
            record,
            fact,
            path,
            namespaces,
            json,
        }) => {
            let ledger_path = path
                .clone()
                .unwrap_or_else(|| format!("{}/provenance.json", store_path));
            let mut ledger = ProvenanceLedger::new();
            match ProvenanceLedger::load(&ledger_path) {
                ProvenanceLoad::Loaded(loaded) => ledger = loaded,
                ProvenanceLoad::Missing => {}
                ProvenanceLoad::Damaged(error) => {
                    eprintln!("来源台账损坏：{}", error);
                    if record.is_some() {
                        eprintln!("拒绝在损坏台账上追加记录；请先修复或另用 --path。");
                        std::process::exit(1);
                    }
                }
            }
            if let Some(file) = record.as_deref() {
                let text = match std::fs::read_to_string(file) {
                    Ok(text) => text,
                    Err(error) => {
                        eprintln!("读取来源记录失败：{}", error);
                        std::process::exit(1);
                    }
                };
                let rows = match parse_provenance_tsv(&text) {
                    Ok(rows) => rows,
                    Err(error) => {
                        eprintln!("来源记录格式错误：{}", error);
                        std::process::exit(2);
                    }
                };
                let mut added = 0usize;
                for (fact, source) in rows {
                    if ledger.record(fact, source) {
                        added += 1;
                    }
                }
                if let Err(error) = ledger.save(&ledger_path) {
                    eprintln!("来源台账写入失败：{}", error);
                    std::process::exit(1);
                }
                println!(
                    "来源台账已登记 {} 条新来源（{}，共 {} 条事实有登记）。",
                    added,
                    ledger_path,
                    ledger.len()
                );
            }

            // 只读视图：台账 + 导入日志 + 可选命名空间（不回写磁盘）。
            let mut view = ledger.clone();
            let journal_path = format!("{}/import_journal.json", store_path);
            if let JournalLoad::Loaded(journal) = ImportJournal::load(&journal_path) {
                view.seed_from_import_journal(&journal);
            }
            let mut facts = match oxigraph_manager.load_fact_refs().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败：{}", error);
                    std::process::exit(1);
                }
            };
            if namespaces {
                match oxigraph_manager.load_fusion_ledger().await {
                    Ok(fusion) => {
                        view.seed_from_fusion_ledger(&fusion);
                        for assertion in &fusion.assertions {
                            facts.push(provenance::FactRef::new(
                                &assertion.subject,
                                &assertion.predicate,
                                &assertion.object,
                            ));
                        }
                        facts.sort();
                        facts.dedup();
                    }
                    Err(error) => eprintln!("读取融合账本失败：{}", error),
                }
            }
            let coverage = view.coverage(&facts);
            if let Some(selector) = fact.as_deref() {
                let parts: Vec<&str> = selector.split('|').collect();
                if parts.is_empty() || parts.len() > 3 || parts[0].trim().is_empty() {
                    eprintln!("--fact 格式应为 subject|predicate|value（value 可省略）");
                    std::process::exit(2);
                }
                let matched: Vec<&provenance::FactRef> = facts
                    .iter()
                    .filter(|candidate| {
                        candidate.subject == parts[0]
                            && (parts.len() < 2 || candidate.predicate == parts[1])
                            && (parts.len() < 3 || candidate.value == parts[2])
                    })
                    .collect();
                if json {
                    let rows: Vec<serde_json::Value> = matched
                        .iter()
                        .map(|candidate| {
                            let sources: Vec<String> = view
                                .sources(candidate)
                                .iter()
                                .map(|source| source.render())
                                .collect();
                            serde_json::json!({
                                "fact": candidate,
                                "display": candidate.display(),
                                "sources": sources,
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into())
                    );
                } else if matched.is_empty() {
                    println!("没有匹配的事实：{}", selector);
                } else {
                    for candidate in matched {
                        let sources = view.sources(candidate);
                        let rendered: Vec<String> =
                            sources.iter().map(|source| source.render()).collect();
                        println!(
                            "{}：{}",
                            candidate.display(),
                            if rendered.is_empty() {
                                "来源未登记".to_string()
                            } else {
                                rendered.join("、")
                            }
                        );
                    }
                }
            } else if json {
                let value = serde_json::json!({
                    "ledger_path": ledger_path,
                    "ledger_entries": ledger.len(),
                    "view_entries": view.len(),
                    "digest": ledger.digest(),
                    "coverage": coverage,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())
                );
            } else {
                println!(
                    "来源台账：{} 条事实有登记（{}）；导入日志/命名空间回填只用于本次查看，不写盘。",
                    view.len(),
                    ledger_path
                );
                println!("{}", coverage.render_zh());
            }
        }
        Some(Commands::Temporal {
            entity,
            at,
            functional,
            json,
        }) => {
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let index = TemporalIndex::from_prolog_facts(&facts);
            let at_span = match at.as_deref() {
                Some(text) => match temporal::parse_time_span(text) {
                    Some(span) => Some(span),
                    None => {
                        eprintln!("无法解析时间「{}」（示例：1868、1868-06、1868年）", text);
                        std::process::exit(2);
                    }
                },
                None => None,
            };
            let at_facts: Vec<_> = at_span
                .as_ref()
                .map(|span| index.overlapping(span))
                .unwrap_or_default();
            let entity_facts = entity
                .as_deref()
                .map(|name| index.for_entity(name))
                .unwrap_or_default();
            let functional: std::collections::BTreeSet<String> = if functional.is_empty() {
                ["capital", "spouse", "president", "首都", "配偶", "总统"]
                    .iter()
                    .map(|name| name.to_string())
                    .collect()
            } else {
                functional.into_iter().collect()
            };
            let conflicts = index.functional_conflicts(&functional);
            if json {
                let payload = serde_json::json!({
                    "dialect": know_prolog::temporal::TEMPORAL_DIALECT,
                    "facts": index.len(),
                    "nodes": index.nodes().len(),
                    "unknown_time": index.unknown_time_facts().len(),
                    "entity": entity,
                    "entity_facts": entity_facts,
                    "at": at,
                    "at_facts": at_facts,
                    "functional_predicates": functional,
                    "conflicts": conflicts,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("时序索引序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!(
                    "时序索引：{} 条关系，{} 个实体，{} 条时间未知。",
                    index.len(),
                    index.nodes().len(),
                    index.unknown_time_facts().len()
                );
                for fact in index.timeline() {
                    println!("  {}", fact.render());
                }
                if let Some(name) = entity.as_deref() {
                    println!("\n实体「{}」：{} 条相关事实。", name, entity_facts.len());
                    for fact in &entity_facts {
                        println!("  {}", fact.render());
                    }
                }
                if let Some(text) = at.as_deref() {
                    println!("\n时间「{}」内成立：{} 条。", text, at_facts.len());
                    for fact in &at_facts {
                        println!("  {}", fact.render());
                    }
                }
                if conflicts.is_empty() {
                    println!("\n时序单值冲突：无。");
                } else {
                    println!("\n时序单值冲突：{} 处。", conflicts.len());
                    for conflict in &conflicts {
                        println!("  {}", conflict.render());
                    }
                }
            }
        }
        Some(Commands::Laws {
            json,
            save,
            adopt,
            retire,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            let _ = oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            if let Err(e) = oxigraph_manager.refresh_predicate_registry().await {
                eprintln!("刷新谓词画像失败: {}", e);
                std::process::exit(1);
            }
            let relations = match oxigraph_manager.load_all_relations().await {
                Ok(relations) => relations,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let triples: Vec<LawTriple> = relations
                .iter()
                .filter_map(|fact| match fact {
                    PrologFact::Relation { from, rel_type, to } => {
                        Some(LawTriple::new(from, rel_type, to))
                    }
                    _ => None,
                })
                .collect();
            let mut predictor = LinkPredictor::new();
            for triple in &triples {
                predictor.add_triple(&triple.subject, &triple.predicate, &triple.object);
            }
            predictor.mine_rules(
                config.reasoning.link_prediction_min_support,
                config.reasoning.link_prediction_min_confidence,
            );
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let path = format!("{}/law_book.json", store_path);
            let mut book = LawBook::from_registry(&registry);
            if let Some(persisted) = LawBook::load_from_path(&path) {
                let changed = book.apply_persisted(&persisted);
                if changed > 0 && !json {
                    eprintln!("载入持久化律典：{} 条显式决定生效", changed);
                }
            }
            book.add_mined_rules(predictor.rules());

            let mut unknown: Vec<&str> = Vec::new();
            for id in adopt.iter().chain(retire.iter()) {
                if book.law(id).is_none() {
                    unknown.push(id);
                }
            }
            if !unknown.is_empty() {
                eprintln!("未知定律 id：{}", unknown.join(", "));
                std::process::exit(1);
            }
            let adopted = adopt.iter().filter(|id| book.adopt(id)).count();
            let retired = retire.iter().filter(|id| book.retire(id)).count();
            if (adopted > 0 || retired > 0) && !save {
                eprintln!("提示：未加 --save，采纳/退役决定仅在本次运行生效");
            }
            let audit = book.audit(&triples);
            if save {
                match book.save_to_path(&path) {
                    Ok(()) => eprintln!(
                        "律典已写入 {}（采纳 {} 条，退役 {} 条）",
                        path, adopted, retired
                    ),
                    Err(e) => {
                        eprintln!("律典写入失败: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            if json {
                let payload = serde_json::json!({ "book": &book, "audit": &audit });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("律典序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                print!("{}{}", book.render_text(), audit.render_text());
            }
        }
        Some(Commands::Eval {
            test_ratio,
            baseline,
            update_baseline,
            allow_regression,
            rrf_k,
        }) => match oxigraph_manager.load_all_relations().await {
            Ok(relations) => {
                let triples = triples_from_prolog_facts(&relations);
                if triples.len() < 2 {
                    eprintln!(
                        "知识库中的关系三元组不足（{} 条），无法评估。可先导入 JSON-LD 知识库。",
                        triples.len()
                    );
                    std::process::exit(1);
                }
                let eval_config = EvalConfig {
                    test_ratio_per_mille: test_ratio,
                    rrf_k: rrf_k.max(1.0),
                    ..Default::default()
                };
                let report = evaluate(&triples, &eval_config);
                println!("{}", report.render_text(&eval_config));

                let baseline_path = baseline
                    .clone()
                    .unwrap_or_else(|| "eval_baseline.json".to_string());
                if update_baseline {
                    let new_baseline = EvalBaseline::from_report(&report);
                    match new_baseline.save(&baseline_path) {
                        Ok(()) => println!("基线已写入 {}", baseline_path),
                        Err(e) => {
                            eprintln!("基线写入失败: {}", e);
                            std::process::exit(1);
                        }
                    }
                } else if baseline.is_some() {
                    match EvalBaseline::load(&baseline_path) {
                        Ok(old_baseline) => {
                            let verdict = compare_baseline(&old_baseline, &report, 1e-6);
                            print!("{}", verdict.render_text());
                            if !verdict.passed && !allow_regression {
                                std::process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("基线读取失败: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("读取知识库失败: {}", e);
                std::process::exit(1);
            }
        },
        // 已在打开存储前处理（见 main 顶部早期分派，无需知识存储）
        Some(Commands::EvalNlu { .. }) => {}
        Some(Commands::EvalLink { .. }) => {}
        Some(Commands::EvalIdentity { .. }) => {}
        Some(Commands::EvalRecovery { .. }) => {}
        Some(Commands::ProofBench { .. }) => {}
        Some(Commands::ReleaseCheck { .. }) => {}
        // 已在打开存储前处理（见 main 顶部早期分派）
        Some(Commands::PromptLint { .. }) => {}
        // 已在打开存储前处理（见 main 顶部早期分派，无需知识存储）
        Some(Commands::ReadTrace { .. }) => {}
        Some(Commands::Frontier { .. }) => {}
        Some(Commands::EvalRecall { .. }) => {}
        Some(Commands::Forge { .. }) => {}
        // 已在加载配置前处理（见 main 顶部早期分派）
        Some(Commands::ConfigLint { .. }) => {}
        // 已在加载默认配置前处理（见 main 顶部早期分派）
        Some(Commands::ConfigFulfillment { .. }) => {}
        Some(Commands::EvalQuality {
            baseline,
            update_baseline,
            allow_regression,
            provenance,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let ledger =
                provenance
                    .as_deref()
                    .and_then(|path| match ProvenanceLedger::load(path) {
                        ProvenanceLoad::Loaded(ledger) => Some(ledger),
                        ProvenanceLoad::Missing => {
                            eprintln!("来源台账不存在：{}（按未登记来源计分）", path);
                            None
                        }
                        ProvenanceLoad::Damaged(error) => {
                            eprintln!("来源台账损坏：{}（按未登记来源计分）", error);
                            None
                        }
                    });
            let report = match quality_report_for_with_provenance(
                &oxigraph_manager,
                &config,
                ledger.as_ref(),
                Some(&store_path),
            )
            .await
            {
                Ok(report) => report,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("质量报告序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", report.render_text());
            }

            let baseline_path = baseline
                .clone()
                .unwrap_or_else(|| "quality_baseline.json".to_string());
            if update_baseline {
                let new_baseline = QualityBaseline::from_report(&report);
                match new_baseline.save(&baseline_path) {
                    Ok(()) => println!("质量基线已写入 {}", baseline_path),
                    Err(e) => {
                        eprintln!("基线写入失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else if baseline.is_some() {
                match QualityBaseline::load(&baseline_path) {
                    Ok(old_baseline) => {
                        let verdict = compare_quality_baseline(&old_baseline, &report, 1e-6);
                        print!("{}", verdict.render_text());
                        if !verdict.passed && !allow_regression {
                            std::process::exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("基线读取失败: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::EvalShapes { json, strict }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            match shape_report_for(&oxigraph_manager, &config).await {
                Ok((_, report)) => {
                    if json {
                        match serde_json::to_string_pretty(&report) {
                            Ok(text) => println!("{}", text),
                            Err(e) => {
                                eprintln!("形状报告序列化失败: {}", e);
                                std::process::exit(1);
                            }
                        }
                    } else {
                        println!("{}", report.render_text());
                    }
                    if strict && !report.conforms {
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::EvalShapeDialogue {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            let report = evaluate_shape_dialogue();
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(error) => {
                        eprintln!("形状对话评测序列化失败: {}", error);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", report.render_text());
            }
            if update_baseline {
                let path = baseline
                    .clone()
                    .unwrap_or_else(|| "shape_dialogue_baseline.json".to_string());
                let baseline_value = ShapeEvalBaseline::from_report(&report);
                match baseline_value.save(&path) {
                    Ok(_) => println!("评测基线已写入 {}", path),
                    Err(error) => {
                        eprintln!("评测基线写入失败: {}", error);
                        std::process::exit(1);
                    }
                }
            } else if let Some(path) = baseline.as_deref() {
                match ShapeEvalBaseline::load(path) {
                    Ok(old_baseline) => {
                        let verdict = compare_shape_eval_baseline(&old_baseline, &report, 1e-9);
                        println!("{}", verdict.render_text());
                        if !verdict.passed && !allow_regression {
                            std::process::exit(1);
                        }
                    }
                    Err(error) => {
                        eprintln!("基线读取失败: {}", error);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::Fuse {
            apply,
            prefer,
            json,
        }) => {
            if let Err(e) = run_fuse(&oxigraph_manager, &config, apply, &prefer, json).await {
                eprintln!("知识融合失败: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Namespace {
            source,
            label,
            trust,
            kind,
            file_path,
            json,
        }) => {
            if let Err(e) = run_namespace_import(
                &oxigraph_manager,
                &source,
                label.as_deref(),
                &trust,
                &kind,
                &file_path,
                json,
            )
            .await
            {
                eprintln!("命名空间导入失败: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Artifacts { fix, json, paths }) => {
            run_artifact_audit(&config, &store_path, fix, json, &paths);
        }
        Some(Commands::Serve {
            addr,
            max_sessions,
            session_dir,
            prompt_mode,
            recovery,
        }) => {
            let prompt_mode = parse_prompt_mode(&prompt_mode);
            let recovery_mode = parse_recovery_mode(&recovery);
            warn_config_contract(&config);
            let frame_program = compile_frame_program(&config);
            let facts_start = Instant::now();
            let facts = match serve_facts(&oxigraph_manager).await {
                Ok(facts) => {
                    startup_ledger.record_since("facts_load", facts_start);
                    facts
                }
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let synonyms = if let Some(synonyms_config) = &config.synonyms {
                synonyms_config.mappings.clone()
            } else {
                create_default_synonyms()
            };
            // 质量护照索引：与交互模式同一确定性构建路径，供每个会话的
            // 「带出处 / 披露冲突 / 追问缺口」话语行为使用（只读派生，不写图谱）。
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            // 实体身份注册表：CLI 里签过字的持久别名同样要对齐进服务暖基座，
            // 否则同一个指称在命令行能解、走 HTTP 却查无此人。
            let entity_registry_path = format!("{}/entity_registry.json", store_path);
            if oxigraph_manager.load_entity_registry(&entity_registry_path) {
                println!("已恢复实体身份注册表（服务会话同源携带）。");
            }
            oxigraph_manager.configure_entities(&config.entity_profiles);
            oxigraph_manager.configure_type_system(&config.type_system);
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let ledger = provenance_seed_for_store(Some(&store_path), None);
            let mut quality_report = build_report_with_provenance(
                &facts,
                &registry,
                &types,
                &config.entity_types,
                Some(&ledger),
            );
            if let Ok(fact_refs) = oxigraph_manager.load_fact_refs().await {
                know_prolog::quality::apply_witnessed_conflicts(
                    &mut quality_report,
                    &fact_refs,
                    Some(&ledger),
                    &registry,
                    &types,
                );
            }
            let quality_index = QualityIndex::from_report_with_provenance(
                &quality_report,
                &facts,
                &registry,
                Some(&ledger),
            );
            println!(
                "KnowProlog 会话服务已启动：http://{}（{} 条事实，最多 {} 个驻留会话{}，提示模式 {}，失败恢复 {}）",
                addr,
                facts.len(),
                max_sessions,
                session_dir
                    .as_deref()
                    .map(|dir| format!("，冷会话目录 {}", dir))
                    .unwrap_or_default(),
                prompt_mode.label(),
                recovery_mode.label()
            );
            let mut api_state = know_prolog::serve_api::ApiState::new(
                facts,
                config.reasoning.clone(),
                synonyms,
                perf_ledger.clone(),
                max_sessions,
                startup_ledger.clone(),
            );
            // 读法观测：服务会话的失败轮同样留下有界回执，指标进共享性能台账。
            api_state.set_nlp_config(config.nlp.clone());
            api_state.set_frame_program(frame_program.clone());
            // 单一本体：配置态本体注入暖基座，之后分叉的每个会话同源携带。
            let api_config_lessons = api_state.configure_ontology(&OntologyProgram::from_config(
                &config.type_system,
                &config.predicate_profiles,
            ));
            if api_config_lessons > 0 {
                println!("服务已注入配置本体公理：{} 条。", api_config_lessons);
            }
            // 知识对齐：谓词别名/换向词面 + 画像律典 + 实体持久别名 + 律典的
            // 跨重启采纳/退役决定，与交互启动同一套注册入口，建会话前完成。
            let aligned_predicates = api_state
                .align_predicate_pairs(&oxigraph_manager.predicate_query_pairs(), &registry);
            let aligned_entities =
                api_state.align_entity_aliases(&oxigraph_manager.confirmed_entity_aliases());
            if aligned_predicates > 0 || aligned_entities > 0 {
                println!(
                    "服务已对齐知识注册表：{} 条谓词别名/换向词面、{} 条实体别名。",
                    aligned_predicates, aligned_entities
                );
            }
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                let decided = api_state.load_law_book(persisted);
                if decided > 0 {
                    println!("服务已载入律典显式决定：{} 条。", decided);
                }
            }
            let quality_index = quality_index.with_revision(api_state.warm_revision());
            api_state.set_quality_index(Some(Arc::new(quality_index)));
            if let Some(dir) = session_dir {
                api_state.set_session_dir(dir);
            }
            api_state.set_prompt_mode(prompt_mode);
            api_state.set_recovery_mode(recovery_mode);
            // 服务级插件运行时：多会话共享改写/配方/模板/风格，模式取自
            // 配置与 <store>/plugin_state.json；HTTP 不提供热切换。
            let plugin_state_path = format!("{}/plugin_state.json", store_path);
            let plugin_registry = PluginRegistry::load_dir(&config.plugins.directory);
            let mut plugin_runtime = PluginRuntime::with_settings(plugin_registry, &config.plugins);
            if let Some(state) = PluginSessionState::load(&plugin_state_path) {
                plugin_runtime.restore_session_state(state);
            }
            if plugin_runtime.installed_count() > 0 {
                println!(
                    "插件目录「{}」：已安装 {} 个，启用 {} 个（服务级共享）。",
                    config.plugins.directory,
                    plugin_runtime.installed_count(),
                    plugin_runtime.enabled_count()
                );
            }
            let plugin_receipt = api_state.configure_plugin_runtime(&plugin_runtime);
            if !plugin_receipt.is_empty() {
                println!(
                    "插件本体已并入服务统一推理器：{} 条公理、{} 条别名、{} 条逆向词面。",
                    plugin_receipt.lessons, plugin_receipt.aliases, plugin_receipt.inverse_surfaces
                );
            }
            let plugin_runtime = (plugin_runtime.installed_count() > 0)
                .then(|| Arc::new(parking_lot::Mutex::new(plugin_runtime)));
            api_state.set_plugin_runtime(plugin_runtime);
            let state = Arc::new(api_state);
            if let Err(e) = know_prolog::serve_api::serve(state, &addr).await {
                eprintln!("服务退出: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Plugins { .. }) => {
            // 已在打开存储前处理；此处仅保持 match 穷尽。
        }
        Some(Commands::FuzzDialogue { .. }) => {
            // 已在打开存储前处理；此处仅保持 match 穷尽。
        }
        Some(Commands::Scenario { .. }) => {
            // 已在打开存储前处理；此处仅保持 match 穷尽。
        }
        Some(Commands::Interactive) | None => {
            // 运行交互模式
            run_interactive_mode(
                config,
                oxigraph_manager,
                perf_ledger,
                recovery_mode,
                trace_mode,
                &store_path,
            )
            .await;
        }
    }
}

/// 默认帧诊断语料：仓库根目录随附的文本知识库（按文件名排序，缺失即跳过）。
fn load_frame_corpus() -> Result<String, String> {
    const FILES: &[&str] = &[
        "additional_knowledge.txt",
        "basic_knowledge.txt",
        "comprehensive_knowledge_expansion.txt",
        "knowledge_base.txt",
        "knowledge_base_expanded.txt",
    ];
    let mut material = String::new();
    let mut used = 0usize;
    for path in FILES {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                material.push_str(&content);
                if !material.ends_with('\n') {
                    material.push('\n');
                }
                used += 1;
            }
            Err(_) => continue,
        }
    }
    if used == 0 {
        return Err("未找到默认语料；请用 --text 或 --file 指定要诊断的文本".to_string());
    }
    Ok(material)
}

/// `eval-nlu`：对话 NLU 金标评测 + 回归棘轮（不依赖知识存储）。
fn run_eval_nlu(
    knowledge: &str,
    prompt_mode: &str,
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let mode = match knowledge {
        "scripted" => EvalKnowledge::Scripted,
        "reasoner" => EvalKnowledge::Reasoner,
        other => {
            eprintln!("未知的知识后端「{}」：可选 reasoner / scripted", other);
            std::process::exit(1);
        }
    };
    let prompt_mode = parse_prompt_mode(prompt_mode);
    let report = evaluate_seed_with(mode, prompt_mode);
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("报告序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "nlu_eval_baseline.json".to_string());
    if update_baseline {
        let new_baseline = DialogueEvalBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("对话 NLU 基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match DialogueEvalBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict = compare_dialogue_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("基线读取失败: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// `eval-link`：会话级实体链接金标评测 + 回归棘轮（不依赖知识存储）。
fn run_eval_link(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let report = know_prolog::link_eval::evaluate_seed();
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("链接评测序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "link_eval_baseline.json".to_string());
    if update_baseline {
        let new_baseline = know_prolog::link_eval::LinkEvalBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("链接基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match know_prolog::link_eval::LinkEvalBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict =
                    know_prolog::link_eval::compare_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("基线读取失败: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// `eval-identity`：身份作用域 / 身份卡 / 会话台账金标评测 + 回归棘轮。
fn run_eval_identity(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let report = evaluate_identity();
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("身份评测序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "identity_eval_baseline.json".to_string());
    if update_baseline {
        let new_baseline = IdentityEvalBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("身份基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match IdentityEvalBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict = compare_identity_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("基线读取失败: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// `eval-recovery`：失败恢复金标评测 + 回归棘轮（不依赖知识存储）。
fn run_eval_recovery(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let report = know_prolog::eval_recovery::evaluate_seed();
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("恢复评测序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "recovery_baseline.json".to_string());
    if update_baseline {
        let new_baseline = know_prolog::eval_recovery::EvalRecoveryBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("恢复基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match know_prolog::eval_recovery::EvalRecoveryBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict =
                    know_prolog::eval_recovery::compare_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("基线读取失败: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// `proof-bench`：证明义务 / 反事实孪生 / 证距评测 + 回归棘轮（不依赖知识存储）。
fn run_proof_bench(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    emit_prolog: Option<&str>,
    json: bool,
) {
    let report = proof_bench::evaluate_seed();
    if let Some(path) = emit_prolog {
        let corpus = proof_bench::render_prolog(proof_bench::seed_corpus());
        match std::fs::write(path, corpus) {
            Ok(()) => {
                if !json {
                    println!("可审计 Prolog 语料已写入 {}", path);
                }
            }
            Err(error) => {
                eprintln!("Prolog 语料写入失败: {}", error);
                std::process::exit(1);
            }
        }
    }
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("ProofBench 报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "proof_bench_baseline.json".to_string());
    if update_baseline {
        let new_baseline = ProofBenchBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("ProofBench 基线已写入 {}", baseline_path),
            Err(error) => {
                eprintln!("基线写入失败: {}", error);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match ProofBenchBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict = proof_bench::compare_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(error) => {
                eprintln!("基线读取失败: {}", error);
                std::process::exit(1);
            }
        }
    }
}

/// 解析 `--prompt-mode`：off / parity / on，非法值立即退出。
fn parse_prompt_mode(value: &str) -> PromptMode {
    PromptMode::parse(value).unwrap_or_else(|| {
        eprintln!("未知的提示模式「{}」：可选 off / parity / on", value);
        std::process::exit(1);
    })
}

/// 解析 `--recovery`：off / observe / on，非法值立即退出。
fn parse_recovery_mode(value: &str) -> RecoveryMode {
    RecoveryMode::parse(value).unwrap_or_else(|| {
        eprintln!("未知的失败恢复模式「{}」：可选 off / observe / on", value);
        std::process::exit(1);
    })
}

fn parse_trace_mode(value: &str) -> TraceMode {
    TraceMode::parse(value).unwrap_or_else(|| {
        eprintln!("未知的分层追踪模式「{}」：可选 off / observe / on", value);
        std::process::exit(1);
    })
}

/// `--store` 覆盖存储时，把相对的对话状态路径重定位到新存储根目录。
///
/// 配置默认写成 `oxigraph_data/dialogue_state.json`；若原样保留，独立
/// 存储下的会话会与主库共享记忆（焦点、槽账本、改口史全部串台）。绝对
/// 路径视为用户显式指定，保持不动。
fn isolate_dialogue_state_path(path: &mut Option<String>, store_path: &str) {
    let Some(current) = path.as_deref() else {
        return;
    };
    if current.trim().is_empty() || std::path::Path::new(current).is_absolute() {
        return;
    }
    let file = std::path::Path::new(current)
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("dialogue_state.json"));
    *path = Some(
        std::path::Path::new(store_path)
            .join(file)
            .to_string_lossy()
            .into_owned(),
    );
}

/// `artifacts`：工件审计（知识版 `rustfmt --check`）。
///
/// 扫描已知持久化工件，按「解码 → 规范重编码 → 与磁盘比对」分类；
/// `--fix` 原子重写非规范工件（损坏工件一律不触碰）。存在损坏或未修好的
/// 非规范工件时以非零码退出，可直接用于 CI 巡检。
fn run_artifact_audit(
    config: &know_prolog::SystemConfig,
    store_path: &str,
    fix: bool,
    json: bool,
    extra_paths: &[String],
) {
    use know_prolog::artifact_audit::{self, ArtifactKind};

    let mut targets: std::collections::BTreeMap<String, bool> = std::collections::BTreeMap::new();
    for path in extra_paths {
        targets.insert(path.clone(), true);
    }
    let mut defaults = vec![
        format!("{}/import_journal.json", store_path),
        format!("{}/external_ledger.json", store_path),
        format!("{}/predicate_registry.json", store_path),
        format!("{}/entity_registry.json", store_path),
        format!("{}/eval_baseline.json", store_path),
        format!("{}/nlu_eval_baseline.json", store_path),
        format!("{}/recovery_baseline.json", store_path),
        format!("{}/quality_baseline.json", store_path),
    ];
    if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
        defaults.push(state_path.to_string());
        defaults.push(format!("{}.manager.json", state_path));
        defaults.push(format!("{}.chronicle.jsonl", state_path));
    } else {
        defaults.push(format!("{}/dialogue.chronicle.jsonl", store_path));
    }
    for path in defaults {
        targets.entry(path).or_insert(false);
    }

    let mut audits = Vec::new();
    for (path, explicit) in &targets {
        let kind = ArtifactKind::infer(path);
        let audit = if fix {
            match artifact_audit::fix(path, kind) {
                Ok(audit) => audit,
                Err(error) => {
                    eprintln!("工件修复被拒：{}", error);
                    artifact_audit::audit(path, kind)
                }
            }
        } else {
            artifact_audit::audit(path, kind)
        };
        if audit.status == artifact_audit::ArtifactStatus::Missing && !explicit {
            continue;
        }
        audits.push(audit);
    }

    let (mut canonical, mut non_canonical, mut repairable, mut damaged, mut missing) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    for audit in &audits {
        match audit.status {
            artifact_audit::ArtifactStatus::Canonical => canonical += 1,
            artifact_audit::ArtifactStatus::NonCanonical => non_canonical += 1,
            artifact_audit::ArtifactStatus::Repairable(_) => repairable += 1,
            artifact_audit::ArtifactStatus::Damaged(_) => damaged += 1,
            artifact_audit::ArtifactStatus::Missing => missing += 1,
        }
    }

    if json {
        let value = serde_json::json!({
            "object": "artifacts.audit",
            "fix": fix,
            "total": audits.len(),
            "canonical": canonical,
            "non_canonical": non_canonical,
            "repairable": repairable,
            "damaged": damaged,
            "missing": missing,
            "artifacts": audits
                .iter()
                .map(|audit| serde_json::json!({
                    "path": audit.path,
                    "kind": audit.kind.label(),
                    "status": audit.status.label(),
                    "detail": audit.status.detail(),
                }))
                .collect::<Vec<_>>(),
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("工件审计序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "工件审计：共 {} 件（规范 {}，非规范 {}，可截尾 {}，损坏 {}，缺失 {}）。",
            audits.len(),
            canonical,
            non_canonical,
            repairable,
            damaged,
            missing
        );
        for audit in &audits {
            let detail = audit.status.detail();
            if detail.is_empty() {
                println!(
                    "- {} [{}] {}",
                    audit.path,
                    audit.kind.label(),
                    audit.status.label()
                );
            } else {
                println!(
                    "- {} [{}] {} —— {}",
                    audit.path,
                    audit.kind.label(),
                    audit.status.label(),
                    detail
                );
            }
        }
        if fix {
            println!(
                "提示：--fix 只原子重写非规范工件、只截编年史的损坏尾行；其余损坏工件一律不触碰（宁可不修，也不误改）。"
            );
        }
    }

    if damaged > 0 || non_canonical > 0 || repairable > 0 {
        std::process::exit(1);
    }
}

/// 配置声明的功能谓词（融合冲突判定的默认策略来源）。
fn functional_predicates(config: &know_prolog::SystemConfig) -> Vec<String> {
    config
        .predicate_profiles
        .iter()
        .filter(|profile| profile.functional)
        .map(|profile| profile.name.clone())
        .collect()
}

/// `namespace`：把 RDF 文件导入命名空间源图（默认图不动）。
async fn run_namespace_import(
    manager: &OxigraphManager,
    source: &str,
    label: Option<&str>,
    trust: &str,
    kind: &str,
    file_path: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let trust = TrustLevel::parse(trust).ok_or_else(|| {
        format!(
            "未知信任层级：{}（可用 verified / curated / imported / external / speculative）",
            trust
        )
    })?;
    let kind = NamespaceKind::parse(kind).ok_or_else(|| {
        format!(
            "未知命名空间种类：{}（可用 global / import / session / hypothesis）",
            kind
        )
    })?;
    let format = OxigraphManager::detect_rdf_format(file_path)?;
    let file = std::fs::File::open(file_path)?;
    let parser = oxigraph::io::RdfParser::from_format(format).for_reader(file);

    let mut assertions = Vec::new();
    let mut skipped = 0usize;
    for quad in parser {
        let quad = quad.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
            format!("解析 RDF 失败：{}", error).into()
        })?;
        match OxigraphManager::assertion_from_rdf_quad(&quad, source, file_path) {
            Some(assertion) => assertions.push(assertion),
            None => skipped += 1,
        }
    }

    let namespace = NamespaceRecord::new(source, label.unwrap_or(source), kind, trust);
    let added = manager
        .record_namespace_assertions(&namespace, &assertions)
        .await?;

    if json {
        let report = serde_json::json!({
            "namespace": namespace.id,
            "label": namespace.label,
            "trust": namespace.trust.label_zh(),
            "kind": namespace.kind.label_zh(),
            "graph": namespace.graph_iri(),
            "parsed": assertions.len(),
            "added": added,
            "skipped": skipped,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "命名空间 {}（{}，{}，{}）",
            namespace.id,
            namespace.label,
            namespace.kind.label_zh(),
            namespace.trust.label_zh()
        );
        println!("  源图：{}", namespace.graph_iri());
        println!(
            "  解析 {} 条断言（跳过 {}），新增源图四元组 {}",
            assertions.len(),
            skipped,
            added
        );
        println!("  默认图未改动；用 `fuse --apply` 物化融合视图");
    }
    Ok(())
}

/// `fuse`：预览 / 物化融合视图，并记录显式裁决。
async fn run_fuse(
    manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    apply: bool,
    prefer: &[String],
    json: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut ledger = manager.load_fusion_ledger().await?;

    let mut recorded = Vec::new();
    for spec in prefer {
        let Some((predicate, namespace)) = spec.split_once('=') else {
            return Err(format!("裁决格式应为 谓词=命名空间：{}", spec).into());
        };
        let predicate = predicate.trim();
        let namespace = namespace.trim();
        if predicate.is_empty() || namespace.is_empty() {
            return Err(format!("裁决格式应为 谓词=命名空间：{}", spec).into());
        }
        let namespace_id = know_prolog::knowledge_fusion::sanitize_namespace_id(namespace);
        if ledger.namespace(&namespace_id).is_none() {
            return Err(
                format!("命名空间「{}」不在账本中：先导入来源，再写裁决", namespace).into(),
            );
        }
        let adjudication = Adjudication::new(predicate, &namespace_id, "命令行裁决", "user");
        ledger.record_adjudication(adjudication.clone());
        recorded.push(adjudication);
    }
    if !recorded.is_empty() {
        ledger.sort_and_dedup();
        manager.sync_fusion_ledger(&ledger).await?;
    }

    let policy = FusionPolicy::new().with_functional(functional_predicates(config));
    let (plan, inserted, removed) = if apply {
        let outcome = manager.apply_fusion(&policy).await?;
        (outcome.plan, outcome.inserted, outcome.removed)
    } else {
        (manager.fusion_plan(&policy).await?, 0, 0)
    };
    if apply {
        ledger = manager.load_fusion_ledger().await?;
    }

    if json {
        let report = serde_json::json!({
            "namespaces": ledger.namespaces,
            "assertions": ledger.assertions.len(),
            "adjudications": ledger.adjudications,
            "applied": ledger.applied.len(),
            "inserted": inserted,
            "removed": removed,
            "plan": plan,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", ledger.render_zh());
        println!("{}", plan.render_zh());
        if apply {
            println!("已物化：写入 {} 条，回收 {} 条", inserted, removed);
        } else {
            println!("（预览模式：加 --apply 才会写入默认图）");
        }
    }
    Ok(())
}

/// `forge`：离线从正例/反例归纳 token 正则，打印展开与探针结果。
/// 只读配置与例子，不打开存储、不写台账；`--activate` 仅预览物化条目。
#[allow(clippy::too_many_arguments)]
fn run_pattern_forge(
    config: &SystemConfig,
    name: &str,
    route: &str,
    positives: &[String],
    negatives: &[String],
    priority: i32,
    activate: bool,
    json: bool,
) {
    let Some(primary) = positives.first() else {
        eprintln!("锻造至少需要一个 --positive 例子");
        std::process::exit(2);
    };
    let mut request = ForgeRequest::new(name, primary, route).with_priority(priority);
    for positive in positives.iter().skip(1) {
        request = request.with_positive(positive.clone());
    }
    for negative in negatives {
        request = request.with_negative(negative.clone());
    }
    let mut forge = PatternForge::new();
    let outcome = match forge.forge(&request, &config.nlp) {
        Ok(outcome) => outcome,
        Err(error) => {
            eprintln!("锻造失败：{}", error);
            std::process::exit(1);
        }
    };
    let activation = if activate {
        match forge.activate(&outcome.pattern.id) {
            Ok(activation) => Some(activation),
            Err(error) => {
                eprintln!("激活失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let mut outcome = outcome;
    if let Some(activation) = &activation {
        outcome.pattern.status = activation.pattern.status;
    }
    if json {
        let payload = serde_json::json!({
            "created": outcome.created,
            "pattern": &outcome.pattern,
            "active": activation.is_some(),
            "entries": activation.as_ref().map(|activation| &activation.entries),
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("序列化锻造结果失败：{}", error),
        }
        return;
    }
    println!("{}", outcome.render());
    if let Some(activation) = activation {
        println!(
            "已物化 {} 条无分支模式（预览；未写入存储）：",
            activation.entries.len()
        );
        for entry in &activation.entries {
            println!(
                "  - {}（{}）→ {} · 来源 {}",
                entry.name,
                entry.render(),
                entry.query_type,
                entry.source
            );
        }
    } else {
        println!("（预览：加 --activate 可查看展开后的路由条目）");
    }
}

/// `frontier`：离线查看每条候选模式停在哪个要素、期望/实见什么、覆盖多少前缀。
///
/// 与运行时同序：先 `preprocess`，再按 `[synonyms.mappings]` 归一，最后调用
/// 模式库的同一匹配器；因此报告里的失败点就是线上失败点，不是另写一遍的近似。
fn run_frontier(
    question: Option<&str>,
    corpus: Option<&str>,
    near_miss: f64,
    json: bool,
    config: &know_prolog::SystemConfig,
) {
    if !(0.0..=1.0).contains(&near_miss) || !near_miss.is_finite() {
        eprintln!(
            "--near-miss 必须在 0 到 1 之间（含端点），实得 {}",
            near_miss
        );
        std::process::exit(2);
    }
    let synonyms = config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_default();
    let library = PatternLibrary::from_config(config);

    if let Some(path) = corpus {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("读取语料失败（{}）：{}", path, error);
                return;
            }
        };
        let questions: Vec<&str> = text.lines().collect();
        let report =
            frontier::analyze_corpus(questions, &library, &config.nlp, &synonyms, near_miss);
        if json {
            match serde_json::to_string_pretty(&report) {
                Ok(text) => println!("{}", text),
                Err(error) => eprintln!("序列化决策前沿失败：{}", error),
            }
        } else {
            println!("{}", report.render_report());
        }
        return;
    }

    let question = question.unwrap_or_default();
    let report =
        frontier::analyze_with_threshold(question, &library, &config.nlp, &synonyms, near_miss);
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("序列化决策前沿失败：{}", error),
        }
    } else {
        println!("{}", report.render_report());
    }
}

/// `eval-recall`：构念表面召回 + 基线棘轮。
///
/// 分母来自配置声称可答的路由规则及其表面变体；`seen` 与 `unseen` 分开，
/// 避免用「字面命中」冒充泛化。只读配置，不打开知识存储。
fn run_eval_recall(
    baseline: Option<&str>,
    update_baseline: bool,
    allow_regression: bool,
    near_miss: f64,
    json: bool,
    config: &know_prolog::SystemConfig,
) {
    if !(0.0..=1.0).contains(&near_miss) || !near_miss.is_finite() {
        eprintln!(
            "--near-miss 必须在 0 到 1 之间（含端点），实得 {}",
            near_miss
        );
        std::process::exit(2);
    }
    let report = know_prolog::recall_audit::evaluate_config(config, near_miss);
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("构念召回序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }

    let baseline_path = baseline.unwrap_or("recall_baseline.json");
    if update_baseline {
        let new_baseline = know_prolog::recall_audit::RecallBaseline::from_report(&report);
        match new_baseline.save(baseline_path) {
            Ok(()) => println!("召回基线已写入 {}", baseline_path),
            Err(error) => {
                eprintln!("基线写入失败：{}", error);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match know_prolog::recall_audit::RecallBaseline::load(baseline_path) {
            Ok(old_baseline) => {
                let verdict =
                    know_prolog::recall_audit::compare_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(error) => {
                eprintln!("基线读取失败：{}", error);
                std::process::exit(1);
            }
        }
    }
}

/// `read-trace`：离线观察一个问句被切成了什么、词性来自哪里、规划器看到了什么。
/// 提供 `--corpus` 时改为聚合词表缺口（未知词形 + 标记边界），不自动改词表。
/// 加 `--lexicon` 时额外做静态覆盖审计：把「计划可执行但谓词/实体没有定义」
/// 的沉默缺口显式化；`--definitions` 可先通过词门准入 alias/compose 定义。
/// 全过程只读：不写知识库、不改变分词、链接、规划或回答。
fn run_read_trace(
    question: Option<&str>,
    corpus: Option<&str>,
    lexicon: bool,
    definitions: Option<&str>,
    json: bool,
    config: &know_prolog::SystemConfig,
) {
    let mut inventory = lexicon.then(|| lexical_trace::LexiconInventory::from_config(config));
    let mut admission = None;
    if let (Some(path), Some(inventory)) = (definitions, inventory.as_mut()) {
        match lex_gate::load_file(path, inventory) {
            Ok(report) => {
                report.apply(inventory);
                admission = Some(report);
            }
            Err(error) => {
                eprintln!("读取定义失败：{}", error);
                return;
            }
        }
    }
    if let Some(admission) = &admission {
        // JSON 模式下报告走 stderr，保证 stdout 仍是单个合法 JSON 文档。
        if json {
            eprintln!("{}", admission.render_report());
        } else {
            println!("{}", admission.render_report());
        }
    }

    if let Some(path) = corpus {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("读取语料失败（{}）：{}", path, error);
                return;
            }
        };
        let questions: Vec<&str> = text.lines().collect();
        if let Some(inventory) = &inventory {
            let report =
                lexical_trace::coverage_report_with_inventory(questions, &config.nlp, inventory);
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(error) => eprintln!("序列化词表覆盖报告失败：{}", error),
                }
            } else {
                println!("{}", report.render_report());
            }
            return;
        }
        let report = lexical_trace::lexicon_gap_report(questions, &config.nlp);
        if json {
            match serde_json::to_string_pretty(&report) {
                Ok(text) => println!("{}", text),
                Err(error) => eprintln!("序列化词表缺口报告失败：{}", error),
            }
        } else {
            println!("{}", report.render_report());
        }
        return;
    }
    let question = question.unwrap_or_default();
    let plan = QueryPlanner::new().parse(question);
    let trace = match &inventory {
        Some(inventory) => lexical_trace::trace_question_with_inventory(
            question,
            &config.nlp,
            plan.as_ref(),
            inventory,
        ),
        None => lexical_trace::trace_question(question, &config.nlp, plan.as_ref()),
    };
    if json {
        match serde_json::to_string_pretty(&trace) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("序列化读法轨迹失败：{}", error),
        }
    } else {
        println!("{}", trace.render_report());
    }
}

/// `prompt-lint`：校验提示资产注册表；存在 Error 时以非零码退出。
fn run_prompt_lint(json: bool) {
    let specs = prompt_spec::default_specs();
    let issues = prompt_spec::lint_default();
    let (covered, composed) = prompt_spec::coverage();
    let errors = issues
        .iter()
        .filter(|issue| issue.severity == LintSeverity::Error)
        .count();
    if json {
        let value = serde_json::json!({
            "version": prompt_spec::PROMPT_SPEC_VERSION,
            "count": specs.len(),
            "examples_checked": specs.len(),
            "coverage": {
                "covered": covered,
                "composed": composed,
            },
            "errors": errors,
            "lint": issues,
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("lint 报告序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "提示资产注册表 v{}：{} 条模板，已外化策略 {}/{}",
            prompt_spec::PROMPT_SPEC_VERSION,
            specs.len(),
            covered.len(),
            covered.len() + composed.len()
        );
        for issue in &issues {
            let severity = match issue.severity {
                LintSeverity::Error => "ERROR",
                LintSeverity::Warning => "WARN ",
            };
            println!("{} [{}] {}", severity, issue.spec_id, issue.message);
        }
        if issues.is_empty() {
            println!(
                "lint 通过：{} 条示例逐字渲染一致，无错误与警告",
                specs.len()
            );
        }
        if !composed.is_empty() {
            println!("仍由组合生成器负责的策略：{}", composed.join("、"));
        }
    }
    if errors > 0 {
        std::process::exit(1);
    }
}

/// `scenario`：离线回放查询板剧本；任一步失败以非零码退出（可直接进 CI）。
fn run_query_scenario(file: &str, json: bool) {
    let report = match know_prolog::query_scenario::run_file(file) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("查询剧本运行失败：{}", error);
            std::process::exit(1);
        }
    };
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("剧本报告序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }
    if !report.passed {
        std::process::exit(1);
    }
}

/// `fuzz-dialogue`：确定性质量差分模糊测试（无配置、无存储、无网络）。
fn run_fuzz_dialogue(
    cases: usize,
    seed: u64,
    json: bool,
    baseline: Option<&str>,
    update_baseline: bool,
    allow_regression: bool,
) {
    let report = dialogue_fuzz::run(&FuzzConfig { cases, seed });
    let baseline_path = baseline
        .map(str::to_string)
        .unwrap_or_else(|| "dialogue_fuzz_baseline.json".to_string());
    let mut verdict: Option<FuzzVerdict> = None;

    if update_baseline {
        let new_baseline = FuzzBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => {
                if !json {
                    println!(
                        "known-gap 基线已写入 {}（{} 条已知缺口）。",
                        baseline_path,
                        new_baseline.gaps.len()
                    );
                }
            }
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match FuzzBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                verdict = Some(compare_fuzz_baseline(&old_baseline, &report));
            }
            Err(e) => {
                eprintln!("基线读取失败: {}", e);
                std::process::exit(1);
            }
        }
    }

    if json {
        let value = match &verdict {
            Some(verdict) => serde_json::json!({
                "report": &report,
                "verdict": verdict,
            }),
            None => serde_json::to_value(&report).unwrap_or_default(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&value).unwrap_or_default()
        );
    } else {
        println!("{}", report.render_text());
        if let Some(verdict) = &verdict {
            println!("{}", verdict.render_text());
        }
    }

    let regressed = if update_baseline {
        false
    } else {
        verdict
            .as_ref()
            .map(|verdict| !verdict.passed)
            .unwrap_or(report.failed > 0)
    };
    if regressed && !allow_regression {
        std::process::exit(1);
    }
}

/// `config-lint`：把 config.toml 当源码做契约编译；默认 Error 非零退出，
/// `--strict` 连 Warning 也非零退出，便于 CI 棘轮推进。
/// `release-check`：把语言资产编译成 schema 快照并与发布锁做差分，再跑离线
/// 语义门禁。`--update-lock` 只在全部门禁通过后原子地固化新基线。
fn run_release_check(
    config_file: &str,
    lock_path: &str,
    update_lock: bool,
    language_version: Option<String>,
    skip_evals: bool,
    allow_regression: bool,
    json: bool,
) {
    let source = match std::fs::read_to_string(config_file) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("读取配置文件 {} 失败: {}", config_file, error);
            std::process::exit(1);
        }
    };
    let config: know_prolog::SystemConfig = match toml::from_str(&source) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("解析配置文件 {} 失败: {}", config_file, error);
            std::process::exit(1);
        }
    };
    let registry = PluginRegistry::load_dir(&config.plugins.directory);
    let options = CheckOptions {
        config_file,
        lock_path,
        language_version: language_version.as_deref(),
        update_lock,
        skip_evals,
    };
    let report = match release::check(&config, &registry, &options) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("发布检查失败: {}", error);
            std::process::exit(1);
        }
    };
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("发布报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }
    if !report.passed && !allow_regression {
        std::process::exit(1);
    }
}

fn run_config_lint(file: Option<&str>, json: bool, strict: bool) {
    let path = file.unwrap_or("config.toml");
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("读取配置文件 {} 失败: {}", path, e);
            std::process::exit(1);
        }
    };
    let issues = match config_contract::lint_str(&source) {
        Ok(issues) => issues,
        Err(e) => {
            if json {
                let value = serde_json::json!({
                    "version": config_contract::CONFIG_CONTRACT_VERSION,
                    "file": path,
                    "errors": 1,
                    "warnings": 0,
                    "issues": [{
                        "severity": "error",
                        "code": "toml_parse",
                        "path": path,
                        "message": e,
                    }],
                });
                match serde_json::to_string_pretty(&value) {
                    Ok(text) => println!("{}", text),
                    Err(inner) => eprintln!("契约报告序列化失败: {}", inner),
                }
            } else {
                eprintln!(
                    "配置契约 v{}（CC）：TOML 解析失败",
                    config_contract::CONFIG_CONTRACT_VERSION
                );
                eprintln!("ERROR [toml_parse] {}：{}", path, e);
            }
            std::process::exit(1);
        }
    };
    let errors = config_contract::error_count(&issues);
    let warnings = config_contract::warning_count(&issues);
    if json {
        let value = serde_json::json!({
            "version": config_contract::CONFIG_CONTRACT_VERSION,
            "file": path,
            "errors": errors,
            "warnings": warnings,
            "issues": issues,
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("契约报告序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("配置文件：{}", path);
        println!("{}", config_contract::render_text(&issues));
    }
    if errors > 0 || (strict && warnings > 0) {
        std::process::exit(1);
    }
}

/// `config-fulfillment`：把配置里的蕴涵声明编译成证明义务，用图上的见证
/// 生成最小问句，再走真实回答管线判定「兑现 / 诚实不知道 / 说错 / 未检验」。
/// 只读：不写知识库、不改变交互路径；图里未登记的谓词单独列出。
async fn run_config_fulfillment(
    file: Option<&str>,
    store: Option<&str>,
    json: bool,
    strict: bool,
    verbose: bool,
    max_obligations: usize,
) {
    let path = file.unwrap_or("config.toml");
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("读取配置文件 {} 失败: {}", path, error);
            std::process::exit(1);
        }
    };
    let config: SystemConfig = match toml::from_str(&source) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("解析配置文件 {} 失败: {}", path, error);
            std::process::exit(1);
        }
    };

    let mut registry = know_prolog::predicate_registry::PredicateRegistry::new();
    registry.set_strict_admission(config.reasoning.require_predicate_admission);
    for profile in &config.predicate_profiles {
        registry.declare(profile);
    }

    let store_path = store.unwrap_or(&config.system.oxigraph_store_path);
    let manager = match OxigraphManager::new(store_path) {
        Ok(manager) => manager,
        Err(error) => {
            eprintln!("打开知识存储 {} 失败: {}", store_path, error);
            std::process::exit(1);
        }
    };
    let facts = match manager.load_graph_snapshot().await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识图谱快照失败: {}", error);
            std::process::exit(1);
        }
    };

    let report = fulfillment::evaluate_with_config(
        &registry,
        facts.as_slice(),
        &config.reasoning,
        max_obligations,
    );

    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("兑现报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("配置文件：{}", path);
        println!("知识存储：{}", store_path);
        println!("{}", report.render_text(verbose));
    }

    if !report.is_clean() {
        std::process::exit(1);
    }
    if strict && report.has_gaps() {
        std::process::exit(1);
    }
}

/// `plugins`：清点清单、报告启用模式、跑包内评测；不打开知识存储。
fn run_plugins(settings: &PluginSettings, directory: Option<&str>, json: bool, eval: bool) {
    let directory = directory.unwrap_or(&settings.directory);
    let registry = PluginRegistry::load_dir(directory);
    let runtime = PluginRuntime::with_settings(registry, settings);
    let results = if eval {
        runtime.registry().evaluate()
    } else {
        Vec::new()
    };
    if json {
        let value = serde_json::json!({
            "directory": directory,
            "plugins": runtime.manifests(),
            "modes": runtime.modes(),
            "issues": runtime.issues(),
            "evals": results,
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("插件报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", runtime.render_report());
        if eval {
            println!("{}", runtime.registry().render_eval(&results));
        }
    }
    let failed_eval = results.iter().any(|result| !result.passed);
    if runtime.registry().has_errors() || failed_eval {
        std::process::exit(1);
    }
}

/// 启动时的静默断链提醒：只在契约存在 Error 时打一行 stderr，
/// 说明去哪里看明细；不改变任何运行路径（观察，不接管）。
fn compile_frame_program(config: &know_prolog::SystemConfig) -> Option<Arc<FrameProgram>> {
    if !config.dialogue_frames.enabled {
        return None;
    }
    match FrameProgram::compile(&config.dialogue_frames) {
        Ok(program) => Some(Arc::new(program)),
        Err(errors) => {
            eprintln!("绑定帧配置无法编译（已停用，回退常规管线）：");
            for error in errors {
                eprintln!("  {}", error);
            }
            None
        }
    }
}

fn warn_config_contract(config: &know_prolog::SystemConfig) {
    let issues = config_contract::lint(config);
    let errors = config_contract::error_count(&issues);
    if errors > 0 {
        eprintln!(
            "配置契约警告：检测到 {} 个 Error、{} 个 Warning（运行 `knowprolog config-lint` 查看明细；本次启动不改变任何回答路径）",
            errors,
            config_contract::warning_count(&issues)
        );
    }
}

/// 构建质量护照（`eval-quality` 与交互命令共用同一确定性路径）。
/// 规则账本条目 → 归结子句（潜在冲突检测的输入）。
fn ledger_entry_to_clause(
    entry: &know_prolog::graph_ledger::RuleLedgerEntry,
) -> know_prolog::resolution_engine::Clause {
    use know_prolog::resolution_engine::{Atom, Clause, Term};
    fn atom(atom: &know_prolog::graph_ledger::LedgerAtom) -> Atom {
        Atom::new(
            &atom.predicate,
            atom.terms
                .iter()
                .map(|term| {
                    if term.is_var {
                        Term::var(&term.name)
                    } else {
                        Term::constant(&term.name)
                    }
                })
                .collect(),
        )
    }
    Clause::new(
        atom(&entry.head),
        entry.body.iter().map(atom).collect(),
        entry.confidence,
        &entry.source,
    )
}

/// 组装事实级来源：导入日志回填 ∪ `<store>/provenance.json` ∪ 调用方台账。
fn provenance_seed_for_store(
    store_path: Option<&str>,
    extra: Option<&ProvenanceLedger>,
) -> ProvenanceLedger {
    let mut ledger = ProvenanceLedger::new();
    if let Some(store_path) = store_path {
        let journal_path = format!("{}/import_journal.json", store_path);
        if let JournalLoad::Loaded(journal) = ImportJournal::load(&journal_path) {
            ledger.seed_from_import_journal(&journal);
        }
        let ledger_path = format!("{}/provenance.json", store_path);
        if let ProvenanceLoad::Loaded(loaded) = ProvenanceLedger::load(&ledger_path) {
            ledger.merge(&loaded);
        }
    }
    if let Some(extra) = extra {
        ledger.merge(extra);
    }
    ledger
}

async fn quality_report_for(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    store_path: &str,
) -> Result<know_prolog::quality::QualityReport, Box<dyn std::error::Error + Send + Sync>> {
    quality_report_for_with_provenance(oxigraph_manager, config, None, Some(store_path)).await
}

async fn quality_report_for_with_provenance(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    provenance: Option<&ProvenanceLedger>,
    store_path: Option<&str>,
) -> Result<know_prolog::quality::QualityReport, Box<dyn std::error::Error + Send + Sync>> {
    let facts = oxigraph_manager.load_graph_snapshot().await?;
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    // 事实级来源：导入日志 ∪ 默认台账 ∪ 外部台账。即使为空也显式传入，
    // 让「谓词被声明过」不再冒充「这条事实有出处」。
    let ledger = provenance_seed_for_store(store_path, provenance);
    let mut report = know_prolog::quality::build_report_with_provenance(
        &facts,
        &registry,
        &types,
        &config.entity_types,
        Some(&ledger),
    );
    // 快照折叠多值；质量护照必须再用无损事实表回填已见证的功能槽冲突。
    if let Ok(fact_refs) = oxigraph_manager.load_fact_refs().await {
        know_prolog::quality::apply_witnessed_conflicts(
            &mut report,
            &fact_refs,
            Some(&ledger),
            &registry,
            &types,
        );
    }
    Ok(report)
}

/// `serve` 的知识装载：与交互模式同源，取整图快照（实体属性 + 关系）。
/// 只喂实体事实会让服务会话的全部关系问法（谁是X的师傅 / 两人什么关系）
/// 与关系派生的索引静默落空，而属性问法却照常有答案，极难被发现。
async fn serve_facts(
    manager: &OxigraphManager,
) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
    Ok((*manager.load_graph_snapshot().await?).clone())
}

#[derive(Debug, serde::Serialize)]
struct AskAnswer {
    question: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    answer: Option<String>,
}

fn is_ask_terminator(question: &str) -> bool {
    matches!(question.trim(), "quit" | "exit" | "退出" | "q")
}

/// `ask`：把交互模式的核心符号推理做成一次性、无横幅的脚本接口。
///
/// 只读知识存储，不写对话记忆；事实、谓词画像、实体别名与律典的装配顺序
/// 与 `serve`/交互模式一致，保证同一问题在三种入口得到同一结论。
async fn run_ask(
    config: &SystemConfig,
    oxigraph_manager: &OxigraphManager,
    store_path: &str,
    recovery_mode: RecoveryMode,
    question: Option<String>,
    json: bool,
    strict: bool,
) {
    let facts = match serve_facts(oxigraph_manager).await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识库失败: {}", error);
            std::process::exit(1);
        }
    };
    let questions: Vec<String> = match question {
        Some(question) => vec![question],
        None => {
            let mut buffer = String::new();
            if let Err(error) = std::io::stdin().read_to_string(&mut buffer) {
                eprintln!("读取标准输入失败: {}", error);
                std::process::exit(1);
            }
            buffer
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        }
    };
    if questions.is_empty() {
        eprintln!("没有问题可回答：请提供问句参数，或从标准输入逐行输入。");
        std::process::exit(1);
    }

    let registry_path = format!("{}/predicate_registry.json", store_path);
    let _ = oxigraph_manager.load_predicate_registry(&registry_path);
    oxigraph_manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );
    let _ = oxigraph_manager.load_entity_registry(&format!(
        "{}/entity_registry.json",
        store_path
    ));
    oxigraph_manager.configure_entities(&config.entity_profiles);
    oxigraph_manager.configure_type_system(&config.type_system);

    let synonyms = config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_else(create_default_synonyms);
    let mut reasoner = SymbolicReasoner::with_perf_ledger(
        config.reasoning.clone(),
        synonyms,
        PerfLedger::shared(),
    );
    reasoner.set_nlp_config(config.nlp.clone());
    reasoner.configure_ontology_program(&OntologyProgram::from_config(
        &config.type_system,
        &config.predicate_profiles,
    ));
    reasoner.align_predicate_pairs(&oxigraph_manager.predicate_query_pairs());
    reasoner.align_entity_aliases(&oxigraph_manager.confirmed_entity_aliases());
    if let Some(persisted) = LawBook::load_from_path(&format!("{}/law_book.json", store_path)) {
        reasoner.load_persisted_laws(persisted);
    }
    reasoner.observe_facts(&facts);

    // 与交互/服务同一条对话前通道：查询板、指代、澄清与身份回合都在
    // reasoner.answer 之前；一次 ask 的多行输入共享同一会话记忆。
    let mut dialogue = DialogueManager::new();
    dialogue.set_recovery_mode(recovery_mode);
    let mut answers: Vec<AskAnswer> = Vec::with_capacity(questions.len());
    for question in questions {
        // 管道输入常沿用交互脚本的收尾行：不要把它当成待回答的问题。
        if is_ask_terminator(&question) {
            break;
        }
        let answer = dialogue
            .plan_pure_dialogue(&question, &reasoner)
            .or_else(|| dialogue.plan_reply(&question, &reasoner))
            .map(|reply| reply.text)
            .or_else(|| reasoner.answer(&question));
        answers.push(AskAnswer {
            question,
            status: if answer.is_some() {
                "answered"
            } else {
                "unanswered"
            },
            answer,
        });
    }

    if json {
        match serde_json::to_string_pretty(&answers) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("回答序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        for (index, answer) in answers.iter().enumerate() {
            if index > 0 {
                println!();
            }
            println!("> {}", answer.question);
            match &answer.answer {
                Some(text) => println!("{}", text),
                None => println!("未能回答：当前知识库中没有可用依据。"),
            }
        }
    }
    if strict && answers.iter().any(|answer| answer.answer.is_none()) {
        std::process::exit(1);
    }
}

/// 编译并执行形状校验（`eval-shapes` 与交互命令共用同一确定性路径）。
async fn shape_report_for(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
) -> Result<(ShapeSet, ShapeReport), Box<dyn std::error::Error + Send + Sync>> {
    let facts = oxigraph_manager.load_graph_snapshot().await?;
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    let shapes = ShapeSet::from_schema(&config.entity_types, &registry, &types);
    let report = shapes.validate(&facts, &registry, &types);
    Ok((shapes, report))
}

/// 关系获取的只读上下文：图快照 + 谓词画像 + 类型体系 + 形状集。
async fn acquisition_context(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
) -> Result<
    (
        Arc<Vec<PrologFact>>,
        know_prolog::predicate_registry::PredicateRegistry,
        know_prolog::type_system::TypeSystem,
        ShapeSet,
    ),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let facts = load_graph_snapshot(oxigraph_manager).await?;
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    let shapes = ShapeSet::from_schema(&config.entity_types, &registry, &types);
    Ok((facts, registry, types, shapes))
}

fn persist_acquisition(acquisition: &RelationAcquisition, path: &str) {
    if let Err(error) = acquisition.save_to_path(path) {
        eprintln!("关系获取账本保存失败：{}", error);
    }
}

fn persist_pattern_forge(forge: &PatternForge, path: &str) {
    if let Err(error) = forge.save_to_path(path) {
        eprintln!("锻造台账保存失败：{}", error);
    }
}

/// 模式命中回执里若来源是锻造规则，则给该规则记一次成功使用。
///
/// 只做计数，不改变回答；台账按需落盘，避免每轮都写 JSON。
fn record_forge_hit(forge: &mut PatternForge, note: &str, path: &str) {
    let Some(rest) = note.split("来源 ").nth(1) else {
        return;
    };
    let source = rest
        .split('，')
        .next()
        .unwrap_or(rest)
        .trim()
        .trim_end_matches(['）', ')']);
    if source.starts_with(pattern_forge::FORGE_SOURCE_PREFIX) && forge.record_hit(source) {
        persist_pattern_forge(forge, path);
    }
}

fn persist_shape_dialogue(dialogue: &ShapeDialogue, path: &str) {
    if let Err(error) = dialogue.save_to_path(path) {
        eprintln!("形状对话状态保存失败：{}", error);
    }
}

/// 把质量护照注入对话管理器；仅在知识版本推进时重建（离线派生，不写图谱）。
async fn refresh_dialogue_quality(
    dialogue_manager: &mut DialogueManager,
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    revision: u64,
    store_path: &str,
) {
    if dialogue_manager.quality_revision() == Some(revision) {
        return;
    }
    let Ok(facts) = oxigraph_manager.load_graph_snapshot().await else {
        return;
    };
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    let ledger = provenance_seed_for_store(Some(store_path), None);
    let mut report = build_report_with_provenance(
        &facts,
        &registry,
        &types,
        &config.entity_types,
        Some(&ledger),
    );
    if let Ok(fact_refs) = oxigraph_manager.load_fact_refs().await {
        know_prolog::quality::apply_witnessed_conflicts(
            &mut report,
            &fact_refs,
            Some(&ledger),
            &registry,
            &types,
        );
    }
    let index =
        QualityIndex::from_report_with_provenance(&report, &facts, &registry, Some(&ledger))
            .with_revision(revision);
    dialogue_manager.set_quality_index(Some(index));
}

async fn run_interactive_mode(
    config: Arc<know_prolog::SystemConfig>,
    oxigraph_manager: OxigraphManager,
    perf_ledger: Arc<PerfLedger>,
    recovery_mode: RecoveryMode,
    trace_mode: TraceMode,
    store_path: &str,
) {
    println!("{}", config.messages.startup);
    warn_config_contract(&config);
    let frame_program = compile_frame_program(&config);
    if !recovery_mode.eq(&RecoveryMode::Off) {
        println!(
            "失败恢复模式：{}（失败会生成结构化回执）。",
            recovery_mode.label()
        );
    }
    if !trace_mode.is_off() {
        println!(
            "分层追踪模式：{}（每轮决策写入追踪图）。",
            trace_mode.label()
        );
    }

    // 谓词画像注册表：跨会话加载已确认的准入/别名/互逆，再叠加配置声明
    let registry_path = format!("{}/predicate_registry.json", store_path);
    if oxigraph_manager.load_predicate_registry(&registry_path) {
        println!("已恢复谓词画像注册表。可用「谓词报告」查看。");
    }
    oxigraph_manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );

    // 实体身份注册表：跨会话加载已确认的别名/重定向，再叠加配置声明；
    // 已确认别名同步进链接器（持久别名词面与规范名同权）。
    let entity_registry_path = format!("{}/entity_registry.json", store_path);
    if oxigraph_manager.load_entity_registry(&entity_registry_path) {
        println!("已恢复实体身份注册表。可用「身份台账」查看。");
    }
    oxigraph_manager.configure_entities(&config.entity_profiles);

    // 类型体系：别名/子类/互斥声明 + 模式闸门闭包档（config.toml [type_system]）
    oxigraph_manager.configure_type_system(&config.type_system);

    let knowledge_extractor = KnowledgeExtractor::new(config.clone());
    // 文法式关系抽取器：DCG 语言器官，词库由谓词画像注册表注入
    let grammar_extractor = GrammarExtractor::new();
    let response_formatter = ResponseFormatter::new(config.clone());
    let nlu_engine = NLUEngine::new(&config);
    // 统一模式库：配置查询规则 + NLU 意图（同库展示、不路由）+ 跨会话教学模式。
    let mut pattern_library = PatternLibrary::from_config(&config);
    match oxigraph_manager.load_pattern_ledger().await {
        Ok(entries) if !entries.is_empty() => {
            let merged = pattern_library.merge_taught(&entries);
            if merged > 0 {
                println!(
                    "已恢复 {} 条教学模式（模式落图）。可用「模式路由」查看。",
                    merged
                );
            }
        }
        _ => {}
    }
    // 锻造台账：以会话账本的 active 状态为准，重建模式库里的锻造来源。
    // 模式落图可能比锻造台账旧（例如上次激活后未同步），先清掉再按 active
    // 重新物化，避免已退役规则在重启后“复活”。
    let forge_path = format!("{}/pattern_forge.json", store_path);
    let mut pattern_forge = PatternForge::load_from_path(&forge_path).unwrap_or_default();
    pattern_library.remove_source_prefix(pattern_forge::FORGE_SOURCE_PREFIX);
    for entry in pattern_forge.active_entries() {
        pattern_library.add(entry);
    }
    if !pattern_forge.is_empty() {
        println!(
            "已恢复锻造台账：{} 条（激活 {} 条）。可用「锻造台账」查看。",
            pattern_forge.len(),
            pattern_forge.active_count()
        );
        // 图谱可能还留着上次退役/遗忘前的锻造条目；按台账重建后再同步一次，
        // 让「退役后重启又复活」在数据层也不存在。
        match oxigraph_manager
            .sync_pattern_ledger(pattern_library.entries())
            .await
        {
            Ok(count) => println!("锻造模式落图校准：{} 条已同步。", count),
            Err(error) => eprintln!("锻造模式落图校准失败：{}", error),
        }
    }
    // 模式经验：跨会话恢复「上次怎么理解」的记录（独立于模式定义图）。
    let mut pattern_episodes = pattern_ir::PatternEpisodeLog::default();
    match oxigraph_manager.load_pattern_episodes().await {
        Ok(episodes) if !episodes.is_empty() => {
            let merged = pattern_episodes.merge_persisted(&episodes);
            if merged > 0 {
                println!(
                    "已恢复 {} 条模式经验（经验落图）。可用「模式经验」查看。",
                    merged
                );
            }
        }
        _ => {}
    }
    // 经验检查点：交互中每 8 轮把变化过的经验台账落图一次，
    // 避免进程被中断时丢掉整段会话的确认/否定反馈（退出时仍会兜底落图）。
    let mut synced_episodes_revision = pattern_episodes.revision();
    let mut turns_since_episode_sync = 0usize;

    let mut prolog_engine = PrologEngine::new();
    let local_semantic_adapter = {
        let nlp_config = &config.nlp;
        LocalSemanticAdapter::new(
            nlp_config.enable_local_semantic.unwrap_or(true)
                && nlp_config.disable_ownthink_api.unwrap_or(false),
        )
    };

    // 创建外部知识服务：外链名册为外部知识留册（出处 + 逻辑时刻 + 时限），
    // 跨轮、跨重启记住"从哪里、在第几轮、读到了什么"，避免重复打网。
    let external_ledger_path = config
        .api
        .as_ref()
        .and_then(|api| api.external_ledger_path.clone())
        .filter(|path| !path.trim().is_empty())
        .unwrap_or_else(|| format!("{}/external_ledger.json", store_path));
    let external_ledger_capacity = config
        .api
        .as_ref()
        .map(|api| api.external_ledger_capacity)
        .unwrap_or(64);
    let external_ledger_ttl = config
        .api
        .as_ref()
        .map(|api| api.external_ledger_ttl_ticks)
        .unwrap_or(0);
    let external_knowledge_service = {
        let ledger = ExternalLedger::with_config(
            external_ledger_capacity,
            external_ledger_ttl,
            perf_ledger.clone(),
        );
        let service = ExternalKnowledgeService::with_ledger(ledger);
        let service = if let Some(api_config) = &config.api {
            service.with_endpoint(
                api_config.ownthink_base_url.clone(),
                api_config.ownthink_timeout_seconds,
            )
        } else {
            service
        };
        match service.load_ledger(&external_ledger_path) {
            LedgerLoad::Loaded(restored) if restored > 0 => {
                println!(
                    "已恢复外链名册：{} 条（「外链台账」查看，逻辑时刻 tick={}）。",
                    restored,
                    service.ledger_tick()
                );
            }
            LedgerLoad::Loaded(_) => {}
            LedgerLoad::Damaged(message) => {
                eprintln!("外链名册损坏，已忽略（不影响本地知识）：{}", message);
            }
            LedgerLoad::Missing => {}
        }
        service
    };

    let synonyms = if let Some(synonyms_config) = &config.synonyms {
        synonyms_config.mappings.clone()
    } else {
        create_default_synonyms()
    };

    // 神经符号对比仲裁器：联想通道（软逻辑+关联+图游走）与符号通道（蕴含+归结）
    // 交叉验证，输出带对冲词的结论，并保留会话级校准记忆
    let mut contrast_arbiter = ContrastArbiter::from_reasoning_config(&config.reasoning);
    // 对比追问进行中：记录当前提问对应的待定推测编号，用户答「是/不是」即结算
    let mut contrast_followup: Option<usize> = None;
    // 模式联想待确认：一次未命中解析给出的候选骨架。下一轮用户可直接答
    // 「是/不是」来转正或否决；否定记入负向反馈，避免同一骨架反复打扰。
    let mut pending_case: Option<(String, pattern_ir::CaseSuggestion)> = None;

    // 符号推理器：对话上下文 + 本体对齐 + 链接预测 + 符号嵌入
    let mut symbolic_reasoner = SymbolicReasoner::with_perf_ledger(
        config.reasoning.clone(),
        synonyms.clone(),
        perf_ledger.clone(),
    );
    // 读法观测：注入与交互/导入路径相同的 NLP 配置；失败轮才留轨迹。
    symbolic_reasoner.set_nlp_config(config.nlp.clone());
    // 谓词归一：把画像里的别名/互逆词面同步到查询解析与生成词面。
    // 正向词面走别名（alias → 槽位，可参与生成叙事）；逆向词面只登记
    // 查询换向（如「徒弟」「学生」命中 师傅 槽位时主宾对调），不改词面表。
    sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
    // 单一本体：配置里的类型层级与谓词画像并入教学本体（同一推理器、同一胶囊）
    let config_lessons = symbolic_reasoner.configure_ontology_program(
        &OntologyProgram::from_config(&config.type_system, &config.predicate_profiles),
    );
    if config_lessons > 0 {
        println!(
            "已并入配置本体公理：{} 条（类型层级与谓词画像，随对话记忆持久保留）。",
            config_lessons
        );
    }
    // 律典：显式采纳/退役决定跨重启生效；缺失文件不影响声明律。
    if let Some(persisted) = LawBook::load_from_path(&format!("{}/law_book.json", store_path)) {
        symbolic_reasoner.load_persisted_laws(persisted);
    }
    // 实体身份：已确认的持久别名同步进链接器（别名与规范名同权）
    sync_entity_aliases(&mut symbolic_reasoner, &oxigraph_manager);
    // 语境指称消歧（CRD）：跨轮「说法 ↔ 实体」绑定账本 + 区分性追问；
    // 账本落盘 {store}/reference_bindings.json，重启后仍记得澄清过的指称。
    let reference_binding_path = format!("{}/reference_bindings.json", store_path);
    let mut crd_ledger = match BindingLedger::load_from_path(&reference_binding_path) {
        Some(ledger) if !ledger.is_empty() => {
            println!(
                "已恢复指称绑定账本：{} 条（「绑定账本」查看）。",
                ledger.len()
            );
            ledger
        }
        _ => BindingLedger::new(),
    };
    let crd_resolver = ContextualResolver::new();
    // 进行中的澄清请求：下一轮优先按「选哪个/不是A是B」结算
    let mut crd_pending: Option<ClarifyRequest> = None;
    // 导入日志：跨重启的有界撤销栈
    let journal_path = format!("{}/import_journal.json", store_path);
    let mut import_journal = match ImportJournal::load(&journal_path) {
        JournalLoad::Loaded(journal) => {
            if !journal.is_empty() {
                println!(
                    "已恢复导入日志：{} 笔可撤销（「导入历史」查看）。",
                    journal.len()
                );
            }
            journal
        }
        JournalLoad::Damaged(message) => {
            eprintln!("导入日志损坏，已忽略（不影响知识库）：{}", message);
            ImportJournal::new()
        }
        JournalLoad::Missing => ImportJournal::new(),
    };
    // 批量导入期间把整图刷新推迟到真正需要时（见循环顶部）
    let mut reasoner_stale = false;
    if let Ok(facts) = load_graph_snapshot(&oxigraph_manager).await {
        symbolic_reasoner.observe_facts(&facts);
        oxigraph_manager.observe_entity_facts(&facts);
    }
    // 图谱观测落盘：重启后身份台账与图谱一致（观测是幂等的）
    let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
    if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
        if std::path::Path::new(state_path).exists() {
            match SymbolicReasoner::read_dialogue_state(state_path) {
                Ok((snapshot, receipt)) => {
                    if !receipt.canonical {
                        eprintln!(
                            "对话记忆字节非规范（已宽容读取，可用 `artifacts --fix` 规范化）：{}",
                            state_path
                        );
                    }
                    let contrast_snapshot = snapshot.contrast.clone();
                    match symbolic_reasoner.restore_dialogue_snapshot(&snapshot) {
                        Ok(()) => {
                            println!(
                                "已恢复上次对话记忆：{}",
                                symbolic_reasoner.dialogue_summary()
                            );
                        }
                        Err(error) => eprintln!(
                            "对话记忆拒绝（本体胶囊校验失败，其余状态未应用）：{}",
                            error
                        ),
                    }
                    if let Some(snapshot) = contrast_snapshot {
                        if contrast_arbiter.restore_snapshot(&snapshot) {
                            println!(
                                "已恢复上次对比校准记忆：{}",
                                contrast_arbiter.memory_summary()
                            );
                        }
                    }
                }
                Err(error) => {
                    eprintln!(
                        "对话记忆读取失败（已忽略，不影响知识库）：{} [decode_{}]",
                        error,
                        error.code()
                    );
                }
            }
        }
    }
    // 规则图账本：命名图物化视图恢复会话规则（跨进程）；账本写入不推进知识版本
    let stored_rule_ledger = match oxigraph_manager.load_rule_ledger().await {
        Ok(ledger) => Some(ledger),
        Err(error) => {
            eprintln!("规则图账本读取失败（已忽略，不影响知识库）：{}", error);
            None
        }
    };
    if let Some(ledger) = stored_rule_ledger.as_ref() {
        if !ledger.entries.is_empty() {
            let restored = symbolic_reasoner.restore_rules_from_ledger(&ledger.entries);
            if restored > 0 {
                println!(
                    "已从规则图账本恢复 {} 条会话规则（命名图 kg:graphs/rules）。",
                    restored
                );
            }
        }
    }
    // 仅当本会话确有规则时回写，避免 enable_resolution=false 时清空账本
    let startup_rule_entries = symbolic_reasoner.rule_ledger_entries();
    if !startup_rule_entries.is_empty() {
        let needs_sync = stored_rule_ledger
            .as_ref()
            .map(|ledger| ledger.fingerprint != symbolic_reasoner.rule_fingerprint())
            .unwrap_or(true);
        if needs_sync {
            match oxigraph_manager
                .sync_rule_ledger(&startup_rule_entries)
                .await
            {
                Ok(count) => println!(
                    "规则图账本已同步：{} 条规则 → 命名图 kg:graphs/rules。",
                    count
                ),
                Err(error) => {
                    eprintln!("规则图账本同步失败（已忽略，不影响会话）：{}", error);
                }
            }
        }
    }

    // 本体化对话管理器：把每轮对话实例化为对话本体，用 Horn 规则策略推导回答方式
    let dialogue_enabled = config.reasoning.enable_dialogue_manager;
    let mut dialogue_manager = DialogueManager::new();
    dialogue_manager.set_frame_program(frame_program);
    dialogue_manager.set_recovery_mode(recovery_mode);
    dialogue_manager.set_trace_mode(trace_mode);
    *dialogue_manager.trace_store_mut() = config.trace_graph.build_store();
    if dialogue_enabled {
        if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
            let manager_path = format!("{}.manager.json", state_path);
            if std::path::Path::new(&manager_path).exists() {
                match dialogue_manager.restore_from_path_report(&manager_path) {
                    Ok(receipt) => {
                        if !receipt.canonical {
                            eprintln!(
                                "对话本体记忆字节非规范（已宽容读取，可用 `artifacts --fix` 规范化）：{}",
                                manager_path
                            );
                        }
                        println!("已恢复对话本体记忆：{}", dialogue_manager.summary());
                    }
                    Err(error) => {
                        eprintln!(
                            "对话本体记忆读取失败（已忽略，不影响会话）：{} [decode_{}]",
                            error,
                            error.code()
                        );
                    }
                }
            }
        }
    }

    // 声明式插件：清单只在启动时合并；运行中切换的是改写/配方/模板/风格的
    // 启用模式。用户显式覆盖落盘到 <store>/plugin_state.json，重启后继续生效。
    let plugin_state_path = format!("{}/plugin_state.json", store_path);
    let plugin_registry = PluginRegistry::load_dir(&config.plugins.directory);
    let mut plugin_runtime = PluginRuntime::with_settings(plugin_registry, &config.plugins);
    if let Some(state) = PluginSessionState::load(&plugin_state_path) {
        let restored = plugin_runtime.restore_session_state(state);
        if restored > 0 {
            println!("已恢复插件会话状态：{} 个模式覆盖。", restored);
        }
    }
    if plugin_runtime.installed_count() > 0 {
        println!(
            "插件目录「{}」：已安装 {} 个，启用 {} 个（「插件」查看，「启用插件：<id>」切换）。",
            config.plugins.directory,
            plugin_runtime.installed_count(),
            plugin_runtime.enabled_count()
        );
    }
    for issue in plugin_runtime.issues() {
        if issue.severity == know_prolog::plugin::PluginSeverity::Error {
            eprintln!("插件：{}", issue.render());
        }
    }
    // 本体贡献按启用集合并进既有谓词对齐器与统一本体推理器（单调、幂等）。
    let plugin_sync = symbolic_reasoner.apply_plugin_runtime(&plugin_runtime);
    if plugin_sync.aliases > 0 {
        println!("插件本体已合并：{} 条谓词别名。", plugin_sync.aliases);
    }
    if plugin_sync.lessons > 0 {
        println!("插件本体已并入统一推理器：{} 条公理。", plugin_sync.lessons);
    }
    // 互逆谓词的逆向词面：发问时主宾对调（「执导」问句 ↔「导演」事实）。
    if plugin_sync.inverse_surfaces > 0 {
        println!(
            "插件本体已合并：{} 条逆向查询词面。",
            plugin_sync.inverse_surfaces
        );
    }
    // 指称线索在启动时补入分词/词性词典，与本体别名一样只在启动时合并。
    let plugin_nlp = plugin_runtime.augmented_nlp_config(&config.nlp);
    let plugin_reference_entries = plugin_nlp
        .pos_dict
        .len()
        .saturating_sub(config.nlp.pos_dict.len());
    if plugin_reference_entries > 0 {
        println!(
            "插件指称线索已合并：{} 条词性提示。",
            plugin_reference_entries
        );
    }

    // 问答式关系获取（QRA）：会话账本 + 覆盖层 + 对话诱导模式。
    // 只读图谱与画像，确认后才进入覆盖层；采纳后才进入会话事实集。
    let acquisition_path = format!("{}/acquisition_ledger.json", store_path);
    let mut acquisition =
        RelationAcquisition::load_from_path(&acquisition_path).unwrap_or_default();
    if acquisition.turn_count() > 0 {
        println!(
            "已恢复关系获取账本：{} 条，诱导模式 {} 个。",
            acquisition.claims().count(),
            acquisition.patterns().count()
        );
    }
    let shape_dialogue_path = format!("{}/shape_dialogue.json", store_path);
    let mut shape_dialogue =
        ShapeDialogue::load_from_path(&shape_dialogue_path).unwrap_or_default();
    if shape_dialogue.turn_count() > 0 {
        println!(
            "已恢复形状填充对话：轮次 {}，{}。",
            shape_dialogue.turn_count(),
            if shape_dialogue.has_pending() {
                "有一个待答问题"
            } else {
                "没有待答问题"
            }
        );
    }

    // 对话编年史：追加式哈希链档案，是对话的真相源；快照只是检查点。
    // 每轮输入在任何处理之前先落盘（写前日志），崩溃最多丢进行中的一轮；
    // 坏行只废尾巴不废全档：可信前缀保留、原子重写截断后续写。
    let chronicle_path = config
        .reasoning
        .dialogue_state_path
        .as_deref()
        .map(|path| format!("{}.chronicle.jsonl", path))
        .unwrap_or_else(|| format!("{}/dialogue.chronicle.jsonl", store_path));
    let mut chronicle = match Chronicle::open(&chronicle_path, FsyncPolicy::EachAppend) {
        Ok((mut chronicle, audit)) => {
            if !audit.is_clean() {
                eprintln!("对话编年史{}", audit.verdict());
                match chronicle.repair_tail() {
                    Ok(_) => {
                        println!(
                            "编年史已截断损坏尾行并保留可信前缀：{} 条记录。",
                            chronicle.len()
                        );
                        Some(chronicle)
                    }
                    Err(error) => {
                        eprintln!("编年史尾行重写失败（本轮不落史以免污染链条）：{}", error);
                        None
                    }
                }
            } else {
                if !chronicle.is_empty() {
                    println!(
                        "已恢复对话编年史：{} 条记录，水位 seq {}（「编年史」查看 / 「回放 N」重建）。",
                        chronicle.len(),
                        chronicle.last_seq()
                    );
                }
                Some(chronicle)
            }
        }
        Err(error) => {
            eprintln!("对话编年史打开失败（本轮不落史）：{}", error);
            None
        }
    };
    // KB 修订号：知识库每发生一次批量变更（导入/撤销）自增，随编年史记录，
    // 用于区分「重演」时的知识漂移（挂钟时间不可断言，序号才是语义时戳）。
    let mut kb_revision: u64 = chronicle
        .as_ref()
        .map(|chronicle| chronicle.kb_revision())
        .unwrap_or(0);

    // 分层追踪回放：把编年史里的 `kind="trace"` 逐轮重建为追踪工作记忆
    //（问据栈/指代轨迹/更正覆盖），使重启后仍能承继谓词焦点与改口。
    if !trace_mode.is_off() {
        if let Some(chronicle) = chronicle.as_ref() {
            let mut replayed = 0usize;
            for entry in chronicle.replay_to(chronicle.last_seq()) {
                if entry.kind != "trace" {
                    continue;
                }
                if let Ok(trace) = serde_json::from_value::<know_prolog::trace_graph::TurnTrace>(
                    entry.payload.clone(),
                ) {
                    dialogue_manager.replay_turn_trace(trace);
                    replayed += 1;
                }
            }
            if replayed > 0 {
                println!("已回放分层追踪：{} 轮（「追踪报告」查看）。", replayed);
            }
        }
    }

    println!("{}", config.messages.ready);
    println!("{}", config.messages.query_prompt);
    println!("{}", config.messages.import_prompt);
    println!("{}", config.messages.quit_prompt);
    println!();

    let stdin_reader = std::io::stdin();
    loop {
        turns_since_episode_sync += 1;
        if turns_since_episode_sync >= 8 {
            turns_since_episode_sync = 0;
            let revision = pattern_episodes.revision();
            if revision != synced_episodes_revision {
                match oxigraph_manager
                    .sync_pattern_episodes(pattern_episodes.entries())
                    .await
                {
                    Ok(_) => synced_episodes_revision = revision,
                    Err(error) => {
                        eprintln!("模式经验检查点落图失败（下次重试）：{}", error);
                    }
                }
            }
        }
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        match stdin_reader.lock().read_line(&mut input) {
            Ok(0) => break, // EOF reached, exit the loop
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
        let input = input.trim();

        if matches!(input, "quit" | "退出" | "exit") {
            break;
        }

        if matches!(input, "帮助" | "help" | "?" | "？" | "h") {
            println!("{}", interactive_help());
            println!();
            continue;
        }

        // 配置契约体检：随时把当前 config.toml 当源码再编译一遍
        if matches!(input, "配置契约" | "配置检查" | "配置体检" | "config-lint") {
            let issues = config_contract::lint(&config);
            println!("{}", config_contract::render_text(&issues));
            continue;
        }

        // 声明式插件：启停、模式、诊断与自述先于一切查询处理。
        let enabled_before = plugin_runtime.enabled_count();
        if let Some(outcome) = plugin_runtime.handle_command(input) {
            println!("{}", outcome.text);
            if outcome.state_changed {
                let sync = symbolic_reasoner.apply_plugin_runtime(&plugin_runtime);
                if sync.aliases > 0 || sync.lessons > 0 || sync.inverse_surfaces > 0 {
                    println!(
                        "插件本体已增量合并：{} 条别名、{} 条公理、{} 条逆向词面。",
                        sync.aliases, sync.lessons, sync.inverse_surfaces
                    );
                } else if plugin_runtime.enabled_count() < enabled_before {
                    println!(
                        "提示：停用插件不撤回本会话已合并的本体与别名，重启后按当前启用集重建。"
                    );
                }
                if let Err(error) = plugin_runtime.session_state().save(&plugin_state_path) {
                    eprintln!("插件会话状态保存失败（不影响本次回答）：{}", error);
                }
            }
            println!();
            continue;
        }

        // 插件改写只作用于自然语言查询：命令、导入行与帮助保持逐字原样。
        let raw_input = input;
        let plugin_outcome = if !raw_input.is_empty()
            && matches!(
                detect_input_mode(raw_input, &config.nlp.import_keywords),
                InputMode::Query
            )
            && !is_command_like(raw_input)
        {
            Some(plugin_runtime.rewrite(raw_input))
        } else {
            None
        };
        let input = plugin_outcome
            .as_ref()
            .map(|outcome| outcome.text.as_str())
            .unwrap_or(raw_input);

        // 外链名册的逻辑时刻：每个非空用户轮次推进一次（TTL 的计量单位）
        if !input.is_empty() {
            external_knowledge_service.advance_tick();
        }

        if dialogue_enabled && !raw_input.is_empty() {
            dialogue_manager.observe_input(raw_input);
        }

        // 全息时序记忆：把本轮输入中提及的实体压入置换编码的焦点叠加（多轮"注意力轨迹"）
        if config.reasoning.enable_holographic_memory && !raw_input.is_empty() {
            symbolic_reasoner.note_holo_context(raw_input);
        }

        // 导入行不立即刷新：进入查询/命令前统一观察整图
        if reasoner_stale
            && !matches!(
                detect_input_mode(input, &config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            reasoner_stale = false;
            kb_revision += 1;
            if let Some(chronicle) = chronicle.as_mut() {
                chronicle.set_kb_revision(kb_revision);
            }
        }
        // 对话层的确认/更正必须成为推理层前提：刷新图谱后再同步，避免被重建覆盖。
        if dialogue_enabled {
            sync_recovery_confirmed_facts(&dialogue_manager, &mut symbolic_reasoner);
        }

        // 编年史写前日志：用户输入先落盘再处理；记录当轮开始时的焦点与对话水位，
        // 重启后据此回放/重建上下文（崩溃语义：最多丢进行中的一轮）。
        if !raw_input.is_empty() {
            if let Some(chronicle) = chronicle.as_mut() {
                let mode = match detect_input_mode(raw_input, &config.nlp.import_keywords) {
                    InputMode::KnowledgeImport => "import",
                    InputMode::Query => "query",
                };
                let kind = if is_command_like(raw_input) {
                    "command"
                } else {
                    mode
                };
                let focus = dialogue_manager.focus_entity();
                let _ =
                    chronicle.append_turn(raw_input, kind, focus, dialogue_manager.turn_count());
            }
        }

        // 关系获取：把「答不出」编译成带期望的槽位问题（以问代抽）。
        if let Some(raw) = strip_any_prefix(
            input,
            &[
                "关系获取：",
                "关系获取:",
                "问答获取：",
                "问答获取:",
                "获取关系：",
                "获取关系:",
            ],
        ) {
            let raw = raw.trim();
            if raw.is_empty() {
                println!("用法：关系获取：<实体> 的 <谓词>，例如「关系获取：姚明的女儿是谁」。");
            } else {
                match acquisition_context(&oxigraph_manager, &config).await {
                    Ok((facts, registry, types, shapes)) => {
                        let opening =
                            acquisition.start_command(raw, &facts, &registry, &types, &shapes);
                        println!("{}", opening.render_text());
                        persist_acquisition(&acquisition, &acquisition_path);
                    }
                    Err(error) => eprintln!("加载关系获取上下文失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 形状缺口 → 下一个问题：把节点的必备属性缺口编译成槽位追问。
        if matches!(input, "获取缺口" | "关系缺口" | "追问缺口" | "缺口追问") {
            match acquisition_context(&oxigraph_manager, &config).await {
                Ok((facts, registry, types, shapes)) => {
                    let report = shapes.validate(&facts, &registry, &types);
                    let mut opened = None;
                    for violation in &report.violations {
                        if let RepairAction::AskProperty { entity, property } = &violation.repair {
                            let candidate = acquisition.start_shape_gap(
                                entity, property, &facts, &registry, &types, &shapes,
                            );
                            if matches!(&candidate, GoalOpen::Opened { .. }) {
                                opened = Some(candidate);
                                break;
                            }
                        }
                    }
                    match opened {
                        Some(opening) => {
                            println!("{}", opening.render_text());
                            persist_acquisition(&acquisition, &acquisition_path);
                        }
                        None => {
                            println!("形状校验没有可追问的必备属性缺口（或缺口值已在覆盖层）。")
                        }
                    }
                }
                Err(error) => eprintln!("加载形状缺口失败：{}", error),
            }
            println!();
            continue;
        }

        // 关系获取台账：候选 / 已确认 / 冲突 / 否证 + 覆盖层 + 模式库
        // 关系获取台账：候选 / 已确认 / 冲突 / 否证 + 覆盖层 + 模式库
        if matches!(
            input,
            "获取台账" | "获取报告" | "关系获取报告" | "问答获取报告" | "获取状态"
        ) {
            println!("{}", acquisition.report());
            println!();
            continue;
        }

        // 统一模式库（Pattern IR）：配置查询 + NLU 意图 + 教学模式同库视图
        if matches!(input, "模式路由" | "查询模式库" | "统一模式库") {
            println!("{}", pattern_ir::render_library_report(&pattern_library));
            let findings = pattern_ir::lint(&pattern_library);
            if findings.is_empty() {
                println!("体检：无重复/冲突/遮蔽。");
            } else {
                println!("体检：{} 条待处理", findings.len());
                for finding in findings.iter().take(12) {
                    println!("  - {}", finding.detail);
                }
            }
            println!();
            continue;
        }

        // 模式关系图：泛化/冗余/冲突落成可查询的边
        if matches!(
            input,
            "模式关系" | "查询模式关系" | "模式关系图" | "模式图体检"
        ) {
            println!("{}", pattern_ir::render_relation_report(&pattern_library));
            println!();
            continue;
        }

        // 模式经验：命中/未命中的可检索记忆（独立命名图）
        if matches!(input, "模式经验" | "模式经验报告" | "经验台账") {
            println!("{}", pattern_ir::render_episode_report(&pattern_episodes));
            println!();
            continue;
        }

        // 模式联想：用相似成功经验泛化出候选骨架，只确认、不直接路由
        if let Some(text) = strip_any_prefix(
            input,
            &[
                "模式联想：",
                "模式联想:",
                "模式联想 ",
                "案例联想：",
                "案例联想:",
            ],
        ) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：模式联想：<句子>，例如「模式联想：德国的首都在哪里」。");
            } else {
                let context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);
                match suggest_case_for_text(
                    text,
                    &config,
                    &synonyms,
                    &pattern_episodes,
                    &context,
                ) {
                    Some(suggestion) => {
                        println!("{}", pattern_ir::render_case_suggestion(text, &suggestion));
                        println!("（可直接回答「是」确认，或回答「不是」否定这次联想。）");
                        pending_case = Some((text.to_string(), suggestion));
                    }
                    None => println!(
                        "模式联想：没有足够相似的成功经验；先处理几条同类问句，或用「教模式」显式教学。"
                    ),
                }
            }
            println!();
            continue;
        }

        // 确认联想：把当前输入的候选骨架转正为可路由模式并落图
        if let Some(text) = strip_any_prefix(
            input,
            &[
                "确认联想：",
                "确认联想:",
                "确认联想 ",
                "采纳联想：",
                "采纳联想:",
                "采纳联想 ",
                "案例转正：",
                "案例转正:",
                "案例转正 ",
            ],
        ) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：确认联想：<句子>，例如「确认联想：德国的首都在哪里」。");
            } else {
                let context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);
                match suggest_case_for_text(
                    text,
                    &config,
                    &synonyms,
                    &pattern_episodes,
                    &context,
                ) {
                    Some(suggestion) => match pattern_library.accept_case(&suggestion) {
                        Ok(entry) => {
                            println!(
                                "已确认案例：{}（来源 {}，路由 {}，优先级 {}）。",
                                entry.name,
                                entry.source,
                                entry.query_type,
                                entry.priority
                            );
                            println!("候选模式：{}", entry.render());
                            match oxigraph_manager
                                .sync_pattern_ledger(pattern_library.entries())
                                .await
                            {
                                Ok(count) => {
                                    println!(
                                        "模式落图：{} 条已同步到 {}。",
                                        count,
                                        pattern_ir::PATTERNS_GRAPH
                                    )
                                }
                                Err(error) => println!("确认后落图失败：{}", error),
                            }
                        }
                        Err(error) => println!("确认联想失败：{}", error),
                    },
                    None => println!(
                        "确认联想：没有足够相似的成功经验；请先用「模式联想：<句子>」确认候选，或使用「教模式」显式教学。"
                    ),
                }
            }
            println!();
            continue;
        }

        // 忘经验：按短 id / 完整 id / 原句删除单条经验（撤销误确认或误否定）
        if let Some(needle) = strip_any_prefix(
            input,
            &["忘经验：", "忘经验:", "忘经验 ", "遗忘经验：", "遗忘经验:"],
        ) {
            let needle = needle.trim();
            if needle.is_empty() {
                println!("用法：忘经验：<短 id 或原句>，短 id 见「模式经验」。");
                println!();
                continue;
            }
            let removed = pattern_episodes.forget(needle);
            match oxigraph_manager
                .sync_pattern_episodes(pattern_episodes.entries())
                .await
            {
                Ok(count) => println!("已删除 {} 条模式经验（经验落图：{} 条）。", removed, count),
                Err(error) => println!("删除 {} 条模式经验，但经验落图失败：{}", removed, error),
            }
            println!();
            continue;
        }

        if matches!(input, "清空模式经验" | "遗忘模式经验") {
            pattern_episodes.clear();
            match oxigraph_manager
                .sync_pattern_episodes(pattern_episodes.entries())
                .await
            {
                Ok(count) => println!("已清空模式经验（经验落图：{} 条）。", count),
                Err(error) => println!("清空模式经验失败：{}", error),
            }
            println!();
            continue;
        }

        // 模式落图：把当前模式库与关系边同步到 kg:graphs/patterns 并读回校验
        if matches!(input, "模式落图" | "同步模式图" | "保存模式库") {
            match oxigraph_manager
                .sync_pattern_ledger(pattern_library.entries())
                .await
            {
                Ok(count) => {
                    println!(
                        "模式落图：{} 条已同步到 {}。",
                        count,
                        pattern_ir::PATTERNS_GRAPH
                    );
                    match oxigraph_manager.load_pattern_ledger().await {
                        Ok(loaded) => {
                            println!("读回校验：{} 条条目。", loaded.len());
                        }
                        Err(error) => println!("读回失败：{}", error),
                    }
                    let edges = pattern_ir::relation_edges(pattern_library.entries());
                    println!(
                        "关系边：{} 条（泛化/冗余/冲突各按图查询可复核）。",
                        edges.len()
                    );
                }
                Err(error) => println!("模式落图失败：{}", error),
            }
            match oxigraph_manager
                .sync_pattern_episodes(pattern_episodes.entries())
                .await
            {
                Ok(count) => {
                    println!(
                        "经验落图：{} 条已同步到 {}。",
                        count,
                        pattern_ir::EPISODES_GRAPH
                    );
                    match oxigraph_manager.load_pattern_episodes().await {
                        Ok(loaded) => println!("经验读回校验：{} 条。", loaded.len()),
                        Err(error) => println!("经验读回失败：{}", error),
                    }
                }
                Err(error) => println!("经验落图失败：{}", error),
            }
            println!();
            continue;
        }

        // 模式匹配解释：对一句话给出胜者、命中榜与落选原因
        if let Some(text) = strip_any_prefix(
            input,
            &[
                "模式匹配：",
                "模式匹配:",
                "模式匹配 ",
                "匹配模式：",
                "匹配模式:",
            ],
        ) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：模式匹配：<句子>，例如「模式匹配：法国的首都是哪里」。");
            } else {
                let tagged = preprocess(text, &config.nlp);
                let normalized = normalize_words(tagged, &synonyms);
                let context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);
                let report = pattern_library.rank_with_context(&normalized, &context);
                println!("{}", pattern_ir::render_match_report(text, &report));
                let scan = pattern_library.scan_route_with_context(&normalized, &context);
                println!(
                    "快速路由扫描：完整评估 {} 条，首要素预筛跳过 {} 条（可路由 {} 条，扫描 {} 条）。",
                    scan.evaluated,
                    scan.skipped,
                    pattern_library.route_count(),
                    scan.scanned()
                );
            }
            println!();
            continue;
        }

        // 锻造模式：从正例/反例归纳 token 正则，候选需显式确认才参与路由。
        if matches!(input, "锻造模式" | "模式锻造") {
            println!("用法：锻造模式：<名> | <正例(多个用;分隔)> | <反例(可空，多个用;分隔)> | <路由> | <优先级(可选)> | <守卫(可选)>");
            println!("示例：锻造模式：女儿问法 | 姚明的女儿是谁 | 姚明今年几岁 | daughter_query");
            println!();
            continue;
        }
        if let Some(spec) = strip_any_prefix(
            input,
            &[
                "锻造模式：",
                "锻造模式:",
                "锻造模式 ",
                "模式锻造：",
                "模式锻造:",
            ],
        ) {
            match pattern_forge::parse_forge_spec(spec.trim())
                .and_then(|request| pattern_forge.forge(&request, &config.nlp))
            {
                Ok(outcome) => {
                    println!("{}", outcome.render());
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 确认锻造：候选转正，展开为无分支模式并入统一模式库。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "确认锻造：",
                "确认锻造:",
                "确认锻造 ",
                "采纳锻造：",
                "采纳锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            if key.is_empty() {
                println!("用法：确认锻造 <id>（先用「锻造台账」查看候选 id）。");
            } else {
                match pattern_forge.activate(key) {
                    Ok(activation) => {
                        let backend_known = pattern_library.entries().iter().any(|entry| {
                            entry.route
                                && entry.query_type == activation.pattern.query_type
                                && !entry.source.starts_with(pattern_forge::FORGE_SOURCE_PREFIX)
                        });
                        pattern_library.remove_source(&activation.pattern.source());
                        for entry in &activation.entries {
                            pattern_library.add(entry.clone());
                        }
                        println!(
                            "已激活锻造规则 {}「{}」：{} 条展开参与路由（仅本会话来源，可用「退役锻造」撤销）。",
                            activation.pattern.id,
                            activation.pattern.name,
                            activation.entries.len()
                        );
                        if !backend_known {
                            println!(
                                "  提示：路由「{}」目前没有其它可执行模式。锻造只让这类问法被认出，后端查询仍需在 config.toml 或 PL 规则中提供。",
                                activation.pattern.query_type
                            );
                        }
                        for entry in &activation.entries {
                            println!("  - {}（{}）", entry.name, entry.render());
                        }
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                        persist_pattern_forge(&pattern_forge, &forge_path);
                    }
                    Err(error) => println!("确认锻造失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 否认锻造：记录一次反驳；达到阈值自动退役并撤出模式库。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "否认锻造：",
                "否认锻造:",
                "否认锻造 ",
                "否决锻造：",
                "否决锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match pattern_forge.refute(key) {
                Ok((pattern, retired)) => {
                    if retired {
                        let removed = pattern_library.remove_source(&pattern.source());
                        println!(
                            "锻造规则 {}「{}」累计被反驳 {} 次，已自动退役并移除 {} 条展开。",
                            pattern.id, pattern.name, pattern.refutations, removed
                        );
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                    } else {
                        println!(
                            "已记下对锻造规则 {}「{}」的反驳：{}/{} 次。",
                            pattern.id,
                            pattern.name,
                            pattern.refutations,
                            pattern_forge::MAX_REFUTATIONS
                        );
                    }
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("否认锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 退役锻造：停止路由并保留出生档案（可再遗忘）。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "退役锻造：",
                "退役锻造:",
                "退役锻造 ",
                "停用锻造：",
                "停用锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match pattern_forge.retire(key) {
                Ok(pattern) => {
                    let removed = pattern_library.remove_source(&pattern.source());
                    println!(
                        "已退役锻造规则 {}「{}」；从模式库移除 {} 条展开。",
                        pattern.id, pattern.name, removed
                    );
                    match oxigraph_manager
                        .sync_pattern_ledger(pattern_library.entries())
                        .await
                    {
                        Ok(count) => println!("模式落图：{} 条已同步。", count),
                        Err(error) => eprintln!("模式落图失败：{}", error),
                    }
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("退役锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 遗忘锻造：彻底删除非激活记录（激活规则必须先退役）。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "遗忘锻造：",
                "遗忘锻造:",
                "遗忘锻造 ",
                "删除锻造：",
                "删除锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match pattern_forge.forget(key) {
                Ok(pattern) => {
                    let removed = pattern_library.remove_source(&pattern.source());
                    println!(
                        "已遗忘锻造规则 {}「{}」（移除 {} 条展开）。",
                        pattern.id, pattern.name, removed
                    );
                    match oxigraph_manager
                        .sync_pattern_ledger(pattern_library.entries())
                        .await
                    {
                        Ok(count) => println!("模式落图：{} 条已同步。", count),
                        Err(error) => eprintln!("模式落图失败：{}", error),
                    }
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("遗忘锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 锻造探针：重新跑正例/反例/逆生成句，防止正则在分词变化后悄悄失效。
        if matches!(input, "锻造探针" | "复核锻造" | "锻造探针报告") {
            match pattern_forge.verify_probes(None, &config.nlp) {
                Ok(report) => {
                    println!("{}", report.render(None));
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("锻造探针失败：{}", error),
            }
            println!();
            continue;
        }
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "锻造探针：",
                "锻造探针:",
                "锻造探针 ",
                "复核锻造：",
                "复核锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let key = (!key.is_empty()).then_some(key);
            match pattern_forge.verify_probes(key, &config.nlp) {
                Ok(report) => {
                    println!("{}", report.render(key));
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("锻造探针失败：{}", error),
            }
            println!();
            continue;
        }

        if matches!(input, "锻造台账" | "锻造报告" | "锻造状态" | "模式锻造台账")
        {
            println!("{}", pattern_forge.report());
            println!();
            continue;
        }

        // 教模式：<名> | <要素 要素 …> | <路由> | <优先级>
        if let Some(spec) = strip_any_prefix(
            input,
            &["教模式：", "教模式:", "教模式 ", "教学模式：", "教学模式:"],
        ) {
            let parts: Vec<&str> = spec.split('|').map(str::trim).collect();
            if parts.len() < 3 {
                println!("用法：教模式：<名> | <要素 要素 …> | <路由> | <优先级(可选)> | <守卫(可选)>\n  要素：词性码（nr/ns/n/m/v…）、词面、{{槽名}}、{{槽名:排序}}、_ 任意\n  守卫：焦点 / 焦点:人物 / 上轮:ask_attribute / 待澄清 / 无待澄清，多项用逗号分隔");
            } else {
                let name = parts[0];
                let pattern = parts[1];
                let route = parts[2];
                let priority = parts
                    .get(3)
                    .and_then(|p| p.parse::<i32>().ok())
                    .unwrap_or(10);
                let guard = match parts.get(4).filter(|part| !part.is_empty()) {
                    Some(spec) => match pattern_ir::parse_context_guard(spec) {
                        Ok(guard) => Some(guard),
                        Err(error) => {
                            println!("教学模式失败：{}", error);
                            println!();
                            continue;
                        }
                    },
                    None => None,
                };
                match pattern_library.teach_with_guard(name, pattern, route, priority, guard) {
                    Ok(entry) => {
                        println!(
                            "已教模式「{}」→ {}（优先级 {}，IR {}{}）",
                            entry.name,
                            entry.query_type,
                            entry.priority,
                            entry.render(),
                            entry
                                .guard
                                .as_ref()
                                .map(|guard| format!("，守卫：{}", guard.render()))
                                .unwrap_or_default()
                        );
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                    }
                    Err(error) => println!("教学模式失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 忘模式：<名 或 id 前缀>（只删教学模式来源）
        if let Some(name) = strip_any_prefix(
            input,
            &["忘模式：", "忘模式:", "忘模式 ", "遗忘模式：", "遗忘模式:"],
        ) {
            let name = name.trim();
            if name.is_empty() {
                println!("用法：忘模式：<名 或 id 前缀>");
            } else {
                match pattern_library.forget(name) {
                    Some(entry) => {
                        println!(
                            "已遗忘教学模式「{}」（IR {}）。",
                            entry.name,
                            entry.render()
                        );
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                    }
                    None => println!("没有找到可遗忘的教学模式「{}」（只删教学来源）。", name),
                }
            }
            println!();
            continue;
        }

        if matches!(input, "获取模式" | "关系获取模式" | "诱导模式" | "模式库") {
            println!("{}", acquisition.patterns_report());
            println!();
            continue;
        }

        if matches!(input, "获取覆盖" | "会话覆盖层" | "覆盖层") {
            println!("{}", acquisition.overlay().render_text());
            println!();
            continue;
        }

        if let Some(rest) = strip_any_prefix(input, &["确认获取", "确认关系获取", "确认问答获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let outcome = acquisition.confirm(if id.is_empty() { None } else { Some(id) });
            println!("{}", outcome.message);
            if let Some(follow_up) = outcome.follow_up {
                println!("{}", follow_up);
            }
            persist_acquisition(&acquisition, &acquisition_path);
            println!();
            continue;
        }

        if let Some(rest) = strip_any_prefix(input, &["否认获取", "否认关系获取", "拒绝获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let outcome = acquisition.deny(if id.is_empty() { None } else { Some(id) });
            println!("{}", outcome.message);
            persist_acquisition(&acquisition, &acquisition_path);
            println!();
            continue;
        }

        if let Some(rest) = strip_any_prefix(input, &["撤回获取", "撤销获取", "撤回关系获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let outcome = acquisition.retract(if id.is_empty() { None } else { Some(id) });
            println!("{}", outcome.message);
            persist_acquisition(&acquisition, &acquisition_path);
            println!();
            continue;
        }

        // 采纳：Confirmed/Admitted 的条目进入会话事实集（仍不写知识图谱）。
        if let Some(rest) = strip_any_prefix(input, &["采纳获取", "采纳关系获取", "提交获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match acquisition.commit_proposal(if id.is_empty() { None } else { Some(id) }) {
                Some(proposal) => {
                    let answer = symbolic_reasoner.assert_nl_fact(&proposal.sentence);
                    println!("{}", answer);
                    if acquisition.mark_admitted(&proposal.claim_id) {
                        println!(
                            "已采纳 {}：{}（会话覆盖层仍可回溯；未写入知识图谱）。",
                            proposal.claim_id, proposal.sentence
                        );
                    }
                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                    persist_acquisition(&acquisition, &acquisition_path);
                }
                None => println!(
                    "没有可采纳的条目；只有「已确认」或「已采纳」的关系才能进入会话事实集。"
                ),
            }
            println!();
            continue;
        }

        // 对话诱导模式扫描：只有被两次确认激活的模式才会命中。
        if let Some(text) = strip_any_prefix(input, &["模式扫描：", "模式扫描:", "模式扫描 "])
        {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：模式扫描：<文本>，例如「模式扫描：王强的女儿是赵敏」。");
            } else {
                match acquisition_context(&oxigraph_manager, &config).await {
                    Ok((facts, registry, types, shapes)) => {
                        let outcome = acquisition
                            .absorb_pattern_scan(text, &facts, &registry, &types, &shapes);
                        println!("{}", outcome.message);
                        persist_acquisition(&acquisition, &acquisition_path);
                    }
                    Err(error) => eprintln!("加载模式扫描上下文失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 归纳模式：把已激活的对话诱导模式（DIP）提升为统一模式库条目。
        if matches!(input, "归纳模式" | "诱导入库" | "模式入库" | "归纳规律") {
            let active: Vec<pattern_ir::PatternEntry> = acquisition
                .patterns()
                .filter(|pattern| pattern.active && !pattern.retired)
                .map(|pattern| {
                    let ir = format!("{{主体}} 的 {} 是 {{客体}}", pattern.surface);
                    pattern.to_pattern_entry(&ir)
                })
                .collect();
            if active.is_empty() {
                println!("没有可归纳的激活模式（同一谓词被两次确认后自动激活）。");
            } else {
                let mut added = 0usize;
                for entry in &active {
                    if pattern_library.add(entry.clone()) {
                        added += 1;
                    }
                }
                println!(
                    "归纳入库：{} 条激活诱导模式，新增 {} 条（来源 induced#…）。",
                    active.len(),
                    added
                );
                for entry in &active {
                    println!(
                        "  - {}（{}）→ {}",
                        entry.name,
                        entry.render(),
                        entry.query_type
                    );
                }
                match oxigraph_manager
                    .sync_pattern_ledger(pattern_library.entries())
                    .await
                {
                    Ok(count) => println!("模式落图：{} 条已同步。", count),
                    Err(error) => eprintln!("模式落图失败：{}", error),
                }
            }
            println!();
            continue;
        }

        if let Some(id) = strip_any_prefix(input, &["退役模式：", "退役模式:", "退役模式 "])
        {
            let id = id.trim();
            if acquisition.retire_pattern(id) {
                println!("已退役诱导模式 {}。", id);
                persist_acquisition(&acquisition, &acquisition_path);
            } else {
                println!("没有找到诱导模式 {}。", id);
            }
            println!();
            continue;
        }

        // 形状填充对话：把形状报告编译成议程，问到缺口后回到原问题。
        if let Some(entity) = strip_any_prefix(
            input,
            &["形状画像：", "形状画像:", "对话画像：", "对话画像:"],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("用法：形状画像：<实体>，例如「形状画像：刘德华」。");
            } else {
                match acquisition_context(&oxigraph_manager, &config).await {
                    Ok((facts, registry, types, shapes)) => {
                        let question = format!("{}是谁？", entity);
                        let turn = shape_dialogue.start_profile(
                            entity,
                            &question,
                            &facts,
                            &registry,
                            &types,
                            &shapes,
                            &mut acquisition,
                        );
                        println!("{}", turn.message);
                        persist_acquisition(&acquisition, &acquisition_path);
                        persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                    }
                    Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 会话级冲突裁定：只收敛本会话画像，不写图谱。
        if let Some(body) = strip_any_prefix(
            input,
            &[
                "裁定：",
                "裁定:",
                "冲突裁定：",
                "冲突裁定:",
                "会话裁定：",
                "会话裁定:",
            ],
        ) {
            match know_prolog::shape_dialogue::parse_ruling(body) {
                Some((entity, predicate, value)) => {
                    match acquisition_context(&oxigraph_manager, &config).await {
                        Ok((facts, registry, types, shapes)) => {
                            match shape_dialogue.resolve_conflict(
                                &entity,
                                &predicate,
                                &value,
                                &facts,
                                &registry,
                                &types,
                                &shapes,
                                &mut acquisition,
                            ) {
                                Ok(turn) => println!("{}", turn.message),
                                Err(message) => println!("{}", message),
                            }
                            persist_acquisition(&acquisition, &acquisition_path);
                            persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                        }
                        Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
                    }
                }
                None => println!(
                    "用法：裁定：<实体> 的 <谓词> = <值>，例如「裁定：刘德华 的 职业 = 演员」。"
                ),
            }
            println!();
            continue;
        }

        // 口语化裁定：只在上一条画像留下待裁定冲突且没有待答缺口时接管。
        if !shape_dialogue.has_pending()
            && ShapeDialogue::looks_like_conflict_ruling(input)
            && !is_command_like(input)
        {
            match acquisition_context(&oxigraph_manager, &config).await {
                Ok((facts, registry, types, shapes)) => {
                    if let Some(result) = shape_dialogue.resolve_conflict_text(
                        input,
                        &facts,
                        &registry,
                        &types,
                        &shapes,
                        &mut acquisition,
                    ) {
                        match result {
                            Ok(turn) => println!("{}", turn.message),
                            Err(message) => println!("{}", message),
                        }
                        persist_acquisition(&acquisition, &acquisition_path);
                        persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                        println!();
                        continue;
                    }
                }
                Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
            }
        }

        if matches!(input, "形状对话报告" | "形状对话状态" | "形状对话") {
            println!("{}", shape_dialogue.report_text());
            println!();
            continue;
        }

        if matches!(input, "形状议程" | "形状缺口议程") {
            match acquisition_context(&oxigraph_manager, &config).await {
                Ok((facts, registry, types, shapes)) => {
                    let (ruled_facts, ruled_overlay) = know_prolog::shape_dialogue::apply_rulings(
                        &facts,
                        &acquisition.overlay(),
                        shape_dialogue.rulings(),
                        &registry,
                    );
                    let effective =
                        know_prolog::shape_dialogue::effective_facts(&ruled_facts, &ruled_overlay);
                    let report = shapes.validate(&effective, &registry, &types);
                    let agenda =
                        know_prolog::shape_dialogue::ShapeAgenda::compile(&report, &registry);
                    println!("{}", agenda.render_text());
                }
                Err(error) => eprintln!("加载形状议程失败：{}", error),
            }
            println!();
            continue;
        }

        if matches!(input, "下一个形状问题" | "下一个缺口问题") {
            match acquisition_context(&oxigraph_manager, &config).await {
                Ok((facts, registry, types, shapes)) => {
                    match shape_dialogue.ask_next(
                        &facts,
                        &registry,
                        &types,
                        &shapes,
                        &mut acquisition,
                    ) {
                        Some(turn) => {
                            println!("{}", turn.message);
                            persist_acquisition(&acquisition, &acquisition_path);
                            persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                        }
                        None => println!(
                            "形状议程没有可追问的缺口（可能已完成、已达提问上限，或已有待答问题）。"
                        ),
                    }
                }
                Err(error) => eprintln!("加载形状议程失败：{}", error),
            }
            println!();
            continue;
        }

        if input == "形状对话重置" {
            shape_dialogue.reset(&mut acquisition);
            persist_acquisition(&acquisition, &acquisition_path);
            persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
            println!("形状填充对话状态已重置（关系获取账本与知识图谱保持不变）。");
            println!();
            continue;
        }

        if shape_dialogue.has_pending()
            && !is_command_like(input)
            && RelationAcquisition::looks_like_answer(input)
            && !matches!(
                detect_input_mode(input, &config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            match acquisition_context(&oxigraph_manager, &config).await {
                Ok((facts, registry, types, shapes)) => {
                    if let Some(turn) = shape_dialogue.respond(
                        input,
                        &facts,
                        &registry,
                        &types,
                        &shapes,
                        &mut acquisition,
                    ) {
                        println!("{}", turn.message);
                    }
                    persist_acquisition(&acquisition, &acquisition_path);
                    persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                }
                Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
            }
            println!();
            continue;
        }

        // 待填槽位的短答：先做对话行为分类，再进形状闸门与账本。
        // 命令、导入行与新问题不拦截，保持既有管线不变。
        if acquisition.has_pending_goal()
            && !is_command_like(input)
            && RelationAcquisition::looks_like_answer(input)
            && !matches!(
                detect_input_mode(input, &config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            match acquisition_context(&oxigraph_manager, &config).await {
                Ok((facts, registry, types, shapes)) => {
                    let outcome =
                        acquisition.interpret_answer(input, &facts, &registry, &types, &shapes);
                    println!("{}", outcome.message);
                    if let Some(follow_up) = outcome.follow_up {
                        println!("{}", follow_up);
                    }
                    persist_acquisition(&acquisition, &acquisition_path);
                }
                Err(error) => eprintln!("加载关系获取上下文失败：{}", error),
            }
            println!();
            continue;
        }

        if let Some(term) = input
            .strip_prefix("对齐候选：")
            .or_else(|| input.strip_prefix("对齐候选:"))
        {
            let term = term.trim();
            if term.is_empty() {
                println!("请提供要对齐的术语，例如：对齐候选：老婆");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => {
                        symbolic_reasoner.observe_facts(&facts);
                        println!("{}", symbolic_reasoner.alignment_candidates_report(term));
                    }
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
            }
            println!();
            continue;
        }

        if input == "对齐报告" || input == "本体对齐" {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => {
                    symbolic_reasoner.observe_facts(&facts);
                    println!("{}", symbolic_reasoner.alignment_report());
                }
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!();
            continue;
        }

        if input == "推理规则" || input == "链接预测规则" {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => {
                    symbolic_reasoner.observe_facts(&facts);
                    println!("{}", symbolic_reasoner.rules_report());
                }
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!();
            continue;
        }

        if let Some(entity) = input
            .strip_prefix("相似实体：")
            .or_else(|| input.strip_prefix("相似实体:"))
            .or_else(|| input.strip_prefix("语义相似："))
            .or_else(|| input.strip_prefix("语义相似:"))
        {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!("{}", symbolic_reasoner.similarity_report(entity));
            println!();
            continue;
        }

        // 证据检索：符号化稠密排序 + 精确约束 + 上界剪枝，并给出可审计回执
        if let Some(query) = strip_any_prefix(
            input,
            &["证据检索：", "证据检索:", "检索证据：", "检索证据:"],
        ) {
            let query = query.trim();
            if query.is_empty() {
                println!("请提供要检索的问题，例如：证据检索：张三的师傅是谁");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => {
                        symbolic_reasoner.observe_facts(&facts);
                        println!(
                            "{}",
                            symbolic_reasoner.evidence_report(
                                query,
                                8,
                                &SearchConstraints::default(),
                                SearchBudget::default(),
                            )
                        );
                    }
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
            }
            println!();
            continue;
        }

        if input == "对话记忆" {
            println!("{}", symbolic_reasoner.dialogue_summary());
            println!();
            continue;
        }

        // 话语帧记忆：带洞查询帧 + 挂账（合一式多轮消解，可审计）
        if matches!(input, "帧记忆" | "话语帧" | "挂账") {
            println!("{}", symbolic_reasoner.frame_memory_report());
            println!();
            continue;
        }

        // 对话本体状态：展示对话管理器实例与显式的策略规则程序
        if input == "对话本体" || input == "对话状态" {
            println!("{}", dialogue_manager.summary());
            println!("{}", DialogueManager::policy_source());
            println!();
            continue;
        }

        // 分层追踪：问据栈 / 指代轨迹 / 逐轮决策（只读；On 模式才会累积）
        if matches!(input, "追踪报告" | "追踪状态" | "追踪图" | "trace") {
            println!("{}", dialogue_manager.trace_report());
            println!();
            continue;
        }

        // 指代轨迹：当前会话的指代绑定与版本链
        if matches!(input, "指代轨迹" | "指代报告") {
            println!("{}", dialogue_manager.trace_store().referent_report());
            println!();
            continue;
        }

        // 问据栈：当前开放/已决的问据（QUD）
        if matches!(input, "问据栈" | "问据报告" | "QUD") {
            println!("{}", dialogue_manager.trace_store().qud_stack().report());
            println!();
            continue;
        }

        // 追踪为什么：沿追踪图 inputs 边回溯某编号节点的决策链
        if let Some(id) = strip_any_prefix(input, &["追踪为什么：", "追踪为什么:", "追踪为什么"])
        {
            match id.trim().parse::<u64>() {
                Ok(id) => match dialogue_manager.trace_why(id) {
                    Some(text) => println!("{}", text),
                    None => println!("追踪图里没有编号为 {} 的节点。", id),
                },
                Err(_) => println!("请给出节点编号，例如「追踪为什么 4」。"),
            }
            println!();
            continue;
        }

        if matches!(input, "查询板" | "查询状态" | "工作记忆") {
            println!("{}", dialogue_manager.board_summary());
            if let Some(frame) = dialogue_manager.board().current() {
                println!(
                    "当前查询：{}；{} 行；筛选/排序/翻页/选行都可用自然语言继续编辑。",
                    frame.program,
                    frame.rows.len()
                );
            } else {
                println!("可以试试：「列出所有人的身高」「哪些实体」。");
            }
            println!();
            continue;
        }

        // 查询剧本：把多轮交互式查询写成可执行的回合轨迹契约（自包含事实，不依赖知识库）
        if let Some(path) = strip_any_prefix(
            input,
            &["查询剧本：", "查询剧本:", "运行剧本：", "运行剧本:"],
        ) {
            let path = path.trim();
            if path.is_empty() {
                println!("用法：查询剧本：<剧本文件.toml>（文件自带实体事实与每轮期望）");
            } else {
                match know_prolog::query_scenario::run_file(path) {
                    Ok(report) => println!("{}", report.render_text()),
                    Err(error) => println!("查询剧本运行失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 对话编年史：追加式哈希链档案（真相源；快照只是检查点）
        if matches!(input, "编年史" | "对话档案" | "对话编年史") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    println!(
                        "对话编年史（追加式哈希链写前日志）：{} 条记录，水位 seq {}，链尾 {}，KB 修订 {}",
                        chronicle.len(),
                        chronicle.last_seq(),
                        if chronicle.last_hash().is_empty() {
                            "（空）"
                        } else {
                            chronicle.last_hash()
                        },
                        chronicle.kb_revision()
                    );
                    println!("  文件：{}", chronicle.path().display());
                    let recent: Vec<TimelineTurn> = chronicle
                        .entries()
                        .iter()
                        .rev()
                        .take(8)
                        .rev()
                        .filter_map(TimelineTurn::from_entry)
                        .collect();
                    if !recent.is_empty() {
                        println!("{}", render_timeline_markdown(&recent));
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 编年史自检：三类损坏（截尾/翻字节/版本拔高）分别可辨
        if matches!(input, "编年史校验" | "档案校验" | "校验编年史") {
            match Chronicle::audit_file(&chronicle_path) {
                Ok(audit) => println!("编年史自检：{}", audit.verdict()),
                Err(error) => println!("编年史自检失败：{}", error),
            }
            println!();
            continue;
        }

        // 回放：重建到第 N 条记录（调试器视图；不改变当前会话状态）
        if input == "回放"
            || input.starts_with("回放 ")
            || input.starts_with("回放：")
            || input.starts_with("回放:")
        {
            let spec = input
                .trim_start_matches("回放")
                .trim_start_matches([' ', '：', ':'])
                .trim();
            match (chronicle.as_ref(), spec.parse::<u64>().ok()) {
                (Some(chronicle), until) => {
                    let until = until.unwrap_or(chronicle.last_seq());
                    let entries = chronicle.replay_to(until);
                    let turns: Vec<TimelineTurn> = entries
                        .iter()
                        .filter_map(|entry| TimelineTurn::from_entry(entry))
                        .collect();
                    println!(
                        "回放到 seq {}（记录 {} 条；当前状态不变，仅视图重建）：",
                        until,
                        entries.len()
                    );
                    println!("{}", render_timeline_markdown(&turns));
                    if (1..=12).contains(&turns.len()) {
                        println!();
                        println!("```mermaid");
                        println!("{}", render_timeline_flow(&turns));
                        println!("```");
                    }
                }
                (None, _) => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 编年史检索：跨会话长程记忆（命中哪一轮、命中几次，可审计）
        if let Some(query) =
            strip_any_prefix(input, &["聊过：", "聊过:", "聊过 ", "查档案：", "查档案:"])
        {
            let query = query.trim();
            match (chronicle.as_ref(), query.is_empty()) {
                (Some(chronicle), false) => {
                    let matches = chronicle.search(query);
                    if matches.is_empty() {
                        println!("编年史里没有聊过「{}」。", query);
                    } else {
                        println!(
                            "编年史检索「{}」：命中 {} 条（按序号，可审计）：",
                            query,
                            matches.len()
                        );
                        for item in matches.iter().rev().take(8).rev() {
                            println!(
                                "  第 {} 条（{}，命中 {} 次）{}",
                                item.seq, item.kind, item.hits, item.excerpt
                            );
                        }
                    }
                }
                (Some(_), true) => println!("用法：聊过：<关键词>，例如：聊过：姚明"),
                (None, _) => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 对话时间线：焦点迁移 / 指代补全 / 出处逐轮可见
        if matches!(input, "时间线" | "对话时间线") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let turns: Vec<TimelineTurn> = chronicle
                        .entries()
                        .iter()
                        .filter_map(TimelineTurn::from_entry)
                        .collect();
                    println!("对话时间线（编年史投影）：");
                    println!("{}", render_timeline_markdown(&turns));
                    if (1..=12).contains(&turns.len()) {
                        println!();
                        println!("```mermaid");
                        println!("{}", render_timeline_flow(&turns));
                        println!("```");
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 证明日志：从编年史回放已具体化的证明凭证（稳定 id，可再叙述）
        if matches!(input, "证明日志" | "证明档案" | "证明记录") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let projection = JournalProjection::from_chronicle(chronicle);
                    if projection.proofs.is_empty() {
                        println!(
                            "证明日志为空。先用「证明图：<目标>」生成证明，凭证会写入编年史。"
                        );
                    } else {
                        println!(
                            "证明日志：{} 条凭证（按写入顺序；id 只依赖规范逻辑形式）：",
                            projection.proofs.len()
                        );
                        for proof in projection.proofs.iter().rev().take(10).rev() {
                            let confidence = proof
                                .confidence
                                .map(|value| format!("，置信度 {:.2}", value))
                                .unwrap_or_else(|| "，无置信度".to_string());
                            let truncation = if proof.truncated {
                                format!("，已折叠 {} 步", proof.omitted_nodes)
                            } else {
                                String::new()
                            };
                            println!(
                                "  {}  {}（{} 步{}）{}",
                                proof.id, proof.goal, proof.step_count, confidence, truncation
                            );
                        }
                        println!("用法：证明编号：<id> 回放叙述。");
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 按 id 回放证明叙述（Brief/Normal/Deep 由内容自适应：默认 Deep）
        if let Some(id) = strip_any_prefix(input, &["证明编号：", "证明编号:", "证明编号 "])
        {
            let id = id.trim();
            match chronicle.as_ref() {
                Some(chronicle) if !id.is_empty() => {
                    let projection = JournalProjection::from_chronicle(chronicle);
                    match projection.find_proof(id) {
                        Some(proof) => println!("{}", proof.narrate(ProofDetail::Deep)),
                        None => println!("证明日志中没有 id 为「{}」的凭证。", id),
                    }
                }
                Some(_) => println!("用法：证明编号：<id>，例如：证明编号：proof:0123456789abcdef"),
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 日志推导的一致性：同题异答 / 知识版本变更 / 过期回答 / 未答问题
        if matches!(
            input,
            "一致性" | "回答一致性" | "改口检查" | "改口" | "日志视图"
        ) {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let projection = JournalProjection::from_chronicle(chronicle);
                    println!(
                        "{}",
                        projection.consistency(chronicle.kb_revision()).render()
                    );
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 对话情景理论：把编年史编译成可查询的有限 Horn 理论（CST）
        if matches!(input, "情景" | "对话情景" | "情景理论") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_report());
                    if !theory.changes().is_empty() {
                        println!();
                        println!("{}", theory.render_changes());
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["情景：", "情景:"]) {
            match (chronicle.as_ref(), spec.trim().parse::<u64>()) {
                (Some(chronicle), Ok(seq)) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_situation(seq));
                }
                (Some(_), Err(_)) => println!("用法：情景：<seq>，例如：情景：12"),
                (None, _) => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(query) = strip_any_prefix(input, &["情景历史：", "情景历史:", "情景历史 "])
        {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    if query.trim().is_empty() {
                        println!("{}", theory.render_report());
                    } else {
                        println!("{}", theory.render_history(query.trim()));
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(query) = strip_any_prefix(input, &["情景归因：", "情景归因:", "情景归因 "])
        {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_causes(query.trim()));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["情景反事实：", "情景反事实:", "情景反事实 "])
        {
            let spec = spec.trim();
            let (seq, query) = match spec.split_once(['：', ':']) {
                Some((left, right)) if left.trim().parse::<u64>().is_ok() => {
                    (left.trim().parse::<u64>().ok(), right.trim())
                }
                _ => (None, spec),
            };
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    match seq {
                        Some(seq) => {
                            let report = theory.counterfactual_undo_seq(seq, query);
                            println!("{}", theory.render_counterfactual_report(&report));
                        }
                        None => println!("{}", theory.render_counterfactual(query)),
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(pattern) = strip_any_prefix(input, &["情景查询：", "情景查询:", "情景查询 "])
        {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_query(pattern.trim()));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 历史可信度：沿支持图回答「哪条回答依赖了错值、经哪条边失效」
        if matches!(input, "历史可信度" | "回答可信度" | "信任报告") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let assessment = TrustAssessment::from_chronicle(chronicle);
                    println!("{}", assessment.render(None));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(query) = strip_any_prefix(
            input,
            &[
                "历史可信度：",
                "历史可信度:",
                "历史可信度 ",
                "可信度：",
                "可信度:",
                "信任：",
                "信任:",
            ],
        ) {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let assessment = TrustAssessment::from_chronicle(chronicle);
                    println!("{}", assessment.render(Some(query.trim())));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 证明图：把归结证明树渲染成 Mermaid/ASCII（看得见的推理）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "证明图：",
                "证明图:",
                "证明树：",
                "证明树:",
                "画证明：",
                "画证明:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要证明的目标，例如：证明图：张三的师傅的师傅是王五");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.resolution_proof_view(target) {
                    Some(view) => {
                        println!(
                            "证明图：{}  〔{}；知识库 {} 条事实 / {} 条规则〕",
                            view.phrase, view.goal_display, view.facts, view.clauses
                        );
                        if view.solutions.is_empty() {
                            if view.search_status == SearchStatus::Incomplete {
                                println!(
                                    "未得证：深度上限 {} 内搜索被截断，不能据此断言目标不存在（开世界假设）。",
                                    view.max_depth
                                );
                            } else {
                                println!(
                                    "未得证：搜索已穷尽所有分支，未找到归结反驳（开世界假设：推不出 ≠ 为假）。"
                                );
                            }
                            println!(
                                "搜索状态：{}{}",
                                view.search_status.label(),
                                if view.search_truncated {
                                    "（结果可能不完整）"
                                } else {
                                    ""
                                }
                            );
                            println!(
                                "搜索统计：{} 个目标、{} 次目标展开、{} 次事实检查、{} 次规则尝试",
                                view.stats.goals,
                                view.stats.goal_attempts,
                                view.stats.fact_checks,
                                view.stats.rule_attempts
                            );
                            const SHOWN_GAPS: usize = 5;
                            for entry in view.frontier.iter().take(SHOWN_GAPS) {
                                println!("知识缺口：{}", entry.describe());
                            }
                            let omitted = view.frontier.len().saturating_sub(SHOWN_GAPS)
                                + view.stats.frontier_omitted;
                            if omitted > 0 {
                                println!("……另有 {} 条知识缺口未展开", omitted);
                            }
                        }
                        let certificates =
                            ProofCertificate::from_view(&view, ProofLimits::default());
                        if let Some(chronicle) = chronicle.as_mut() {
                            let _ =
                                know_prolog::proof_journal::record_proofs(chronicle, &certificates);
                        }
                        for (index, solution) in view.solutions.iter().enumerate() {
                            let bindings = if solution.bindings.is_empty() {
                                String::new()
                            } else {
                                format!(
                                    "，绑定 {}",
                                    solution
                                        .bindings
                                        .iter()
                                        .map(|(name, value)| format!("{}={}", name, value))
                                        .collect::<Vec<_>>()
                                        .join("；")
                                )
                            };
                            println!(
                                "解 {}（置信度 {:.2}{}）：",
                                index + 1,
                                solution.confidence,
                                bindings
                            );
                            println!("{}", render_proof_ascii(&solution.proofs));
                            println!("```mermaid");
                            println!("{}", render_proof_mermaid(&solution.proofs));
                            println!("```");
                            if let Some(certificate) = certificates.get(index) {
                                println!(
                                    "证明编号：{}（已写入证明日志；用「证明编号：{}」回放）",
                                    certificate.id, certificate.id
                                );
                            }
                        }
                    }
                    None => println!(
                        "无法把「{}」解析为可归结目标；可用「证明：」查看文本报告。",
                        target
                    ),
                }
            }
            println!();
            continue;
        }

        // 知识子图：焦点实体二跳邻域的确定性分层视图（★ 焦点，==> 本轮出处边）
        if let Some(entity) = strip_any_prefix(
            input,
            &[
                "知识子图：",
                "知识子图:",
                "子图：",
                "子图:",
                "邻域：",
                "邻域:",
            ],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供焦点实体，例如：知识子图：姚明");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => {
                        let edges = prolog_fact_edges(&facts);
                        let highlight = latest_used_edges(&dialogue_manager, &edges);
                        let subgraph = KnowledgeSubgraph::bfs(entity, &edges, 2, &highlight);
                        println!(
                            "知识子图「{}」：{} 个节点，{} 条边（★ 焦点，==> 本轮推理用到）：",
                            entity,
                            subgraph.node_count(),
                            subgraph.edge_count()
                        );
                        println!("```mermaid");
                        println!("{}", subgraph.render_mermaid());
                        println!("```");
                        println!("JSON：{}", subgraph.render_json());
                    }
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
            }
            println!();
            continue;
        }

        // 覆盖报告：悬空引用 / 孤立实体 / 未登记谓词（不依赖 schema）
        if matches!(input, "覆盖报告" | "质量报告" | "覆盖检查") {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => {
                    let mut coverage = CoverageInput::default();
                    for fact in facts.iter() {
                        match fact {
                            PrologFact::Entity {
                                name, properties, ..
                            } => {
                                coverage.entities.push(name.clone());
                                for key in properties.keys() {
                                    coverage.attributes.push((name.clone(), key.clone()));
                                }
                            }
                            PrologFact::Relation { from, rel_type, to } => {
                                coverage
                                    .edges
                                    .push((from.clone(), rel_type.clone(), to.clone()));
                            }
                        }
                    }
                    for profile in &config.predicate_profiles {
                        coverage.registered_predicates.push(profile.name.clone());
                        coverage
                            .registered_predicates
                            .extend(profile.aliases.iter().cloned());
                        if let Some(inverse) = &profile.inverse {
                            coverage.registered_predicates.push(inverse.clone());
                        }
                    }
                    println!("{}", render_coverage_report(&coverage));
                }
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!();
            continue;
        }

        // 虚词算符表：功能词唯一正本 + 投影快照（语言地基审计视图）
        if matches!(input, "虚词表" | "功能词表" | "虚词报告" | "算符表") {
            println!("{}", know_prolog::function_words::render_registry_report());
            println!();
            continue;
        }

        // 神经符号对比：对任意陈述做联想通道与符号通道的交叉验证与仲裁
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "对比：",
                "对比:",
                "神经符号对比：",
                "神经符号对比:",
                "双通道对比：",
                "双通道对比:",
                "仲裁：",
                "仲裁:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要对比的陈述句，例如：对比：李四的国籍是德国");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let case = contrast_arbiter.contrast_statement(target, &symbolic_reasoner);
                println!("{}", contrast_arbiter.render_case(&case));
            }
            println!();
            continue;
        }

        // 对比校准报告：会话级一致率/冲突率与最近案例
        if matches!(
            input,
            "对比报告"
                | "对比统计"
                | "校准报告"
                | "对比校准"
                | "仲裁报告"
                | "仲裁统计"
                | "对比记忆"
                | "校准记忆"
        ) {
            println!("{}", contrast_arbiter.memory_summary());
            println!("{}", contrast_arbiter.calibration_report());
            println!();
            continue;
        }

        // 待定联想推测清单
        if matches!(input, "待定推测" | "对比推测" | "联想推测") {
            println!("{}", contrast_arbiter.pending_report());
            println!();
            continue;
        }

        // 确认联想推测：采信为会话事实（不修改知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "对比确认",
                "对比确认：",
                "对比确认:",
                "对比采纳",
                "对比采纳：",
                "对比采纳:",
            ],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => {
                    println!(
                        "{}",
                        contrast_arbiter
                            .confirm_guess(index, &mut symbolic_reasoner)
                            .unwrap_or_else(|| format!(
                                "没有待定的联想推测 #{}；可用「待定推测」查看列表。",
                                index
                            ))
                    );
                }
                None => println!(
                    "请给出推测编号，例如：对比确认 1。当前待定推测：\n{}",
                    contrast_arbiter.pending_report()
                ),
            }
            println!();
            continue;
        }

        // 否认联想推测：加入屏蔽集，同类推测被抑制
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "对比否认",
                "对比否认：",
                "对比否认:",
                "对比否决",
                "对比否决：",
                "对比否决:",
            ],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => {
                    println!(
                        "{}",
                        contrast_arbiter
                            .refute_guess(index)
                            .unwrap_or_else(|| format!(
                                "没有待定的联想推测 #{}；可用「待定推测」查看列表。",
                                index
                            ))
                    );
                }
                None => println!(
                    "请给出推测编号，例如：对比否认 1。当前待定推测：\n{}",
                    contrast_arbiter.pending_report()
                ),
            }
            println!();
            continue;
        }

        // 对比追问：系统主动就最旧的待定推测提问，把联想直觉逆转为澄清对话
        if matches!(input, "对比追问" | "对比提问" | "对比核实" | "逐条核实") {
            match contrast_arbiter.next_followup() {
                Some(followup) => {
                    println!("{}", followup.question);
                    contrast_followup = Some(followup.guess_id);
                }
                None => println!(
                    "当前没有待定联想推测。可输入「对比：<陈述句>」生成新的双通道对比，或「待定推测」查看清单。"
                ),
            }
            println!();
            continue;
        }

        // 对比追问进行中：把「是 / 不是 / 跳过」直接结算为确认或否认（多轮对话）
        if let Some(guess_id) = contrast_followup {
            let answer = input.trim().trim_end_matches(['。', '！', '!', '~']);
            let affirmative = [
                "是", "是的", "对", "对的", "没错", "属实", "确认", "yes", "y",
            ]
            .contains(&answer);
            let negative = ["不是", "不对", "否", "不属实", "错", "no", "n"].contains(&answer);
            if affirmative || negative {
                let outcome = if affirmative {
                    contrast_arbiter.confirm_guess(guess_id, &mut symbolic_reasoner)
                } else {
                    contrast_arbiter.refute_guess(guess_id)
                };
                println!(
                    "{}",
                    outcome.unwrap_or_else(|| format!(
                        "推测 #{} 已不在待定列表（可能已被处理）。",
                        guess_id
                    ))
                );
                contrast_followup = None;
                println!();
                continue;
            }
            if matches!(answer, "跳过" | "下一个" | "不知道" | "不确定") {
                println!("已跳过推测 #{}；可稍后再输入「对比追问」。", guess_id);
                contrast_followup = None;
                println!();
                continue;
            }
            // 非作答输入：结束本轮追问，按普通输入继续处理
            contrast_followup = None;
        }

        // 帧澄清进行中：短答（序号 / 实体名 / 都不是）→ 确定性绑定 → 重放原问句。
        // 绑定有出处（候选 + 轮次），无源可填的洞不猜；非回答输入零状态变化、放行。
        let mut frame_replay: Option<String> = None;
        match handle_frame_clarification(&mut symbolic_reasoner, input) {
            FrameClarificationStep::Replay(question) => frame_replay = Some(question),
            FrameClarificationStep::Handled => continue,
            FrameClarificationStep::Pass => {}
        }
        let input = frame_replay.as_deref().unwrap_or(input);

        // 对话式澄清进行中：优先把输入当作对当前提问的回答（是 / 不是 / 不知道）
        if symbolic_reasoner.clarification_active()
            && !matches!(input, "结束澄清" | "停止澄清" | "退出澄清" | "quit")
            && !is_command_like(input)
        {
            match symbolic_reasoner.answer_clarification(input) {
                Some(text) => println!("{}", text),
                None => println!("（澄清会话已结束）"),
            }
            println!();
            continue;
        }

        // 开始对话式澄清：把待定假设逆转为系统主动提问
        if matches!(input, "澄清" | "开始澄清" | "澄清提问" | "提问确认") {
            match symbolic_reasoner.start_clarification() {
                Some(text) => println!("{}", text),
                None => println!(
                    "当前没有可澄清的待定假设。可先输入「溯因：<陈述>」或「场景：<问句>」生成候选假设。"
                ),
            }
            println!();
            continue;
        }

        // 结束澄清会话：输出已确认/否定/不确定清单
        if matches!(input, "结束澄清" | "停止澄清" | "退出澄清") {
            match symbolic_reasoner.stop_clarification() {
                Some(text) => println!("{}", text),
                None => println!("当前没有进行中的澄清会话。"),
            }
            println!();
            continue;
        }

        // 生成式联合解释：枚举假设组合并注入验证，输出"哪些假设一起才能解释目标"
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "场景：",
                "场景:",
                "联合解释：",
                "联合解释:",
                "解释场景：",
                "解释场景:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要联合解释的目标，例如：场景：张三的师祖是赵六");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.scenario_report(target));
            }
            println!();
            continue;
        }

        // 形式化语义与语义蕴含：把陈述句解析为逻辑形式并判定蕴含/矛盾/未知
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "判断：",
                "判断:",
                "蕴含：",
                "蕴含:",
                "校验：",
                "校验:",
                "形式化：",
                "形式化:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要判定的陈述句，例如：判断：张三是李四的师傅");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
                println!("{}", symbolic_reasoner.entailment_report(target));
            }
            println!();
            continue;
        }

        // 多轮生成：接着上一段介绍继续讲
        if matches!(
            input,
            "接着说" | "继续说" | "继续介绍" | "继续讲" | "接着讲"
        ) {
            match symbolic_reasoner.continue_generation() {
                Some(text) => println!("{}", text),
                None => println!("还没有可续说的生成内容，请先输入「介绍：实体名」。"),
            }
            println!();
            continue;
        }

        // 多轮生成：换个角度重述
        if matches!(
            input,
            "换个角度说" | "换个说法" | "换个角度" | "换一个角度" | "重新介绍"
        ) {
            match symbolic_reasoner.switch_generation_angle() {
                Some(text) => println!("{}", text),
                None => println!("还没有可重述的生成内容，请先输入「介绍：实体名」。"),
            }
            println!();
            continue;
        }

        // 读法追踪：token/词性来源/字符边界/计划/失败分类（只读，不改回答）
        if let Some(text) = strip_any_prefix(input, &["读法追踪：", "读法追踪:", "读法：", "读法:"])
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let text = text.trim();
            if text.is_empty() {
                println!("请提供要追踪的问句，例如：读法追踪：张三的所有权归谁");
            } else {
                println!("{}", symbolic_reasoner.lexical_trace_report(text));
            }
            println!();
            continue;
        }

        // 读法台账：失败轮的有限枚举计数与词性来源分布（不保存原文）
        if matches!(input, "读法台账" | "读法报告" | "失败分类台账" | "失败分类")
        {
            println!("{}", symbolic_reasoner.read_trace_ledger_report());
            println!();
            continue;
        }

        // 提及账本：每个指称的证据收据（边界 + 类型证据 × 疑问词期待 + 焦点先验）
        if let Some(text) = strip_any_prefix(input, &["提及账本：", "提及账本:", "账本：", "账本:"])
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let text = text.trim();
            if text.is_empty() {
                println!("请提供要审计的文本，例如：提及账本：谁写过《背影》");
            } else {
                println!("{}", symbolic_reasoner.mention_ledger_report(text));
            }
            println!();
            continue;
        }

        // 身份台账：规范身份 ↔ 别名 ↔ 歧义（实体 URI 的只读审计视图）
        if matches!(input, "身份台账" | "身份注册表" | "身份报告" | "实体身份") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", oxigraph_manager.entity_registry_report());
            println!("{}", symbolic_reasoner.identity_report());
            println!();
            continue;
        }

        // 指称绑定账本：跨轮「说法 ↔ 实体」绑定与澄清裁决记录（CRD）
        if matches!(input, "绑定账本" | "指称账本" | "语境账本") {
            println!("{}", crd_ledger.report());
            println!();
            continue;
        }

        // 提及格全局解码：不重叠最大证据路径（精确/别名/模糊/嵌套统一选路）
        if let Some(text) = strip_any_prefix(input, &["提及格：", "提及格:", "格解码：", "格解码:"])
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let text = text.trim();
            if text.is_empty() {
                println!("请提供要解码的文本，例如：提及格：北京大学的校长是谁");
            } else {
                println!("{}", symbolic_reasoner.mention_lattice_report(text));
            }
            println!();
            continue;
        }

        // OOV 准入：提名 → 用户确认 → 写入知识库（显式词汇增长，可回溯）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "准入：",
                "准入:",
                "收录：",
                "收录:",
                "确认录入：",
                "确认录入:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要准入的词面，例如：准入：未名科技公司");
            } else {
                // 铸 URI 前的宪法检查：已存在/重定向/折叠同形/歧义时只引导，不新铸
                match oxigraph_manager.entity_mint_check(target) {
                    MintVerdict::Fresh => {}
                    verdict => {
                        println!("{}", verdict.line(target));
                        if let MintVerdict::Variant { canonical, .. } = verdict {
                            println!(
                                "若「{}」确指「{}」，请用「记住身份：{} = {}」登记别名。",
                                target, canonical, target, canonical
                            );
                        }
                        println!();
                        continue;
                    }
                }
                match symbolic_reasoner.admit_vocabulary(target) {
                    Some((name, etype)) => {
                        let mut properties = std::collections::HashMap::new();
                        properties.insert("name".to_string(), name.clone());
                        let items = vec![KnowledgeItem::Entity {
                            label: etype.clone(),
                            properties,
                        }];
                        match oxigraph_manager.save_knowledge(items).await {
                            Ok(()) => {
                                refresh_symbolic_reasoner(
                                    &mut symbolic_reasoner,
                                    &oxigraph_manager,
                                )
                                .await;
                                let _ =
                                    oxigraph_manager.save_entity_registry(&entity_registry_path);
                                println!(
                                    "已准入：「{}」→ 类型 {}（已写入知识库，可在后续问句中被识别）。",
                                    name, etype
                                );
                            }
                            Err(error) => println!("准入写入知识库失败：{}", error),
                        }
                    }
                    None => println!("词面为空，无法准入。"),
                }
            }
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &["实体链接：", "实体链接:", "识别实体：", "识别实体:"],
        ) {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要链接的文本，例如：实体链接：刘德桦的出生地");
            } else {
                println!("{}", symbolic_reasoner.entity_link_report(entity));
            }
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &[
                "介绍：",
                "介绍:",
                "生成介绍：",
                "生成介绍:",
                "介绍一下",
                "讲讲",
                "说说",
            ],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要介绍的实体名，例如：介绍：姚明");
                println!();
                continue;
            }
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            match symbolic_reasoner.generate_about(entity) {
                Some(text) => println!("{}", text),
                None => println!("知识图谱中没有关于「{}」的可生成事实。", entity),
            }
            println!();
            continue;
        }

        // 符号主义叙述：OWL 本体定义开场 + 层级/继承推理 + DCG 文法实现的多句连贯段落
        if let Some(entity) = strip_any_prefix(
            input,
            &[
                "叙述：",
                "叙述:",
                "讲述：",
                "讲述:",
                "详细叙述：",
                "详细叙述:",
            ],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要叙述的实体名，例如：叙述：姚明");
                println!();
                continue;
            }
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match symbolic_reasoner.narrate(entity) {
                Some(text) => println!("{}", text),
                None => println!("知识图谱中没有关于「{}」的叙述材料。", entity),
            }
            println!();
            continue;
        }

        // DCG 文法复述：对同一条事实给出多种合规句式（纯符号改写）
        if let Some(spec) = strip_any_prefix(input, &["复述：", "复述:", "换个句式：", "换个句式:"])
        {
            let spec = spec.trim();
            match spec.rsplit_once('的') {
                Some((entity, predicate))
                    if !entity.trim().is_empty() && !predicate.trim().is_empty() =>
                {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    match symbolic_reasoner.paraphrase_fact(entity.trim(), predicate.trim(), 4) {
                        Some(text) => println!("{}", text),
                        None => println!(
                            "没有找到「{}」的「{}」相关事实。",
                            entity.trim(),
                            predicate.trim()
                        ),
                    }
                }
                _ => println!("用法：复述：<实体>的<属性>，例如：复述：姚明的身高"),
            }
            println!();
            continue;
        }

        // OWL 本体教学：把中文公理（子类/互斥/定义域/固有属性……）纳入本体推理
        if let Some(axiom) =
            strip_any_prefix(input, &["教本体：", "教本体:", "本体教学：", "本体教学:"])
        {
            let axiom = axiom.trim();
            if axiom.is_empty() {
                println!("请提供本体公理，例如：教本体：人是一种动物");
                println!();
                continue;
            }
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match symbolic_reasoner.teach_ontology(axiom) {
                Ok(summary) => println!("已学习本体公理：{}", summary),
                Err(e) => println!("本体公理解析失败：{}", e),
            }
            println!();
            continue;
        }

        if let Some(prefix) = strip_any_prefix(
            input,
            &["续写：", "续写:", "接龙：", "接龙:", "补全：", "补全:"],
        ) {
            let prefix = prefix.trim();
            if prefix.is_empty() {
                println!("请提供要续写的前缀，例如：续写：姚明的");
                println!();
                continue;
            }
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            match symbolic_reasoner.predict_continuation(prefix) {
                Some(text) => println!("{}", text),
                None => println!("无法为「{}」生成续写（实体未知或不含「的」后缀）。", prefix),
            }
            println!();
            continue;
        }

        // 归结证明：对陈述句做 SLD 反驳并输出证明树（纯符号）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "证明：",
                "证明:",
                "归结：",
                "归结:",
                "证明目标：",
                "证明目标:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要证明的陈述，例如：证明：张三的师祖是王五");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.resolution_proof_report(target));
                if let Some(view) = symbolic_reasoner.resolution_proof_view(target) {
                    let certificates = ProofCertificate::from_view(&view, ProofLimits::default());
                    if let Some(chronicle) = chronicle.as_mut() {
                        let _ = know_prolog::proof_journal::record_proofs(chronicle, &certificates);
                    }
                    if let Some(certificate) = certificates.first() {
                        println!(
                            "证明编号：{}（已写入证明日志；用「证明编号：{}」回放）",
                            certificate.id, certificate.id
                        );
                    }
                }
            }
            println!();
            continue;
        }

        // 推理链：对开放目标求全部归结解，逐解展示证明链
        if let Some(target) =
            strip_any_prefix(input, &["推理链：", "推理链:", "寻找证明：", "寻找证明:"])
        {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供推理目标，例如：推理链：张三的师傅的师傅是谁");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.resolution_query_report(target));
                if let Some(view) = symbolic_reasoner.resolution_proof_view(target) {
                    let certificates = ProofCertificate::from_view(&view, ProofLimits::default());
                    if let Some(chronicle) = chronicle.as_mut() {
                        let _ = know_prolog::proof_journal::record_proofs(chronicle, &certificates);
                    }
                    if let Some(certificate) = certificates.first() {
                        println!(
                            "证明编号：{}（已写入证明日志；用「证明编号：{}」回放）",
                            certificate.id, certificate.id
                        );
                    }
                }
            }
            println!();
            continue;
        }

        // 关联规则报告：支持度/置信度/提升度/杠杆率（可带过滤词）
        if input == "关联规则" || input == "关联规则报告" {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.association_report(None));
            println!();
            continue;
        }
        if let Some(filter) = strip_any_prefix(input, &["关联规则：", "关联规则:"]) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.association_report(Some(filter)));
            println!();
            continue;
        }

        // 联想：由关联规则推荐实体可能还具备的缺失特征
        if let Some(entity) =
            strip_any_prefix(input, &["联想：", "联想:", "关联联想：", "关联联想:"])
        {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要联想的实体，例如：联想：姚明");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.association_suggestions(entity) {
                    Some(text) => println!("{}", text),
                    None => {
                        println!("没有可用的关联联想（实体未见过，或没有满足阈值的关联规则）。")
                    }
                }
            }
            println!();
            continue;
        }

        // 补全回合：打开一次可确认、可归纳、可批量闭包的学习回合。
        // 解析与 HTTP 服务共用同一入口，避免两套命令文法漂移。
        if input.contains("补全") || input.starts_with("学习回合") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            if let Some(text) = symbolic_reasoner.handle_completion_command(input) {
                println!("{}", text);
                println!();
                continue;
            }
        }

        // 知识容器层次清单：把回答所依赖的四层前提摊开，供逐层审计。
        if matches!(
            input,
            "知识层次" | "层次清单" | "容器清单" | "会话镜像" | "回答层次"
        ) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.knowledge_layers_report());
            println!();
            continue;
        }

        // 结构联想：RWR 图嵌入（随机游走重启）检索结构相关实体与解释路径
        if let Some(entity) =
            strip_any_prefix(input, &["结构联想：", "结构联想:", "图游走：", "图游走:"])
        {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要联想的实体，例如：结构联想：张三");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.structural_related_report(entity));
            }
            println!();
            continue;
        }

        // 话题延伸：以对话焦点实体栈为多源种子的 RWR 图游走联想
        if matches!(input, "话题延伸" | "顺着话题" | "话题联想") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.topic_extension_report());
            println!();
            continue;
        }

        // 全息符号记忆（VSA/HDC）：状态报告与召回自检（纯符号、零训练）
        if matches!(input, "全息状态" | "全息记忆" | "全息自检" | "全息报告") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.holographic().report());
            println!();
            continue;
        }

        // 全息检索：软召回（绑定/解缠/净噪）→ 硬核查（三元组登记）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "全息检索：",
                "全息检索:",
                "全息查询：",
                "全息查询:",
                "全息补全：",
                "全息补全:",
            ],
        ) {
            match parse_holo_pair(target) {
                Some((subject, predicate)) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    let hits = symbolic_reasoner
                        .holographic()
                        .recall_object(&subject, &predicate);
                    println!(
                        "{}",
                        render_holo_hits(
                            &format!("全息召回「{} --{}--> ?」", subject, predicate),
                            &hits
                        )
                    );
                }
                None => println!(
                    "请给出「主体 谓词」，例如：全息检索：张三 师傅 或 全息检索：张三的师傅"
                ),
            }
            println!();
            continue;
        }

        // 全息反查：(谓词, 客体) → ?主体
        if let Some(target) = strip_any_prefix(input, &["全息反查：", "全息反查:"]) {
            match parse_holo_pair(target) {
                Some((predicate, object)) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    let hits = symbolic_reasoner
                        .holographic()
                        .recall_subject(&predicate, &object);
                    println!(
                        "{}",
                        render_holo_hits(
                            &format!("全息反查「? --{}--> {}」", predicate, object),
                            &hits
                        )
                    );
                }
                None => println!("请给出「谓词 客体」，例如：全息反查：师傅 李四"),
            }
            println!();
            continue;
        }

        // 全息类比：A:B :: C:? —— 解缠关系谓词（硬核查通过）再迁移绑定
        if let Some(target) = strip_any_prefix(input, &["全息类比：", "全息类比:"]) {
            match parse_holo_terms(target, 3) {
                Some(terms) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    let hits = symbolic_reasoner
                        .holographic()
                        .analogy(&terms[0], &terms[1], &terms[2]);
                    println!(
                        "{}",
                        render_holo_analogy(&terms[0], &terms[1], &terms[2], &hits)
                    );
                }
                None => println!(
                    "请给出三个词，例如：全息类比：张三 李四 王五（意为 张三:李四 :: 王五:?）"
                ),
            }
            println!();
            continue;
        }

        // 加权软推理：MLN-lite 加权规则 + 软真值传播（输出概率与模糊限定词）
        if matches!(input, "软推理" | "加权推理") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.soft_facts_report(8));
            println!();
            continue;
        }
        if let Some(target) =
            strip_any_prefix(input, &["软推理：", "软推理:", "加权推理：", "加权推理:"])
        {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供软推理对象，例如：软推理：张三 或 软推理：张三的师傅");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.soft_inference_report(target));
            }
            println!();
            continue;
        }

        // 因果反事实/干预：do(X=真/假) 后对比"事实世界 vs 干预世界"的下游概率差
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "因果反事实：",
                "因果反事实:",
                "反事实干预：",
                "反事实干预:",
                "干预：",
                "干预:",
                "do：",
                "do:",
            ],
        ) {
            let (target, value) = parse_intervention_value(target.trim());
            if target.is_empty() {
                println!("请提供要干预的对象，例如：干预：张三的师傅 或 干预：张三的师傅=真");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.counterfactual_report(target, value));
            }
            println!();
            continue;
        }

        // 因果推理：因果图 + 直接原因 + 事实根因链 + 混杂因子（do-演算视角）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "因果链：",
                "因果链:",
                "因果关系：",
                "因果关系:",
                "因果：",
                "因果:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要因果解释的对象，例如：因果：李四的师祖 或 因果：张三");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.causal_report(target));
            }
            println!();
            continue;
        }

        // 符号梯度归因：可微推理的纯符号版（解析求导 + 求证优先级）
        if let Some(target) = strip_any_prefix(
            input,
            &["梯度归因：", "梯度归因:", "可微推理：", "可微推理:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要归因的对象，例如：梯度归因：李四的师祖 或 梯度归因：张三");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.gradient_report(target));
            }
            println!();
            continue;
        }

        // 因果分词：知识/因果图引导的 token 格 k-best DP + token 级因果归因（纯符号 attention）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "因果分词：",
                "因果分词:",
                "分词：",
                "分词:",
                "token归因：",
                "token归因:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要分词的文本，例如：因果分词：张三的师父是谁");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.causal_tokenize_report(target));
            }
            println!();
            continue;
        }

        // 概率可能世界精确推理：ProbLog 风格 BDD 加权模型计数（概率事实/规则/注释析取）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "概率推理：",
                "概率推理:",
                "可能世界：",
                "可能世界:",
                "ProbLog：",
                "ProbLog:",
                "probLog：",
                "probLog:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要精确查询的对象，例如：概率推理：李四的师祖 或 概率推理：李四");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.problog_report(target));
            }
            println!();
            continue;
        }

        // 默认逻辑推理：可反驳常识（通常…除非…）的接地良基扩展 + 例外阻断 + 概率视图对照
        if let Some(target) = strip_any_prefix(
            input,
            &["默认推理：", "默认推理:", "默认逻辑：", "默认逻辑:"],
        ) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.default_logic_report(target.trim()));
            println!();
            continue;
        }

        // 中文概率知识教学：自然语言句子 → ProbLog 概率知识（会话级，不写知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "教学：",
                "教学:",
                "教我：",
                "教我:",
                "概率知识：",
                "概率知识:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!(
                    "请给出要教的中文知识，例如：教学：如果张三的师傅是李四，那么张三的师祖是王五"
                );
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.teach_report(target));
                sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            }
            println!();
            continue;
        }

        // 规则预演：解析但不入会话规则库，先看重复/不安全变量/功能冲突与派生影响
        if let Some(target) = strip_any_prefix(
            input,
            &["规则预演：", "规则预演:", "教规则预演：", "教规则预演:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出中文规则，例如：规则预演：如果X的师傅是Y，那么X的老师是Y");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.belief_rule_preview(target));
            }
            println!();
            continue;
        }

        // 中文规则学习：教一条 Horn 规则（含占位符自动变量化），直接进入后向链
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "规则：",
                "规则:",
                "教规则：",
                "教规则:",
                "学习规则：",
                "学习规则:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出中文规则，例如：规则：如果X的师傅是Y，那么X的老师是Y");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let report = symbolic_reasoner.learn_rule_report(target);
                let learned = report.contains("已学习中文规则");
                println!("{}", report);
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ =
                        conversation_theory::append_rule(chronicle, target, learned, kb_revision);
                }
                sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            }
            println!();
            continue;
        }

        // 信念一致性：提交前闸门（功能谓词多值 / 定义域 / 值域 / 规则形状）
        if matches!(input, "信念检查" | "信念一致性" | "推理一致性") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.belief_check_report());
            println!();
            continue;
        }

        // 会话修订回执：查看最近写入及其可撤销操作
        if matches!(input, "修订回执" | "修订记录" | "会话修订" | "写入回执") {
            println!("{}", symbolic_reasoner.session_revision_report());
            println!();
            continue;
        }

        // 撤销最新一次会话修订：按回执逆操作恢复事实/规则/教学原句
        if matches!(input, "撤销修订" | "撤销上次修订" | "撤回修订" | "撤销写入")
        {
            let report = symbolic_reasoner.undo_last_session_revision();
            println!("{}", report);
            sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            log_chronicle_reply(&mut chronicle, &report, &[], &[]);
            println!();
            continue;
        }

        // 会话改口：显式更正一个功能槽，记录 before/after，不写知识图谱
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "改口：",
                "改口:",
                "更正记忆：",
                "更正记忆:",
                "纠正记忆：",
                "纠正记忆:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出改口内容，例如：改口：姚明的身高是229cm");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let report = symbolic_reasoner.correct_nl_fact(target);
                println!("{}", report);
                log_chronicle_reply(&mut chronicle, &report, &[], &[]);
            }
            println!();
            continue;
        }

        // 会话槽账本：当前赢家 / 压制值 / 仅证据值
        if matches!(input, "槽账本" | "槽位账本" | "改口账本") {
            println!("{}", symbolic_reasoner.slot_ledger_report());
            println!();
            continue;
        }

        // 单个槽的改口史：改口史：姚明 身高 / 改口史：姚明的身高
        if let Some(target) =
            strip_any_prefix(input, &["改口史：", "改口史:", "槽历史：", "槽历史:"])
        {
            let target = target.trim();
            let pair = target
                .split_once('|')
                .or_else(|| target.split_once('的'))
                .map(|(subject, predicate)| (subject.trim(), predicate.trim()))
                .filter(|(subject, predicate)| !subject.is_empty() && !predicate.is_empty());
            match pair {
                Some((subject, predicate)) => {
                    println!(
                        "{}",
                        symbolic_reasoner.slot_history_report(subject, predicate)
                    );
                }
                None => {
                    println!("请给出主语和谓词，例如：改口史：姚明 身高（或 改口史：姚明的身高）")
                }
            }
            println!();
            continue;
        }

        // 修订预演：更正落库前先展示受影响事实、派生结论与阻止级冲突；只计算不写图
        if let Some(target) = strip_any_prefix(
            input,
            &["修订预演：", "修订预演:", "提交预演：", "提交预演:"],
        ) {
            let target = target.trim();
            match parse_correction_spec(target) {
                None => println!(
                    "请给出实体、属性、旧值与新值，例如：修订预演：姚明的身高从226改为229（或 修订预演：姚明|height|226|229）。"
                ),
                Some((entity, predicate, old_value, new_value)) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    println!(
                        "{}",
                        symbolic_reasoner.belief_revision_preview(
                            &entity,
                            &predicate,
                            &old_value,
                            &new_value
                        )
                    );
                }
            }
            println!();
            continue;
        }

        // 更正知识：删除旧值、写入新值，并把「旧值错了」写成可追踪的事件
        if let Some(target) = strip_any_prefix(
            input,
            &["更正：", "更正:", "纠正：", "纠正:", "修正：", "修正:"],
        ) {
            let target = target.trim();
            match parse_correction_spec(target) {
                None => println!(
                    "请给出实体、属性、旧值与新值，例如：更正：姚明的身高从226改为229（或 更正：姚明|height|226|229）。"
                ),
                Some((entity, predicate, old_value, new_value)) => {
                    match oxigraph_manager
                        .correct_object_value(&entity, &predicate, &old_value, &new_value)
                        .await
                    {
                        Ok((removed, inserted)) if removed > 0 => {
                            let correction = conversation_theory::TheoryCorrection {
                                entity: entity.clone(),
                                predicate: predicate.clone(),
                                from: old_value.clone(),
                                to: new_value.clone(),
                            };
                            let answer = format!(
                                "已更正：{}.{} 由「{}」改为「{}」（移除 {} 条旧三元组，写入 {} 条新值）。",
                                entity,
                                predicate,
                                old_value,
                                new_value,
                                removed,
                                if inserted { 1 } else { 0 }
                            );
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_correction(
                                    chronicle,
                                    target,
                                    &[correction],
                                    kb_revision + 1,
                                );
                            }
                            log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                            // 推理器只增不减：用更正后的图快照重建，使回答基于更新后的图谱
                            rebuild_symbolic_reasoner(
                                &mut symbolic_reasoner,
                                &oxigraph_manager,
                            )
                            .await;
                            reasoner_stale = true;
                            let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                            println!("{}", answer);
                        }
                        Ok(_) => println!(
                            "没有找到「{}」的 {} = 「{}」：更正是「旧值错了」，不是无条件覆盖；未改动知识库。",
                            entity, predicate, old_value
                        ),
                        Err(error) => eprintln!("更正失败：{}", error),
                    }
                }
            }
            println!();
            continue;
        }

        // 记住会话事实：断言为会话级事实（不写知识图谱，可随对话记忆持久化）
        if let Some(target) = strip_any_prefix(
            input,
            &["记住：", "记住:", "记录：", "记录:", "断言：", "断言:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出要记住的事实，例如：记住：张三是李四的师傅");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let answer = symbolic_reasoner.assert_nl_fact(target);
                println!("{}", answer);
                let effects = conversation_theory::effects_from_text(
                    target,
                    conversation_theory::EffectOp::Add,
                );
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_assert(
                        chronicle,
                        target,
                        &effects,
                        kb_revision,
                    );
                }
                log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
            }
            println!();
            continue;
        }

        // 忘记会话事实：从会话事实集撤回（不触碰知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &["忘记：", "忘记:", "遗忘：", "遗忘:", "撤回：", "撤回:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出要忘记的事实，例如：忘记：张三是李四的师傅");
            } else {
                let answer = symbolic_reasoner.retract_nl_fact(target);
                println!("{}", answer);
                let effects = conversation_theory::effects_from_text(
                    target,
                    conversation_theory::EffectOp::Remove,
                );
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_forget(
                        chronicle,
                        target,
                        &effects,
                        kb_revision,
                    );
                }
                log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
            }
            println!();
            continue;
        }

        // 忘记会话规则：子句库与教学原句一致撤回，并重同步规则图账本
        if let Some(target) = strip_any_prefix(
            input,
            &["忘记规则：", "忘记规则:", "撤回规则：", "撤回规则:"],
        ) {
            let answer = symbolic_reasoner.forget_rule_report(target);
            println!("{}", answer);
            log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
            sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            println!();
            continue;
        }

        // 会话规则库：列出学过的 Horn 规则与会话事实
        if matches!(input, "会话规则" | "规则库" | "后向链规则" | "规则列表") {
            println!("{}", symbolic_reasoner.session_rules_report());
            println!();
            continue;
        }

        // 谓词画像：注册表报告 + 知识库观测 + 候选发现（功能/对称/互逆）
        if matches!(input, "谓词报告" | "谓词画像" | "谓词注册表" | "谓词列表") {
            if let Err(e) = oxigraph_manager.refresh_predicate_registry().await {
                eprintln!("刷新谓词画像失败: {}", e);
            }
            println!("{}", oxigraph_manager.predicate_registry_report());
            let _ = oxigraph_manager.save_predicate_registry(&registry_path);
            println!();
            continue;
        }

        // 律典：声明律 + 挖掘规则的统一台账，附确定性审计（矛盾/缺口分开）
        if matches!(input, "律典" | "定律台账" | "语义定律" | "law-report") {
            if let Err(e) = oxigraph_manager.refresh_predicate_registry().await {
                eprintln!("刷新谓词画像失败: {}", e);
            }
            sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                symbolic_reasoner.load_persisted_laws(persisted);
            }
            println!("{}", symbolic_reasoner.law_report());
            println!();
            continue;
        }

        // 质量护照：确定性可信向量（低分只用于排序/措辞/追问，不写回、不删事实）
        if matches!(input, "质量护照" | "质量总览" | "护照总览") {
            match quality_report_for(&oxigraph_manager, &config, &store_path).await {
                Ok(report) => println!("{}", report.render_text()),
                Err(e) => eprintln!("构建质量护照失败: {}", e),
            }
            println!();
            continue;
        }

        // 形状校验：必备属性 / 基数 / 数据类型 / 定义域值域（只读派生）
        if matches!(
            input,
            "形状报告" | "形状校验" | "SHACL报告" | "shacl报告" | "shapes"
        ) {
            match shape_report_for(&oxigraph_manager, &config).await {
                Ok((_, report)) => println!("{}", report.render_text()),
                Err(e) => eprintln!("构建形状报告失败: {}", e),
            }
            println!();
            continue;
        }

        // 单实体质量护照：类型来源 / 四维分数 / 可行动缺口
        if let Some(entity) = strip_any_prefix(input, &["质量护照：", "质量护照:"]) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请给出实体名，例如：质量护照：张三");
            } else {
                match quality_report_for(&oxigraph_manager, &config, &store_path).await {
                    Ok(report) => match report.passport_for(entity) {
                        Some(passport) => println!("{}", render_entity_passport(passport)),
                        None => println!(
                            "知识库中没有实体「{}」（护照只覆盖图谱中存在的实体）。",
                            entity
                        ),
                    },
                    Err(e) => eprintln!("构建质量护照失败: {}", e),
                }
            }
            println!();
            continue;
        }

        // 类型体系：别名/子类/互斥声明与闭包档状态（config.toml [type_system]）
        if matches!(input, "类型体系" | "类型报告" | "类型格") {
            let report = oxigraph_manager.type_system_report();
            if report.trim().is_empty() {
                println!("未声明类型体系（config.toml 的 [type_system] 段可选）。");
            } else {
                println!("{}", report);
            }
            println!();
            continue;
        }

        // 登记谓词别名：别名 = 规范谓词（查询/生成/导入共用同一词面）
        if let Some(spec) = strip_any_prefix(
            input,
            &[
                "登记别名：",
                "登记别名:",
                "谓词别名：",
                "谓词别名:",
                "同义谓词：",
                "同义谓词:",
            ],
        ) {
            match parse_predicate_pair(spec) {
                Some((alias, canonical)) => {
                    if oxigraph_manager.register_predicate_alias(&alias, &canonical) {
                        sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        println!(
                            "已登记谓词别名：{} → {}（问「{}」与「{}」将命中同一事实）",
                            alias, canonical, alias, canonical
                        );
                    } else {
                        println!("未发生变化（别名已存在或与规范名相同）。");
                    }
                }
                None => {
                    println!("请用「登记别名：别名 = 规范谓词」的格式，例如：登记别名：老师 = 师傅")
                }
            }
            println!();
            continue;
        }

        // 登记互逆谓词：左 = 右（p(a,b) 与 q(b,a) 等价，导入自动统一方向）
        if let Some(spec) = strip_any_prefix(
            input,
            &[
                "登记逆谓词：",
                "登记逆谓词:",
                "逆谓词：",
                "逆谓词:",
                "互逆谓词：",
                "互逆谓词:",
            ],
        ) {
            match parse_predicate_pair(spec) {
                Some((left, right)) => {
                    if oxigraph_manager.register_predicate_inverse(&left, &right) {
                        sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        println!(
                            "已登记互逆谓词：{} ↔ {}（导入时统一为规范方向，问句自动换算）",
                            left, right
                        );
                    } else {
                        println!("未发生变化（互逆关系已存在或两侧相同）。");
                    }
                }
                None => {
                    println!("请用「登记逆谓词：左 = 右」的格式，例如：登记逆谓词：师傅 = 徒弟")
                }
            }
            println!();
            continue;
        }

        // 标记功能谓词：同主语只允许唯一取值，冲突不覆盖、只标记
        if let Some(predicate) = strip_any_prefix(input, &["功能谓词：", "功能谓词:"]) {
            let predicate = predicate.trim();
            if predicate.is_empty() {
                println!("请给出谓词名，例如：功能谓词：年龄");
            } else if oxigraph_manager.mark_predicate_functional(predicate, true) {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!(
                    "已标记功能谓词「{}」：同主语的新值将与原值冲突并等待裁决。",
                    predicate
                );
            } else {
                println!("「{}」已是功能谓词。", predicate);
            }
            println!();
            continue;
        }

        if let Some(predicate) = strip_any_prefix(input, &["取消功能谓词：", "取消功能谓词:"])
        {
            let predicate = predicate.trim();
            if !predicate.is_empty() && oxigraph_manager.mark_predicate_functional(predicate, false)
            {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!("已取消「{}」的功能谓词标记。", predicate);
            } else {
                println!("「{}」不是功能谓词。", predicate);
            }
            println!();
            continue;
        }

        // 实体身份：确认/拒绝/记住/忘记/合并/铸造检查（谓词注册表的实体侧对偶）
        if let Some(spec) = strip_any_prefix(input, &["确认身份：", "确认身份:"]) {
            match parse_predicate_pair(spec) {
                Some((surface, canonical)) => {
                    if oxigraph_manager.confirm_entity_alias(&surface, &canonical) {
                        symbolic_reasoner.register_entity_alias(&surface, &canonical);
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                        println!(
                            "已确认身份：「{}」≡「{}」（持久别名，重启后仍生效）。",
                            surface, canonical
                        );
                    } else {
                        println!("确认未有改动（词面为空、词面即规范名或已确认）。");
                    }
                }
                None => println!("请给出身份对，例如：确认身份：苹果 = 苹果公司"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["记住身份：", "记住身份:"]) {
            match parse_predicate_pair(spec) {
                Some((surface, canonical)) => {
                    if oxigraph_manager.confirm_entity_alias(&surface, &canonical) {
                        symbolic_reasoner.register_entity_alias(&surface, &canonical);
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                        println!(
                            "已记住身份：「{}」≡「{}」（提名即时确认并落盘）。",
                            surface, canonical
                        );
                    } else {
                        println!("记住未有改动（词面为空、词面即规范名或已确认）。");
                    }
                }
                None => println!("请给出身份对，例如：记住身份：苹果 = 苹果公司"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["拒绝身份：", "拒绝身份:"]) {
            match parse_predicate_pair(spec) {
                Some((surface, canonical)) => {
                    if oxigraph_manager.reject_entity_alias(&surface, &canonical) {
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                        println!(
                            "已拒绝身份：「{}」↛「{}」（立碑：不予自动提名、不参与解析；重启后生效）。",
                            surface, canonical
                        );
                    } else {
                        println!("拒绝未有改动（未找到该提名，或该词面即规范名）。");
                    }
                }
                None => println!("请给出身份对，例如：拒绝身份：苹果 = 苹果果实"),
            }
            println!();
            continue;
        }

        if let Some(surface) = strip_any_prefix(input, &["忘记别名：", "忘记别名:"]) {
            let surface = surface.trim();
            if surface.is_empty() {
                println!("请给出要忘记的词面，例如：忘记别名：苹果");
            } else {
                let affected = oxigraph_manager.forget_entity_alias(surface);
                if affected.is_empty() {
                    println!("没有找到已确认别名「{}」。", surface);
                } else {
                    let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                    println!(
                        "已忘记别名「{}」（影响 {} 个身份；墓碑已立，重启后链接器彻底忘记）。",
                        surface,
                        affected.len()
                    );
                }
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["合并身份：", "合并身份:"]) {
            match parse_predicate_pair(spec) {
                Some((from, into)) => {
                    match oxigraph_manager.merge_entity_identities(&from, &into) {
                        Ok(record) => {
                            symbolic_reasoner.register_entity_alias(&from, &into);
                            let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                            println!(
                            "已合并身份：「{}」→「{}」（迁移 {} 个别名；旧名永久重定向，URI 不回收）。",
                            record.from,
                            record.into,
                            record.moved_aliases
                        );
                        }
                        Err(e) => println!("合并失败：{}", e),
                    }
                }
                None => println!("请给出合并对，例如：合并身份：苹果公司 = 苹果"),
            }
            println!();
            continue;
        }

        if let Some(name) = strip_any_prefix(input, &["铸造检查：", "铸造检查:"]) {
            let name = name.trim();
            if name.is_empty() {
                println!("请给出要检查的名称，例如：铸造检查：苹果公司");
            } else {
                let verdict = oxigraph_manager.entity_mint_check(name);
                println!("{}", verdict.line(name));
            }
            println!();
            continue;
        }

        // 确认/拒绝待定谓词：准入对话的裁决入口
        if let Some(predicate) = strip_any_prefix(input, &["确认谓词：", "确认谓词:"]) {
            let predicate = predicate.trim();
            if !predicate.is_empty() && oxigraph_manager.confirm_predicate(predicate) {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!(
                    "已确认谓词「{}」：画像转为已声明，可用于稳定推理。",
                    predicate
                );
            } else {
                println!("「{}」不是待确认谓词（可能已确认或未登记）。", predicate);
            }
            println!();
            continue;
        }

        if let Some(predicate) = strip_any_prefix(input, &["拒绝谓词：", "拒绝谓词:"]) {
            let predicate = predicate.trim();
            if predicate.is_empty() {
                println!("请给出谓词名，例如：拒绝谓词：师从");
            } else {
                match oxigraph_manager.reject_predicate(predicate).await {
                    Ok(removed) if removed > 0 => {
                        reasoner_stale = true;
                        sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        println!(
                            "已拒绝谓词「{}」并移除 {} 条相关三元组。",
                            predicate, removed
                        );
                    }
                    Ok(_) => println!(
                        "没有可拒绝的待定谓词「{}」（仅自动准入且未确认的谓词可拒绝）。",
                        predicate
                    ),
                    Err(e) => eprintln!("拒绝谓词失败: {}", e),
                }
            }
            println!();
            continue;
        }

        // 撤销最近一次文本导入：逐条移除本次写入的三元组（重启后仍可撤销）
        if matches!(input, "撤销导入" | "撤销上次导入" | "撤回导入") {
            match import_journal.pop() {
                Some(receipt) => match oxigraph_manager.undo_import(&receipt).await {
                    Ok(removed) => {
                        if let Some(chronicle) = chronicle.as_mut() {
                            let _ = conversation_theory::append_retract(
                                chronicle,
                                &receipt,
                                "撤销导入",
                                kb_revision + 1,
                            );
                        }
                        reasoner_stale = true;
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        let _ = import_journal.save(&journal_path);
                        println!(
                            "已撤销上次导入：移除 {} 条三元组（日志余 {} 笔）。",
                            removed,
                            import_journal.len()
                        );
                    }
                    Err(e) => {
                        import_journal.push(receipt);
                        eprintln!("撤销导入失败: {}", e);
                    }
                },
                None => println!("没有可撤销的导入记录。"),
            }
            println!();
            continue;
        }

        // 上次导入回执：新增/合并/重复/冲突/待确认谓词
        if matches!(input, "导入回执" | "上次导入" | "导入记录") {
            match import_journal.last() {
                Some(receipt) => println!("{}", receipt.summary()),
                None => println!("还没有可查询的文本导入记录。"),
            }
            println!();
            continue;
        }

        // 导入历史：跨会话撤销栈（新 → 旧）
        if matches!(input, "导入历史" | "导入日志" | "撤销历史") {
            println!("{}", import_journal.summary());
            println!();
            continue;
        }

        // 外链名册：外部知识的出处、取回时刻与时限（跨轮、跨重启）
        if matches!(input, "外链台账" | "外链名册" | "外链报告" | "外链统计") {
            println!("{}", external_knowledge_service.ledger_report());
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &["外链查看：", "外链查看:", "外链详情：", "外链详情:"],
        ) {
            println!(
                "{}",
                external_knowledge_service.ledger_entry_report(entity.trim())
            );
            println!();
            continue;
        }

        // 显式刷新：删除缓存页，下次提问重新取回（外链数据不在本地写权限内，
        // 用"逻辑时刻 + 显式刷新"代替时间戳轮询）
        if let Some(entity) = strip_any_prefix(
            input,
            &["外链刷新：", "外链刷新:", "外链重取：", "外链重取:"],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请给出实体名，例如：外链刷新：法国");
            } else if external_knowledge_service.refresh_entity(entity) {
                let _ = external_knowledge_service.save_ledger(&external_ledger_path);
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_external(
                        chronicle,
                        entity,
                        "标记刷新",
                        "外链名册",
                        kb_revision,
                    );
                }
                println!("已标记「{}」待刷新：下次提问将重新取回外部知识。", entity);
            } else {
                println!("外链名册中没有「{}」的条目；下次提问将直接取回。", entity);
            }
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &["外链删除：", "外链删除:", "外链遗忘：", "外链遗忘:"],
        ) {
            let entity = entity.trim();
            if external_knowledge_service.forget_entity(entity) {
                let _ = external_knowledge_service.save_ledger(&external_ledger_path);
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_external(
                        chronicle,
                        entity,
                        "删除缓存",
                        "外链名册",
                        kb_revision,
                    );
                }
                println!("已从外链名册删除「{}」。", entity);
            } else {
                println!("外链名册中没有「{}」的条目。", entity);
            }
            println!();
            continue;
        }

        // 确认同义链接：把推测记为别名知识（纯符号的 sameAs 学习，可显式撤销）
        if let Some(body) = strip_any_prefix(
            input,
            &["同义链接：", "同义链接:", "外链别名：", "外链别名:"],
        ) {
            match parse_alias_link(body) {
                Some((alias, target)) => {
                    if external_knowledge_service.confirm_alias(alias, target) {
                        let _ = external_knowledge_service.save_ledger(&external_ledger_path);
                        if let Some(chronicle) = chronicle.as_mut() {
                            let _ = conversation_theory::append_external(
                                chronicle,
                                alias,
                                &format!("同义链接 = {}", target),
                                "外链名册",
                                kb_revision,
                            );
                        }
                        println!("已确认同义链接：「{}」= 「{}」。", alias, target);
                    } else {
                        println!("同义链接不成立：别名与规范实体不能相同或为空。");
                    }
                }
                None => println!(
                    "请使用「同义链接：别名 = 规范实体」格式，例如：同义链接：姚明 = ownthink:姚明"
                ),
            }
            println!();
            continue;
        }

        // 性能报告：存储/推理共享台账（缓存命中、图扫描、版本推进等）
        if matches!(input, "性能报告" | "性能台账" | "运行统计") {
            println!(
                "{}",
                perf_ledger.report_with_title("性能台账（本会话累计）")
            );
            println!();
            continue;
        }

        // 派生台账：知识版本、重建/依赖打戳与推理器内部计数
        if matches!(input, "派生台账" | "版本台账" | "增量报告") {
            println!("{}", symbolic_reasoner.derivation_report());
            println!();
            continue;
        }

        // 推理发现：正向饱和推出多跳新事实（含前提与规则来源）
        if input == "推理发现" || input == "推导发现" {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.derived_facts_report(10));
            println!();
            continue;
        }

        // 物化推理：前向饱和的派生事实写入命名图（含来源/前提/物化戳，可审计）
        if matches!(input, "物化推理" | "推理物化" | "物化派生") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let entries = symbolic_reasoner.derived_ledger_entries(100);
            let revision = oxigraph_manager.revision();
            let fingerprint = symbolic_reasoner.rule_fingerprint();
            match oxigraph_manager
                .materialize_derived_ledger(&entries, revision, &fingerprint)
                .await
            {
                Ok(0) => println!(
                    "没有可物化的派生事实（先教学规则，或启用 enable_resolution）。命名图 kg:graphs/derived 已置空。"
                ),
                Ok(count) => {
                    println!(
                        "已物化 {} 条派生事实 → 命名图 kg:graphs/derived（知识版本 {}，规则指纹 {}）。",
                        count, revision, fingerprint
                    );
                    for (index, entry) in entries.iter().take(5).enumerate() {
                        println!(
                            "  {}. {} ← {}（轮次 {}，置信度 {:.2}，前提 {} 条）",
                            index + 1,
                            entry.atom.display(),
                            entry.rule_source,
                            entry.depth,
                            entry.confidence,
                            entry.premises.len()
                        );
                    }
                    if entries.len() > 5 {
                        println!("  …共 {} 条，用「图账本」查看完整出处。", entries.len());
                    }
                }
                Err(error) => eprintln!("派生账本物化失败：{}", error),
            }
            println!();
            continue;
        }

        // 图账本：命名图物化视图审计（规则账本 + 派生账本 + 过期判定）
        if matches!(input, "图账本" | "规则账本" | "规则图账本" | "派生图账本") {
            println!(
                "{}",
                graph_ledger_report(&symbolic_reasoner, &oxigraph_manager).await
            );
            println!();
            continue;
        }

        // 知识融合：来源命名空间视图（只读，不写默认图）
        if matches!(input, "融合" | "融合视图" | "知识融合" | "来源台账") {
            match oxigraph_manager.load_fusion_ledger().await {
                Ok(ledger) => println!("{}", ledger.render_zh()),
                Err(e) => println!("读取融合账本失败：{}", e),
            }
            match oxigraph_manager
                .fusion_plan(&FusionPolicy::new().with_functional(functional_predicates(&config)))
                .await
            {
                Ok(plan) => println!("{}", plan.render_zh()),
                Err(e) => println!("计算融合视图失败：{}", e),
            }
            println!();
            continue;
        }

        // 知识融合物化：把当前视图的赢家写入默认图（可重复执行，幂等）
        if matches!(input, "融合应用" | "物化融合" | "应用融合") {
            match oxigraph_manager
                .apply_fusion(&FusionPolicy::new().with_functional(functional_predicates(&config)))
                .await
            {
                Ok(outcome) => {
                    println!("{}", outcome.plan.render_zh());
                    println!(
                        "已物化：写入 {} 条，回收 {} 条",
                        outcome.inserted, outcome.removed
                    );
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                }
                Err(e) => println!("融合物化失败：{}", e),
            }
            println!();
            continue;
        }

        // 显式裁决：裁决：谓词=来源（只影响声明的槽位，可审计）
        if let Some(body) = strip_any_prefix(input, &["裁决：", "裁决:"]) {
            match parse_alias_link(body) {
                Some((predicate, namespace)) => {
                    let namespace_id =
                        know_prolog::knowledge_fusion::sanitize_namespace_id(namespace);
                    match oxigraph_manager.load_fusion_ledger().await {
                        Ok(mut ledger) if ledger.namespace(&namespace_id).is_some() => {
                            ledger.record_adjudication(Adjudication::new(
                                predicate,
                                &namespace_id,
                                "交互裁决",
                                "user",
                            ));
                            ledger.sort_and_dedup();
                            match oxigraph_manager.sync_fusion_ledger(&ledger).await {
                                Ok(_) => println!(
                                    "已记录裁决：{} 优先来源 {}（「融合应用」后生效）",
                                    predicate, namespace_id
                                ),
                                Err(e) => println!("裁决保存失败：{}", e),
                            }
                        }
                        Ok(_) => println!(
                            "命名空间「{}」不在账本中：先用 `namespace` 子命令导入来源",
                            namespace
                        ),
                        Err(e) => println!("读取融合账本失败：{}", e),
                    }
                }
                None => println!("裁决格式：裁决：谓词=来源，例如「裁决：年龄=friend」"),
            }
            println!();
            continue;
        }

        // 溯因·假设生成：后向链失败前沿 → 候选假设（可证明则直接给证明）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "溯因：",
                "溯因:",
                "假设生成：",
                "假设生成:",
                "为什么：",
                "为什么:",
            ],
        ) {
            let target = target.trim().trim_end_matches(['？', '?', '。', ' ']);
            if target.is_empty() {
                println!("请提供要溯因的陈述，例如：溯因：张三是赵六的师祖");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.abduce_statement(target) {
                    Some(text) => println!("{}", text),
                    None => println!(
                        "无法把「{}」形式化为可溯因的目标。支持句式：S的P是O / O是S的P / S是类型。",
                        target
                    ),
                }
            }
            println!();
            continue;
        }

        // 引擎层故障同步：Prolog / 事实加载失败注入推理器，
        // 恢复模式开启时对话管理器会显式说「我出错了」，绝不并入「没有记录」。
        if dialogue_enabled {
            if let Some(fault) = prolog_engine.take_fault() {
                symbolic_reasoner.note_engine_fault(fault);
            }
        }

        // 假设世界命令先于挂账结算：避免「退出假设」「记住这个假设」被
        // 当成澄清答案吞掉。带「那么/则」的一次性反事实句仍走下方通道。
        if let Some(text) = symbolic_reasoner.handle_assumption_command(input) {
            println!("{}", text);
            println!();
            continue;
        }

        // 本体化对话管理器（纯对话行为）：回顾、解释依据、续聊、展开、寒暄。
        // 用对话本体 + 逻辑程序策略推导话语行为，再由模板组合生成回答。
        if dialogue_enabled {
            refresh_dialogue_quality(
                &mut dialogue_manager,
                &oxigraph_manager,
                &config,
                symbolic_reasoner.revision(),
                &store_path,
            )
            .await;
            if let Some(reply) = dialogue_manager.plan_pure_dialogue(input, &symbolic_reasoner) {
                let text = plugin_runtime.style_wrap(&reply.text);
                println!("{}", text);
                let provenance: Vec<String> = dialogue_manager
                    .turns()
                    .last()
                    .map(|turn| turn.provenance.clone())
                    .unwrap_or_default();
                log_chronicle_reply(&mut chronicle, &text, &reply.plan, &provenance);
                log_chronicle_trace(&mut chronicle, &dialogue_manager);
                println!();
                continue;
            }
        }

        // "为什么……？"：先尝试后向链证明，未得证则生成假设解释
        if let Some(target) = input.strip_prefix("为什么") {
            let target = target.trim().trim_end_matches(['？', '?', '。', ' ']);
            if target.is_empty() {
                println!("请提供要解释的结论，例如：为什么张三是王五的师祖？");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.abduce_statement(target) {
                    Some(text) => println!("{}", text),
                    None => println!(
                        "无法把「{}」形式化为可解释的目标。支持句式：S的P是O / O是S的P / S是类型。",
                        target
                    ),
                }
            }
            println!();
            continue;
        }

        // 反事实模拟："如果……那么……" / "假设：……"，正向链推演后自动回滚
        if input.starts_with("如果")
            || input.starts_with("假如")
            || input.starts_with("假设")
            || strip_any_prefix(
                input,
                &["反事实：", "反事实:", "假设：", "假设:", "假如：", "假如:"],
            )
            .is_some()
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match symbolic_reasoner.what_if_report(input) {
                Some(text) => println!("{}", text),
                None => println!(
                    "无法把条件形式化为可模拟的假设。支持句式：如果S的P是O那么… / 假设：S是类型。"
                ),
            }
            println!();
            continue;
        }

        // 采纳待定假设：加入本轮会话事实集（不写知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "采纳假设",
                "采纳假设：",
                "采纳假设:",
                "确认假设",
                "确认假设：",
                "确认假设:",
            ],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => println!(
                    "{}",
                    symbolic_reasoner
                        .adopt_hypothesis(index)
                        .unwrap_or_else(|| format!(
                            "没有第 {} 条待定假设；可用「待定假设」查看列表。",
                            index
                        ))
                ),
                None => println!(
                    "请给出假设编号，例如：采纳假设 1。当前待定假设：\n{}",
                    symbolic_reasoner.pending_hypotheses_report()
                ),
            }
            println!();
            continue;
        }

        // 待定假设列表
        if matches!(input, "待定假设" | "假设列表" | "候选假设" | "溯因假设") {
            println!("{}", symbolic_reasoner.pending_hypotheses_report());
            println!();
            continue;
        }

        // 导入指令先于输入模式判定解析：简写（如「导入文件：」）不在 import_keywords
        // 中也可用；参数正文只做前缀剥离，路径/JSON 中的关键词不受影响。
        if let Some(directive) = parse_import_directive(input) {
            match directive {
                ImportDirective::JsonldFile(file_path) => {
                    if file_path.is_empty() {
                        println!("请提供 JSON-LD 文件路径，例如：导入jsonld文件：test_data.jsonld");
                        println!();
                        continue;
                    }
                    println!("正在导入 JSON-LD 文件: {}", file_path);
                    match oxigraph_manager.import_jsonld_from_file(&file_path).await {
                        Ok(stats) => {
                            println!("JSON-LD 文件导入成功：{}", stats.summary());
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_bulk_import(
                                    chronicle,
                                    "JSON-LD",
                                    &file_path,
                                    stats.triples,
                                    kb_revision + 1,
                                );
                            }
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 文件导入失败: {}", e);
                        }
                    }
                }
                ImportDirective::JsonldContent(jsonld_content) => {
                    if jsonld_content.is_empty() {
                        println!(
                            "请提供 JSON-LD 内容，例如：导入内容：{{\"@id\": \"…\", \"@type\": \"Person\"}}"
                        );
                        println!();
                        continue;
                    }
                    println!("正在导入 JSON-LD 内容...");
                    match oxigraph_manager.import_jsonld(&jsonld_content).await {
                        Ok(stats) => {
                            println!("JSON-LD 内容导入成功：{}", stats.summary());
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_bulk_import(
                                    chronicle,
                                    "JSON-LD",
                                    "内容",
                                    stats.triples,
                                    kb_revision + 1,
                                );
                            }
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 内容导入失败: {}", e);
                        }
                    }
                }
                ImportDirective::RdfFile(file_path) => {
                    if file_path.is_empty() {
                        println!(
                            "请提供 RDF 文件路径（支持 .ttl / .nt / .nq / .trig / .rdf / .xml）"
                        );
                        println!();
                        continue;
                    }
                    println!("正在导入 RDF 文件: {}", file_path);
                    match oxigraph_manager.import_rdf_file(&file_path).await {
                        Ok(added) => {
                            println!("RDF 文件导入成功：新增 {} 条三元组！", added);
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_bulk_import(
                                    chronicle,
                                    "RDF",
                                    &file_path,
                                    added,
                                    kb_revision + 1,
                                );
                            }
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("RDF 文件导入失败: {}", e);
                        }
                    }
                }
            }
            println!();
            continue;
        }

        // 导出默认图为 RDF 文件（格式按扩展名，只读）
        if let Some(directive) = parse_export_directive(input) {
            match directive {
                ExportDirective::RdfFile(file_path) => {
                    if file_path.is_empty() {
                        println!(
                            "请提供导出路径（.ttl / .nt / .nq / .trig / .rdf / .xml / .jsonld）"
                        );
                        println!();
                        continue;
                    }
                    match oxigraph_manager.export_rdf_file(&file_path).await {
                        Ok(count) => println!("导出成功：{} 条三元组 → {}", count, file_path),
                        Err(e) => println!("导出失败：{}", e),
                    }
                }
            }
            println!();
            continue;
        }

        // 任意 SPARQL 1.1 查询（只读，不改变知识库与推理器状态）
        if let Some(sparql_query) = parse_sparql_command(input) {
            if sparql_query.is_empty() {
                println!(
                    "请提供 SPARQL 查询，例如：SPARQL：SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT 10"
                );
                println!();
                continue;
            }
            match oxigraph_manager.query_sparql(sparql_query).await {
                Ok(result) => println!("{}", result.render_text()),
                Err(e) => println!("SPARQL 查询失败：{}", e),
            }
            println!();
            continue;
        }

        let mode = detect_input_mode(input, &config.nlp.import_keywords);

        match mode {
            InputMode::KnowledgeImport => {
                {
                    // 知识导入逻辑
                    println!("{}", config.messages.importing);

                    let clean_text = config
                        .nlp
                        .import_keywords
                        .iter()
                        .fold(input.to_string(), |text, keyword| {
                            // 关键词未出现时原样返回
                            if !text.contains(keyword.as_str()) {
                                return text;
                            }
                            text.replace(&format!("{}：", keyword), "")
                                .replace(&format!("{}:", keyword), "")
                        })
                        .trim()
                        .to_string();

                    let tagged_words = preprocess(&clean_text, &plugin_nlp);
                    let normalized_words = normalize_words(tagged_words, &synonyms);
                    let mut items = knowledge_extractor.extract(&clean_text, &normalized_words);
                    // 中文关系句 → 关系三元组（如「张三是李四的师傅」），
                    // 叠加 DCG 文法抽取（被字句/把字句/并列/所有格换向）
                    grammar_extractor
                        .set_lexicon_if_changed(&oxigraph_manager.predicate_query_pairs());
                    items.extend(extract_relationships_from_text(
                        &clean_text,
                        &normalized_words,
                        &grammar_extractor,
                    ));
                    // 已抽出显式关系时，通用兜底实体会把整句内容词拼成
                    // property（如「张三李四老师」），属于重复噪声，予以剔除；
                    // 具名类型实体（Person 等）保留。
                    if items
                        .iter()
                        .any(|item| matches!(item, KnowledgeItem::Relationship { .. }))
                    {
                        items.retain(|item| {
                            !matches!(item, KnowledgeItem::Entity { label, .. } if label == "Entity")
                        });
                    }

                    if items.is_empty() {
                        println!("{}", config.messages.no_knowledge_extracted);
                    } else {
                        for item in &items {
                            match item {
                                KnowledgeItem::Entity { label, properties } => {
                                    println!("实体 - 类型: {}, 属性: {:?}", label, properties);
                                }
                                KnowledgeItem::Relationship {
                                    from_label,
                                    from_id,
                                    to_label,
                                    to_id,
                                    rel_type,
                                    ..
                                } => {
                                    println!(
                                        "关系 - {}({}) -[{}]-> {}({})",
                                        from_label, from_id, rel_type, to_label, to_id
                                    );
                                }
                            }
                        }

                        match oxigraph_manager.save_knowledge_with_receipt(items).await {
                            Ok(receipt) => {
                                let summary = receipt.summary();
                                println!("{}", summary);
                                if let Some(chronicle) = chronicle.as_mut() {
                                    let _ = conversation_theory::append_import(
                                        chronicle,
                                        &receipt,
                                        "文本录入",
                                        kb_revision + 1,
                                    );
                                    if !receipt.conflicts.is_empty() {
                                        let _ = conversation_theory::append_conflict(
                                            chronicle,
                                            &receipt.conflicts,
                                            "文本录入",
                                            kb_revision + 1,
                                        );
                                    }
                                }
                                log_chronicle_reply(&mut chronicle, &summary, &[], &[]);
                                if receipt.can_undo() {
                                    import_journal.push(receipt);
                                    let _ = import_journal.save(&journal_path);
                                }
                                reasoner_stale = true;
                                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                                // 文本别名模式（全称/简称/又称…）→ 只提名，不自动改写身份
                                let candidates = extract_alias_candidates(&clean_text);
                                if !candidates.is_empty() {
                                    let outcomes =
                                        oxigraph_manager.nominate_entity_aliases(&candidates);
                                    let pending = outcomes
                                        .iter()
                                        .filter(|o| o.kind == NominationKind::Pending)
                                        .count();
                                    let conflicts = outcomes
                                        .iter()
                                        .filter(|o| o.kind == NominationKind::TypeConflict)
                                        .count();
                                    if pending > 0 || conflicts > 0 {
                                        println!(
                                            "身份提名：待确认 {} 个、类型冲突 {} 个（用「身份台账」查看，「确认身份：A = B」裁决）。",
                                            pending, conflicts
                                        );
                                    }
                                }
                                let _ =
                                    oxigraph_manager.save_entity_registry(&entity_registry_path);
                            }
                            Err(e) => eprintln!("保存失败: {}", e),
                        }
                    }
                }
            }
            InputMode::Query => {
                // 待确认的模式联想：先结算「是/不是」短答，再把原问句重放一次。
                // 未识别为短答时静默丢弃待确认状态，按新输入继续处理。
                let mut case_replay: Option<String> = None;
                if let Some((origin, suggestion)) = pending_case.take() {
                    let mut handled = true;
                    match pattern_ir::classify_case_reply(input) {
                        pattern_ir::CaseReply::Accept => {
                            match pattern_library.accept_case(&suggestion) {
                                Ok(entry) => {
                                    pattern_episodes.confirm_case(&suggestion);
                                    println!(
                                        "已确认案例：{}（来源 {}，路由 {}）。",
                                        entry.name, entry.source, entry.query_type
                                    );
                                    match oxigraph_manager
                                        .sync_pattern_ledger(pattern_library.entries())
                                        .await
                                    {
                                        Ok(count) => println!("模式落图：{} 条已同步。", count),
                                        Err(error) => {
                                            eprintln!("模式落图失败：{}", error);
                                        }
                                    }
                                    match oxigraph_manager
                                        .sync_pattern_episodes(pattern_episodes.entries())
                                        .await
                                    {
                                        Ok(count) => {
                                            println!("经验落图：{} 条已同步。", count)
                                        }
                                        Err(error) => {
                                            eprintln!("经验落图失败：{}", error);
                                        }
                                    }
                                    println!("正在用确认后的模式重新理解原问题…");
                                    case_replay = Some(origin);
                                    handled = false;
                                }
                                Err(error) => println!("确认联想失败：{}", error),
                            }
                        }
                        pattern_ir::CaseReply::Reject => {
                            if pattern_episodes.reject_case(&suggestion) {
                                println!("好的，已记下这次否定；该骨架不会再被主动推荐。");
                            } else {
                                println!("好的，已关闭这次模式联想。");
                            }
                            if let Err(error) = oxigraph_manager
                                .sync_pattern_episodes(pattern_episodes.entries())
                                .await
                            {
                                eprintln!("经验落图失败：{}", error);
                            }
                        }
                        pattern_ir::CaseReply::Unclear => {
                            println!("好的，先不确认；这次联想已关闭。");
                        }
                        pattern_ir::CaseReply::NewInput => {
                            handled = false;
                        }
                    }
                    if handled {
                        println!();
                        continue;
                    }
                }
                let input = case_replay.as_deref().unwrap_or(input);

                // 语境指称消歧（CRD）：先结算上一轮反问，再对本轮做绑定/改写/追问。
                crd_ledger.begin_turn();
                let mut crd_input: Option<String> = None;
                if let Some(pending) = crd_pending.take() {
                    match parse_clarify_act(input, &pending.question) {
                        Some(ClarifyAct::Choose { entity, .. })
                        | Some(ClarifyAct::Restate { entity }) => {
                            crd_ledger.bind(
                                &pending.key,
                                &entity,
                                BindingScope::Topic,
                                &format!("澄清回答：{}", input),
                            );
                            println!("（语境绑定：{} ⇒ {}）", pending.mention, entity);
                            crd_input = Some(rewrite_at(
                                &pending.original_input,
                                pending.start,
                                pending.mention.chars().count(),
                                &entity,
                            ));
                            let _ = crd_ledger.save_to_path(&reference_binding_path);
                        }
                        Some(ClarifyAct::Correct { to, .. }) => {
                            crd_ledger.correct(&pending.key, &to, &format!("澄清纠正：{}", input));
                            println!("（语境纠错：{} ⇒ {}）", pending.mention, to);
                            crd_input = Some(rewrite_at(
                                &pending.original_input,
                                pending.start,
                                pending.mention.chars().count(),
                                &to,
                            ));
                            let _ = crd_ledger.save_to_path(&reference_binding_path);
                        }
                        Some(ClarifyAct::Unknown) | Some(ClarifyAct::Decline) => {
                            println!("（已跳过这次澄清，仍按原句理解。）");
                        }
                        None => {
                            // 不是澄清回答：丢弃待定问题，按新输入继续。
                        }
                    }
                }
                // 二次确认的绑定 → 别名提名（类型闸门 + 人工复核，不自动改库）
                if crd_input.is_some() {
                    let nominations = crd_ledger.take_alias_nominations();
                    if !nominations.is_empty() {
                        let candidates: Vec<know_prolog::entity_registry::AliasCandidate> =
                            nominations
                                .iter()
                                .map(|nomination| know_prolog::entity_registry::AliasCandidate {
                                    surface: nomination.surface.clone(),
                                    canonical: nomination.canonical.clone(),
                                    pattern: "语境指称绑定".to_string(),
                                    evidence: nomination.evidence.clone(),
                                })
                                .collect();
                        let outcomes = oxigraph_manager.nominate_dialogue_aliases(&candidates);
                        for outcome in outcomes {
                            match outcome.kind {
                                know_prolog::entity_registry::NominationKind::Confirmed => {
                                    println!(
                                        "（语境绑定已获二次确认：「{}」≡「{}」，已登记为别名。）",
                                        outcome.surface, outcome.canonical
                                    );
                                    let _ = symbolic_reasoner.register_entity_alias(
                                        &outcome.surface,
                                        &outcome.canonical,
                                    );
                                }
                                know_prolog::entity_registry::NominationKind::Pending => {
                                    println!(
                                        "（语境绑定提名待确认：「{}」≡「{}」，用「身份台账」复核。）",
                                        outcome.surface, outcome.canonical
                                    );
                                }
                                know_prolog::entity_registry::NominationKind::TypeConflict => {
                                    println!(
                                        "（语境绑定提名类型冲突：「{}」≡「{}」，需人工裁决。）",
                                        outcome.surface, outcome.canonical
                                    );
                                }
                                _ => {}
                            }
                        }
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                    }
                }
                if crd_input.is_none()
                    && (contains_question_marker(input)
                        || input.contains('？')
                        || input.contains('?'))
                {
                    let crd_hints = symbolic_reasoner.reference_hints(input);
                    let outcome = crd_resolver.resolve_text(
                        input,
                        symbolic_reasoner.linker(),
                        &symbolic_reasoner,
                        &crd_ledger,
                        &crd_hints,
                        dialogue_manager.focus_entity(),
                    );
                    // 只在确有竞争/绑定/追问/描述改写时展示证据，避免每问都刷屏
                    let show_notes = outcome.ask.is_some()
                        || outcome.rewrite.is_some()
                        || outcome.resolutions.iter().any(|resolution| {
                            resolution.candidates.len() > 1
                                || resolution.verdict
                                    == know_prolog::contextual_reference::Verdict::Bound
                        });
                    if show_notes {
                        for note in &outcome.notes {
                            println!("（{}）", note);
                        }
                    }
                    if let Some(ask) = outcome.ask {
                        println!("{}", ask.question.render());
                        crd_pending = Some(ask);
                        println!();
                        continue;
                    }
                    if let Some(rewrite) = outcome.rewrite {
                        if rewrite != input {
                            println!("（语境改写：{}）", rewrite);
                        }
                        crd_input = Some(rewrite);
                    }
                }
                let crd_base_input = crd_input.unwrap_or_else(|| input.to_string());

                // 多轮对话上下文：指代消解与省略补全（话语帧合一，符号方式，无神经网络）
                let contextual_input =
                    match symbolic_reasoner.contextualize_with_receipt(&crd_base_input) {
                        Some(rewrite) => {
                            for note in &rewrite.notes {
                                println!("（{}）", note);
                            }
                            println!("（结合对话上下文理解为：{}）", rewrite.text);
                            rewrite.text
                        }
                        None => {
                            // 无源可填的实体洞：不落进「无法理解」，转为一次显式绑定请求
                            // （澄清 = 请求一次绑定；常规通道必然失败时才开启）。
                            if let Some(prompt) =
                                symbolic_reasoner.start_frame_clarification(&crd_base_input)
                            {
                                println!("{}", prompt);
                                println!();
                                continue;
                            }
                            crd_base_input
                        }
                    };
                let input = contextual_input.as_str();

                for note in symbolic_reasoner.entity_link_notes(input) {
                    println!("（{}）", note);
                }

                // 会话覆盖层优先：只回答本会话已确认的关系，绝不冒充图谱事实。
                if let Some(answer) = acquisition
                    .answer_from_overlay(input, &oxigraph_manager.predicate_registry_snapshot())
                {
                    println!("{}", answer.render_text());
                    log_chronicle_reply(&mut chronicle, &answer.render_text(), &[], &[]);
                    println!();
                    continue;
                }

                // 插件配方：只读既有知识源，模板必须全部槽位接地；取不到值则让位核心管线。
                if let Some(outcome) = plugin_outcome.as_ref() {
                    let focus = dialogue_manager.focus_entity().map(str::to_string);
                    if let Some(reply) = symbolic_reasoner.answer_plugin_recipe(
                        outcome,
                        &mut plugin_runtime,
                        focus.as_deref(),
                    ) {
                        println!("{}", reply.text);
                        if dialogue_enabled {
                            dialogue_manager.note_plugin_turn(
                                raw_input,
                                MoveKind::AskAttribute,
                                reply.entity.as_deref(),
                                reply.attribute.as_deref(),
                                &reply.text,
                                reply.plan.clone(),
                                reply.provenance.clone(),
                            );
                            if let Some(focus) = reply.focus.as_deref() {
                                dialogue_manager.note_plugin_focus(focus);
                            }
                        }
                        log_chronicle_reply(
                            &mut chronicle,
                            &reply.text,
                            &reply.plan,
                            &reply.provenance,
                        );
                        log_chronicle_trace(&mut chronicle, &dialogue_manager);
                        println!();
                        continue;
                    }
                }

                // 本体化对话管理器：实体已知的知识问句给出"结论 + 出处 + 追问建议"式组合回答
                if dialogue_enabled {
                    refresh_dialogue_quality(
                        &mut dialogue_manager,
                        &oxigraph_manager,
                        &config,
                        symbolic_reasoner.revision(),
                        &store_path,
                    )
                    .await;
                    if let Some(reply) = dialogue_manager.plan_reply(input, &symbolic_reasoner) {
                        let text = plugin_runtime.style_wrap(&reply.text);
                        println!("{}", text);
                        let provenance: Vec<String> = dialogue_manager
                            .turns()
                            .last()
                            .map(|turn| turn.provenance.clone())
                            .unwrap_or_default();
                        log_chronicle_reply(&mut chronicle, &text, &reply.plan, &provenance);
                        log_chronicle_trace(&mut chronicle, &dialogue_manager);
                        println!();
                        continue;
                    }
                }

                let tagged_words = preprocess(input, &plugin_nlp);
                let precomputed_query_entities =
                    extract_query_entities_from_tagged(&tagged_words, &config);
                let nlu_tagged_words = if config.nlp.enable_nlu.unwrap_or(false) {
                    Some(tagged_words.clone())
                } else {
                    None
                };
                let normalized_words = normalize_words(tagged_words, &synonyms);
                // 路由上下文：让带守卫的模式只在相应对话状态下参与匹配。
                let route_context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);

                // 使用NLU进行语义理解
                if config.nlp.enable_nlu.unwrap_or(false) {
                    let semantic_analysis = match nlu_tagged_words {
                        Some(tagged) => nlu_engine.analyze_tagged(input, tagged),
                        None => nlu_engine.analyze(input),
                    };

                    if !semantic_analysis.semantics.is_empty() {
                        let best_semantic = &semantic_analysis.semantics[0];
                        let confidence_threshold =
                            config.nlp.nlu_confidence_threshold.unwrap_or(0.6);

                        if best_semantic.score >= confidence_threshold {
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            println!("NLU语义理解结果:");
                            println!("  领域: {}", best_semantic.domain);
                            println!("  意图: {}", best_semantic.intent);
                            println!("  置信度: {:.2}", best_semantic.score);
                            if !best_semantic.slots.is_empty() {
                                println!("  词槽:");
                                for (slot_name, slot_value) in &best_semantic.slots {
                                    println!("    {}: {}", slot_name, slot_value);
                                }
                            }
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                            // 根据NLU结果处理查询
                            match best_semantic.domain.as_str() {
                                "天气" => {
                                    let city = best_semantic
                                        .slots
                                        .iter()
                                        .find(|(name, _)| name == "城市")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "未知城市".to_string());
                                    let date = best_semantic
                                        .slots
                                        .iter()
                                        .find(|(name, _)| name == "日期")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "今天".to_string());

                                    match best_semantic.intent.as_str() {
                                        "雨" | "天气查询" => {
                                            println!("正在查询{}{}的天气信息...", city, date);
                                            println!("（天气API未接入，尝试从本地知识库回答）");
                                        }
                                        _ => {
                                            println!(
                                                "暂不支持该天气查询类型: {}，尝试本地符号推理...",
                                                best_semantic.intent
                                            );
                                        }
                                    }
                                    // 天气 API 不可用时不能丢问题：回落到本地图谱/符号推理
                                    if let Some(answer) = try_symbolic_reasoning(
                                        &mut symbolic_reasoner,
                                        &mut contrast_arbiter,
                                        &oxigraph_manager,
                                        input,
                                        &precomputed_query_entities,
                                    )
                                    .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else {
                                        println!("本地知识库中没有相关天气记录。");
                                    }
                                }
                                "人物" => {
                                    let person = best_semantic
                                        .slots
                                        .iter()
                                        .find(|(name, _)| name == "人名")
                                        .map(|(_, value)| value.clone());

                                    if let Some(person_name) = person {
                                        match best_semantic.intent.as_str() {
                                            "身高查询" => {
                                                println!("正在查询{}的身高信息...", person_name);
                                                // 本地适配器返回了实体但没有身高/高度属性时，
                                                // 不能就此收尾：回落到本地图谱与符号推理
                                                let found_height = match local_semantic_adapter
                                                    .get_entity_knowledge(&person_name)
                                                    .await
                                                {
                                                    Ok(response) => {
                                                        println!(
                                                            "─────────────────────────────────"
                                                        );
                                                        println!("实体: {}", response.data.entity);
                                                        println!("描述: {}", response.data.desc);
                                                        let mut found = false;
                                                        for avp in &response.data.avp {
                                                            if avp.len() >= 2
                                                                && (avp[0].contains("身高")
                                                                    || avp[0].contains("高度"))
                                                            {
                                                                println!("身高: {}", avp[1]);
                                                                found = true;
                                                            }
                                                        }
                                                        println!(
                                                            "─────────────────────────────────"
                                                        );
                                                        found
                                                    }
                                                    Err(e) => {
                                                        println!("获取{}身高信息时出错: {}，回退到本地符号推理...", person_name, e);
                                                        false
                                                    }
                                                };
                                                if !found_height {
                                                    if let Some(answer) = try_symbolic_reasoning(
                                                        &mut symbolic_reasoner,
                                                        &mut contrast_arbiter,
                                                        &oxigraph_manager,
                                                        input,
                                                        &precomputed_query_entities,
                                                    )
                                                    .await
                                                    {
                                                        println!("{}", answer);
                                                        log_chronicle_reply(
                                                            &mut chronicle,
                                                            &answer,
                                                            &[],
                                                            &[],
                                                        );
                                                    } else {
                                                        println!(
                                                            "知识库中没有{}的身高记录。",
                                                            person_name
                                                        );
                                                    }
                                                }
                                            }
                                            "年龄查询" => {
                                                println!("正在查询{}的年龄信息...", person_name);
                                                if let Some(answer) = try_symbolic_reasoning(
                                                    &mut symbolic_reasoner,
                                                    &mut contrast_arbiter,
                                                    &oxigraph_manager,
                                                    input,
                                                    &precomputed_query_entities,
                                                )
                                                .await
                                                {
                                                    println!("{}", answer);
                                                    log_chronicle_reply(
                                                        &mut chronicle,
                                                        &answer,
                                                        &[],
                                                        &[],
                                                    );
                                                } else {
                                                    println!(
                                                        "知识库中没有{}的年龄记录。",
                                                        person_name
                                                    );
                                                }
                                            }
                                            _ => {
                                                println!("暂不支持该人物查询类型: {}，尝试本地符号推理...", best_semantic.intent);
                                                if let Some(answer) = try_symbolic_reasoning(
                                                    &mut symbolic_reasoner,
                                                    &mut contrast_arbiter,
                                                    &oxigraph_manager,
                                                    input,
                                                    &precomputed_query_entities,
                                                )
                                                .await
                                                {
                                                    println!("{}", answer);
                                                    log_chronicle_reply(
                                                        &mut chronicle,
                                                        &answer,
                                                        &[],
                                                        &[],
                                                    );
                                                }
                                            }
                                        }
                                    } else {
                                        println!("未能识别人名，请重新输入");
                                    }
                                }
                                "聊天" => {
                                    let chat_reply = match best_semantic.intent.as_str() {
                                        "问候" => {
                                            "你好！我是KnowProlog知识图谱系统，很高兴为您服务！"
                                        }
                                        "感谢" => "不客气！如果您有其他问题，随时可以问我。",
                                        _ => "很高兴和您聊天！",
                                    };
                                    println!("{}", chat_reply);
                                    log_chronicle_reply(&mut chronicle, chat_reply, &[], &[]);
                                }
                                "知识查询" => {
                                    // 继续使用查询逻辑
                                    if let Some((query_type, entities, pattern_note)) =
                                        route_query_with_patterns(
                                            &pattern_library,
                                            input,
                                            &normalized_words,
                                            &mut pattern_episodes,
                                            &route_context,
                                        )
                                    {
                                        // 执行查询流程
                                        if let Some(note) = &pattern_note {
                                            println!("命中模式: {}", note);
                                            record_forge_hit(&mut pattern_forge, note, &forge_path);
                                        }
                                        continue_original_query(
                                            &config,
                                            &oxigraph_manager,
                                            &mut prolog_engine,
                                            &response_formatter,
                                            &local_semantic_adapter,
                                            &external_knowledge_service,
                                            &mut symbolic_reasoner,
                                            &mut contrast_arbiter,
                                            &mut plugin_runtime,
                                            input,
                                            &query_type,
                                            &entities,
                                            &precomputed_query_entities,
                                        )
                                        .await;
                                    } else if let Some(answer) = try_symbolic_reasoning(
                                        &mut symbolic_reasoner,
                                        &mut contrast_arbiter,
                                        &oxigraph_manager,
                                        input,
                                        &precomputed_query_entities,
                                    )
                                    .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some((text, plan)) =
                                        plugin_policy_text(&mut plugin_runtime, input)
                                    {
                                        println!("{}", text);
                                        log_chronicle_reply(
                                            &mut chronicle,
                                            &text,
                                            &[plan.clone()],
                                            &[format!("插件策略 {}", plan)],
                                        );
                                    } else {
                                        let fallback = "无法理解您的查询，请重新描述您的问题。";
                                        println!("{}", fallback);
                                        log_chronicle_reply(&mut chronicle, fallback, &[], &[]);
                                    }
                                }
                                _ => {
                                    println!("暂不支持该领域: {}", best_semantic.domain);
                                    // 如果NLU不能处理，回退到查询流程
                                    if let Some((query_type, entities, pattern_note)) =
                                        route_query_with_patterns(
                                            &pattern_library,
                                            input,
                                            &normalized_words,
                                            &mut pattern_episodes,
                                            &route_context,
                                        )
                                    {
                                        if let Some(note) = &pattern_note {
                                            println!("命中模式: {}", note);
                                            record_forge_hit(&mut pattern_forge, note, &forge_path);
                                        }
                                        continue_original_query(
                                            &config,
                                            &oxigraph_manager,
                                            &mut prolog_engine,
                                            &response_formatter,
                                            &local_semantic_adapter,
                                            &external_knowledge_service,
                                            &mut symbolic_reasoner,
                                            &mut contrast_arbiter,
                                            &mut plugin_runtime,
                                            input,
                                            &query_type,
                                            &entities,
                                            &precomputed_query_entities,
                                        )
                                        .await;
                                    } else if let Some(answer) = try_symbolic_reasoning(
                                        &mut symbolic_reasoner,
                                        &mut contrast_arbiter,
                                        &oxigraph_manager,
                                        input,
                                        &precomputed_query_entities,
                                    )
                                    .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some((text, plan)) =
                                        plugin_policy_text(&mut plugin_runtime, input)
                                    {
                                        println!("{}", text);
                                        log_chronicle_reply(
                                            &mut chronicle,
                                            &text,
                                            &[plan.clone()],
                                            &[format!("插件策略 {}", plan)],
                                        );
                                    } else {
                                        println!("{}", config.messages.cannot_understand);
                                        log_chronicle_reply(
                                            &mut chronicle,
                                            &config.messages.cannot_understand,
                                            &[],
                                            &[],
                                        );
                                    }
                                }
                            }
                            continue; // 跳过后续查询逻辑
                        }
                    }

                    // 如果NLU置信度不够，使用传统模式处理
                    println!("NLU置信度不足，使用传统模式处理查询...");
                }

                // 检查是否是直接的实体知识查询（例如："刘德华的信息" 或 "告诉我关于刘德华"）
                if input.contains("的信息")
                    || input.contains("告诉我关于")
                    || input.contains("什么是")
                {
                    // 提取实体名
                    let entity_name = input
                        .replace("的信息", "")
                        .replace("告诉我关于", "")
                        .replace("什么是", "")
                        .trim()
                        .to_string();

                    if !entity_name.is_empty() {
                        println!("正在获取实体 '{}' 的知识...", entity_name);

                        match local_semantic_adapter
                            .get_entity_knowledge(&entity_name)
                            .await
                        {
                            Ok(response) => {
                                println!("─────────────────────────────────");
                                println!("实体: {}", response.data.entity);
                                println!("描述: {}", response.data.desc);
                                if !response.data.tag.is_empty() {
                                    println!("标签: {}", response.data.tag.join(", "));
                                }
                                println!("\n属性信息:");
                                for avp in &response.data.avp {
                                    if avp.len() >= 2 {
                                        println!("  {}: {}", avp[0], avp[1]);
                                    }
                                }
                                println!("─────────────────────────────────");
                                continue;
                            }
                            Err(e) => {
                                println!("获取实体知识时出错: {}，尝试本地查询...", e);
                            }
                        }
                    }
                }

                // 检查是否是实体属性查询（例如："刘德华的年龄" 或 "苹果的颜色"）
                let attribute_re = ENTITY_ATTRIBUTE_RE
                    .get_or_init(|| regex::Regex::new(r"(.+?)的(.+?)$").unwrap());
                if let Some(captures) = attribute_re.captures(input) {
                    if let (Some(entity_match), Some(attr_match)) =
                        (captures.get(1), captures.get(2))
                    {
                        let entity_name = entity_match.as_str().trim();
                        let attribute = attr_match.as_str().trim();

                        // 排除已经处理过的模式与符号推理问句（谁/吗/几/多少等）
                        if !input.contains("信息")
                            && !input.contains("什么")
                            && !input.contains("谁")
                            && !input.contains("吗")
                            && !input.contains("呢")
                            && !input.contains("几")
                            && !input.contains("多少")
                        {
                            println!("正在查询实体 '{}' 的属性 '{}'...", entity_name, attribute);

                            match local_semantic_adapter
                                .get_entity_attribute_value(entity_name, attribute)
                                .await
                            {
                                Ok(response) => {
                                    println!("─────────────────────────────────");
                                    println!("实体: {}", response.data.entity);
                                    println!("属性: {}", response.data.attribute);
                                    println!("值: {}", response.data.value.join(", "));
                                    println!("─────────────────────────────────");
                                    continue;
                                }
                                Err(e) => {
                                    println!("获取属性值时出错: {}，尝试本地查询...", e);
                                }
                            }
                        }
                    }
                }

                if let Some((query_type, entities, pattern_note)) = route_query_with_patterns(
                    &pattern_library,
                    input,
                    &normalized_words,
                    &mut pattern_episodes,
                    &route_context,
                ) {
                    println!("查询类型: {}, 实体: {:?}", query_type, entities);
                    if let Some(note) = &pattern_note {
                        println!("命中模式: {}", note);
                        record_forge_hit(&mut pattern_forge, note, &forge_path);
                    }

                    // 提取查询中的实体
                    let mut seen_entities = std::collections::HashSet::new();
                    let mut query_entities: Vec<String> =
                        Vec::with_capacity(precomputed_query_entities.len() + entities.len());
                    for entity in precomputed_query_entities.iter().chain(entities.iter()) {
                        if seen_entities.insert(entity.as_str()) {
                            query_entities.push(entity.clone());
                        }
                    }

                    // === 歧义解析 ===
                    if let Some(api_config) = &config.api {
                        if api_config.enable_entity_disambiguation {
                            let resolved_entities = resolve_query_entities_with_disambiguation(
                                input,
                                &query_entities,
                                api_config,
                                &local_semantic_adapter,
                            )
                            .await;

                            // 使用解析后的实体进行查询
                            let final_entities = if !resolved_entities.is_empty() {
                                resolved_entities
                            } else {
                                entities.clone()
                            };

                            // 仅加载相关知识子集
                            let max_hops = 2; // 可配置的跳跃数
                            match oxigraph_manager
                                .get_relevant_facts(&final_entities, max_hops)
                                .await
                            {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);

                                    // 先尝试直接从RDF数据获取答案
                                    if let Some(answer) = oxigraph_manager
                                        .get_direct_answer(&query_type, &final_entities)
                                        .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some(prolog_query) =
                                        PrologEngine::generate_prolog_query(
                                            &query_type,
                                            &final_entities,
                                        )
                                    {
                                        println!("Prolog查询: {}", prolog_query);

                                        let response = run_prolog_query(
                                            &mut prolog_engine,
                                            &mut symbolic_reasoner,
                                            &prolog_query,
                                            &query_type,
                                            &response_formatter,
                                        );
                                        println!("{}", response);
                                        log_chronicle_reply(&mut chronicle, &response, &[], &[]);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", config.messages.process_error, e);
                                }
                            }
                        } else {
                            // 歧义解析功能被禁用，直接查询
                            let max_hops = 2;
                            match oxigraph_manager
                                .get_relevant_facts(&query_entities, max_hops)
                                .await
                            {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);

                                    if let Some(answer) = oxigraph_manager
                                        .get_direct_answer(&query_type, &entities)
                                        .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some(prolog_query) =
                                        PrologEngine::generate_prolog_query(&query_type, &entities)
                                    {
                                        println!("Prolog查询: {}", prolog_query);

                                        let response = run_prolog_query(
                                            &mut prolog_engine,
                                            &mut symbolic_reasoner,
                                            &prolog_query,
                                            &query_type,
                                            &response_formatter,
                                        );
                                        println!("{}", response);
                                        log_chronicle_reply(&mut chronicle, &response, &[], &[]);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", config.messages.process_error, e);
                                }
                            }
                        }
                    } else {
                        // 没有API配置，直接查询
                        let max_hops = 2;
                        match oxigraph_manager
                            .get_relevant_facts(&query_entities, max_hops)
                            .await
                        {
                            Ok(facts) => {
                                println!("加载了 {} 个相关事实", facts.len());
                                prolog_engine.load_relevant_facts(facts);

                                if let Some(answer) = oxigraph_manager
                                    .get_direct_answer(&query_type, &entities)
                                    .await
                                {
                                    println!("{}", answer);
                                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                } else if let Some(prolog_query) =
                                    PrologEngine::generate_prolog_query(&query_type, &entities)
                                {
                                    println!("Prolog查询: {}", prolog_query);

                                    let response = run_prolog_query(
                                        &mut prolog_engine,
                                        &mut symbolic_reasoner,
                                        &prolog_query,
                                        &query_type,
                                        &response_formatter,
                                    );
                                    println!("{}", response);
                                    log_chronicle_reply(&mut chronicle, &response, &[], &[]);
                                } else {
                                    println!("无法生成有效的Prolog查询");
                                }
                            }
                            Err(e) => {
                                println!("{}: {}", config.messages.process_error, e);
                            }
                        }
                    }
                } else if let Some(answer) = try_symbolic_reasoning(
                    &mut symbolic_reasoner,
                    &mut contrast_arbiter,
                    &oxigraph_manager,
                    input,
                    &precomputed_query_entities,
                )
                .await
                {
                    println!("{}", answer);
                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                } else {
                    // 先区分「实体未知」与「问法不支持」：库里完全没有提到过的实体，
                    // 坦白说没有记录，比笼统的「不理解」更接近人类对话
                    let mut missing_entities: Vec<String> = Vec::new();
                    for entity in &precomputed_query_entities {
                        if entity.chars().count() < 2 {
                            continue;
                        }
                        match oxigraph_manager
                            .get_relevant_facts(std::slice::from_ref(entity), 0)
                            .await
                        {
                            Ok(facts) if facts.is_empty() => missing_entities.push(entity.clone()),
                            _ => {}
                        }
                    }
                    if missing_entities.is_empty() {
                        println!("{}", config.messages.cannot_understand);
                    } else {
                        let names = missing_entities
                            .iter()
                            .map(|name| format!("「{}」", name))
                            .collect::<Vec<_>>()
                            .join("、");
                        println!(
                            "我的知识库里暂时没有{}的记录。如果你想教我，可以说「录入：{}…」。",
                            names,
                            missing_entities.first().map(String::as_str).unwrap_or("")
                        );
                    }

                    // 问法未命中但存在相似成功经验：给出可确认的候选骨架，
                    // 把「没听懂」变成一次可复用的上下文学习机会。
                    if let Some(suggestion) = pattern_episodes.suggest_case_with_context(
                        &normalized_words,
                        &route_context,
                        0.2,
                    ) {
                        println!(
                            "模式联想：{}（你是在问这个意思吗？答「是」确认，答「不是」否定；也可用「确认联想：{}」）",
                            suggestion.pattern.render(),
                            input
                        );
                        pending_case = Some((input.to_string(), suggestion));
                    }

                    // 插件声明的兜底策略先于外链/常识别源；只改措辞，不伪造事实。
                    if let Some((policy_text, policy_plan)) = plugin_runtime.policy_reply(raw_input)
                    {
                        let text = plugin_runtime.style_wrap(&policy_text);
                        println!("{}", text);
                        log_chronicle_reply(
                            &mut chronicle,
                            &text,
                            &[policy_plan.clone()],
                            &[format!("插件策略 {}", policy_plan)],
                        );
                    } else if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                        log_chronicle_reply(
                            &mut chronicle,
                            &format!("根据常识知识库：{}", direct_answer),
                            &[],
                            &[],
                        );
                    } else {
                        // 尝试使用外链知识（名册优先，未命中才打网，回答带出处）
                        match external_knowledge_service
                            .answer_question_with_entities(
                                input,
                                &external_candidates(&[], &precomputed_query_entities),
                            )
                            .await
                        {
                            Ok(outcome) => {
                                if let Some(chronicle) = chronicle.as_mut() {
                                    match &outcome {
                                        AnswerOutcome::Answered(answer) => {
                                            let _ = conversation_theory::append_external(
                                                chronicle,
                                                &answer.entity,
                                                "回答",
                                                &answer.source_uri,
                                                kb_revision,
                                            );
                                        }
                                        AnswerOutcome::KnownAbsent {
                                            entity, source_uri, ..
                                        } => {
                                            let _ = conversation_theory::append_external(
                                                chronicle,
                                                entity,
                                                "确认未收录",
                                                source_uri,
                                                kb_revision,
                                            );
                                        }
                                        AnswerOutcome::Unknown => {}
                                    }
                                }
                                match external_outcome_text(&outcome) {
                                    Some(text) => {
                                        println!("{}", text);
                                        log_chronicle_reply(&mut chronicle, &text, &[], &[]);
                                    }
                                    None => {
                                        let fallback =
                                            "很抱歉，我无法理解您的问题。请尝试重新描述。";
                                        println!("{}", fallback);
                                        log_chronicle_reply(&mut chronicle, fallback, &[], &[]);
                                    }
                                }
                            }
                            Err(e) => {
                                println!("查询外部知识时出错：{}。请检查网络连接。", e);
                            }
                        }
                    }
                }
            }
        }

        println!();
    }

    if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
        match symbolic_reasoner
            .save_dialogue_state_with_contrast(state_path, Some(&contrast_arbiter.snapshot()))
        {
            Ok(_) => println!(
                "对话记忆已保存到 {}（含对比校准记忆：{}）",
                state_path,
                contrast_arbiter.memory_summary()
            ),
            Err(e) => eprintln!("保存对话记忆失败: {}", e),
        }
    }

    if dialogue_enabled {
        if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
            let manager_path = format!("{}.manager.json", state_path);
            match dialogue_manager.save_to_path(&manager_path) {
                Ok(_) => println!(
                    "对话本体记忆已保存到 {}（{}）",
                    manager_path,
                    dialogue_manager.summary()
                ),
                Err(e) => eprintln!("保存对话本体记忆失败: {}", e),
            }
        }
    }

    // 模式经验落图：会话中积累的命中/未命中经验跨重启可检索。
    if !pattern_episodes.is_empty() {
        match oxigraph_manager
            .sync_pattern_episodes(pattern_episodes.entries())
            .await
        {
            Ok(count) => println!("模式经验已保存（经验落图：{} 条）。", count),
            Err(error) => eprintln!("保存模式经验失败：{}", error),
        }
    }

    // 外链名册落盘：跨重启保留取回的外部知识（出处 + 逻辑时刻）
    match external_knowledge_service.save_ledger(&external_ledger_path) {
        Ok(_) => println!(
            "外链名册已保存到 {}（{} 条条目）。",
            external_ledger_path,
            external_knowledge_service.ledger_len()
        ),
        Err(e) => eprintln!("保存外链名册失败: {}", e),
    }

    persist_acquisition(&acquisition, &acquisition_path);
    persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
    println!("系统已退出。");
}

async fn load_graph_snapshot(
    oxigraph_manager: &OxigraphManager,
) -> Result<
    Arc<Vec<know_prolog::prolog_engine::PrologFact>>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    oxigraph_manager.load_graph_snapshot().await
}

/// 把对话层已确认的会话事实桥接为推理前提（幂等；永不写知识图谱）。
///
/// 确认/更正发生在对话管理器里，推理器需要同一组事实才能让多跳链、
/// 证明与会话快照看到它们；返回本轮新增条数。
fn sync_recovery_confirmed_facts(
    dialogue_manager: &DialogueManager,
    reasoner: &mut SymbolicReasoner,
) -> usize {
    let state = dialogue_manager.recovery_state();
    for fact in &state.confirmed {
        reasoner.note_dialogue_verdict(
            &fact.entity,
            &fact.attribute,
            &fact.value,
            true,
            &fact.receipt_id,
        );
    }
    for candidate in &state.refuted {
        reasoner.note_dialogue_verdict(
            &candidate.entity,
            &candidate.attribute,
            &candidate.value,
            false,
            &candidate.receipt_id,
        );
    }
    state
        .confirmed
        .iter()
        .filter(|fact| reasoner.adopt_session_triple(&fact.entity, &fact.attribute, &fact.value))
        .count()
}

async fn refresh_symbolic_reasoner(
    symbolic_reasoner: &mut SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) {
    if let Ok(facts) = load_graph_snapshot(oxigraph_manager).await {
        symbolic_reasoner.observe_facts(&facts);
        oxigraph_manager.observe_entity_facts(&facts);
    }
}

/// 重建推理器的全局知识（用于删值/更正）：保留会话态，只携带当前三元组。
async fn rebuild_symbolic_reasoner(
    symbolic_reasoner: &mut SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) {
    if let Ok(facts) = load_graph_snapshot(oxigraph_manager).await {
        symbolic_reasoner.rebuild_from_graph(&facts);
        oxigraph_manager.observe_entity_facts(&facts);
    }
}

/// 规则图账本同步：把当前会话规则全量重写进命名图（幂等；不推进知识版本）。
async fn sync_rule_ledger_report(
    symbolic_reasoner: &SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) {
    let entries = symbolic_reasoner.rule_ledger_entries();
    match oxigraph_manager.sync_rule_ledger(&entries).await {
        Ok(0) => println!("规则图账本已同步：0 条规则（命名图 kg:graphs/rules 已置空）。"),
        Ok(count) => println!(
            "规则图账本已同步：{} 条规则 → 命名图 kg:graphs/rules（指纹 {}）。",
            count,
            symbolic_reasoner.rule_fingerprint()
        ),
        Err(error) => eprintln!("规则图账本同步失败：{}", error),
    }
}

/// 图账本审计：规则账本与派生账本读取 + 与当前会话对账（指纹/版本过期判定）。
async fn graph_ledger_report(
    symbolic_reasoner: &SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) -> String {
    let mut output = String::from("图账本审计（命名图物化视图，与知识图谱隔离）：\n");
    match oxigraph_manager.load_rule_ledger().await {
        Ok(ledger) => {
            if !ledger.synced {
                output.push_str(
                    "■ 规则账本 kg:graphs/rules：尚未同步过（教一条「规则：…」即会入账）。\n",
                );
            } else {
                output.push_str(&format!(
                    "■ 规则账本 kg:graphs/rules：{} 条规则，规则集指纹 {}。\n",
                    ledger.entries.len(),
                    ledger.fingerprint
                ));
                for (index, entry) in ledger.entries.iter().take(10).enumerate() {
                    output.push_str(&format!(
                        "  {}. {}（来源：{}；置信度 {:.2}）\n",
                        index + 1,
                        entry.display_zh(),
                        entry.source,
                        entry.confidence
                    ));
                }
                if ledger.entries.len() > 10 {
                    output.push_str(&format!("  …共 {} 条。\n", ledger.entries.len()));
                }
                let current = symbolic_reasoner.rule_fingerprint();
                if ledger.fingerprint == current {
                    output.push_str("  对账：与当前会话规则集一致。\n");
                } else {
                    output.push_str(&format!(
                        "  对账：账本指纹 {} ≠ 当前会话指纹 {}——会话有未同步变更。\n",
                        ledger.fingerprint, current
                    ));
                }
            }
        }
        Err(error) => output.push_str(&format!("■ 规则账本读取失败：{}\n", error)),
    }
    match oxigraph_manager.load_derived_ledger().await {
        Ok(ledger) => {
            if !ledger.materialized {
                output
                    .push_str("■ 派生账本 kg:graphs/derived：尚未物化过（用「物化推理」写入）。\n");
            } else {
                output.push_str(&format!(
                    "■ 派生账本 kg:graphs/derived：{} 条派生事实，物化于知识版本 {}，规则指纹 {}。\n",
                    ledger.entries.len(),
                    ledger.revision,
                    ledger.fingerprint
                ));
                for (index, entry) in ledger.entries.iter().take(10).enumerate() {
                    let premises = entry
                        .premises
                        .iter()
                        .map(|premise| premise.display())
                        .collect::<Vec<_>>()
                        .join("、");
                    output.push_str(&format!(
                        "  {}. {} ← {}（轮次 {}，置信度 {:.2}；前提：{}）\n",
                        index + 1,
                        entry.atom.display(),
                        entry.rule_source,
                        entry.depth,
                        entry.confidence,
                        premises
                    ));
                }
                if ledger.entries.len() > 10 {
                    output.push_str(&format!("  …共 {} 条。\n", ledger.entries.len()));
                }
                let current_revision = oxigraph_manager.revision();
                let current_fingerprint = symbolic_reasoner.rule_fingerprint();
                if ledger.revision == current_revision && ledger.fingerprint == current_fingerprint
                {
                    output.push_str("  对账：物化戳与当前知识版本、规则集一致。\n");
                } else {
                    output.push_str(&format!(
                        "  对账：已过期（物化于版本 {} / 指纹 {}；当前版本 {} / 指纹 {}）——用「物化推理」重写。\n",
                        ledger.revision, ledger.fingerprint, current_revision, current_fingerprint
                    ));
                }
            }
        }
        Err(error) => output.push_str(&format!("■ 派生账本读取失败：{}\n", error)),
    }
    output.trim_end().to_string()
}

/// 从知识图谱快照抽取关系边（子图视图的输入；确定性排序由视图层负责）。
fn prolog_fact_edges(facts: &[PrologFact]) -> Vec<(String, String, String)> {
    facts
        .iter()
        .filter_map(|fact| match fact {
            PrologFact::Relation { from, rel_type, to } => {
                Some((from.clone(), rel_type.clone(), to.clone()))
            }
            PrologFact::Entity { .. } => None,
        })
        .collect()
}

/// 编年史回复登记：与写前日志配对，失败不打断主流程。
fn log_chronicle_reply(
    chronicle: &mut Option<Chronicle>,
    answer: &str,
    plan: &[String],
    provenance: &[String],
) {
    if let Some(chronicle) = chronicle.as_mut() {
        let _ = chronicle.append_reply(answer, plan, provenance);
    }
}

/// 编年史轨迹归档：把对话管理器最近一轮的分层追踪写入 `kind="trace"`，
/// 重启后可据此回放问据栈/指代轨迹（不重放决策）。
fn log_chronicle_trace(chronicle: &mut Option<Chronicle>, manager: &DialogueManager) {
    if manager.trace_mode().is_off() {
        return;
    }
    if let (Some(chronicle), Some(trace)) = (chronicle.as_mut(), manager.last_turn_trace_json()) {
        let _ = chronicle.append("trace", trace);
    }
}

/// 外链候选实体：预计算实体在前、本轮解析实体在后，去重保序。
fn external_candidates(entities: &[String], precomputed: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();
    for entity in precomputed.iter().chain(entities.iter()) {
        if seen.insert(entity.clone()) {
            merged.push(entity.clone());
        }
    }
    merged
}

/// 外链结果的可展示文本（回答带出处标注）；`None` 表示没有可用信息。
fn external_outcome_text(outcome: &AnswerOutcome) -> Option<String> {
    match outcome {
        AnswerOutcome::Answered(answer) => {
            Some(format!("根据外部知识：{}", answer.with_provenance()))
        }
        AnswerOutcome::KnownAbsent {
            entity,
            source_uri,
            fetched_tick,
        } => Some(format!(
            "外部知识源已确认未收录「{}」（来源：{}，第 {} 轮核验）。",
            entity, source_uri, fetched_tick
        )),
        AnswerOutcome::Unknown => None,
    }
}

/// 交互式导入指令的解析结果。
#[derive(Debug, PartialEq, Eq)]
enum ImportDirective {
    JsonldFile(String),
    JsonldContent(String),
    RdfFile(String),
}

/// 解析「导入…文件：路径」/「导入…内容：内容」指令。
///
/// 兼容全角「：」与半角「:」，并接受 README 中的简写「导入文件：」「导入内容：」；
/// 只剥离前缀关键词，参数正文不做替换（路径/JSON 中再次出现关键词也不会被删改）。
fn parse_import_directive(input: &str) -> Option<ImportDirective> {
    let trimmed = input.trim();
    let lower = trimmed.to_ascii_lowercase();
    let groups: [(&[&str], fn(String) -> ImportDirective); 3] = [
        (
            &[
                "导入jsonld内容",
                "导入内容",
                "import jsonld content",
                "import content",
            ],
            ImportDirective::JsonldContent,
        ),
        (
            &[
                "导入jsonld文件",
                "导入文件",
                "import jsonld file",
                "import jsonld",
            ],
            ImportDirective::JsonldFile,
        ),
        (
            &["导入rdf文件", "导入rdf", "import rdf"],
            ImportDirective::RdfFile,
        ),
    ];
    for (keywords, build) in groups {
        for keyword in keywords {
            let needle = keyword.to_ascii_lowercase();
            for colon in [':', '：'] {
                let prefix = format!("{}{}", needle, colon);
                if lower.starts_with(&prefix) {
                    return Some(build(trimmed[prefix.len()..].trim().to_string()));
                }
            }
        }
    }
    None
}

/// 交互式导出指令的解析结果。
#[derive(Debug, PartialEq, Eq)]
enum ExportDirective {
    RdfFile(String),
}

/// 解析「导出rdf文件：路径」指令（兼容全角/半角冒号与「导出文件：」简写）。
fn parse_export_directive(input: &str) -> Option<ExportDirective> {
    let trimmed = input.trim();
    let lower = trimmed.to_ascii_lowercase();
    for keyword in ["导出rdf文件", "导出rdf", "导出文件", "export rdf"] {
        let needle = keyword.to_ascii_lowercase();
        for colon in [':', '：'] {
            let prefix = format!("{}{}", needle, colon);
            if lower.starts_with(&prefix) {
                return Some(ExportDirective::RdfFile(
                    trimmed[prefix.len()..].trim().to_string(),
                ));
            }
        }
    }
    None
}

/// 解析「SPARQL：<查询>」指令（兼容全角/半角冒号与 `sparql <查询>` 空格形式）。
///
/// 只剥离前缀，查询正文原样保留（其中的冒号/关键词不受影响）。
fn parse_sparql_command(input: &str) -> Option<&str> {
    let trimmed = input.trim();
    let lower = trimmed.to_ascii_lowercase();
    for (needle, suffix_len) in [("sparql：", "sparql：".len()), ("sparql:", "sparql:".len())] {
        if lower.starts_with(needle) {
            return Some(trimmed[suffix_len..].trim());
        }
    }
    if lower.starts_with("sparql") {
        let rest = &trimmed["sparql".len()..];
        if rest.starts_with(|c: char| c.is_whitespace()) {
            return Some(rest.trim());
        }
    }
    // 中文等价的「查询sparql：」（半角/全角冒号长度不同，按实际前缀切分）
    if lower.starts_with("查询sparql：") {
        return Some(trimmed["查询SPARQL：".len()..].trim());
    }
    if lower.starts_with("查询sparql:") {
        return Some(trimmed["查询SPARQL:".len()..].trim());
    }
    None
}

/// 交互模式命令速查（只列常用命令，完整命令集见各分支实现与 CLAUDE.md）。
fn interactive_help() -> String {
    [
        "KnowProlog 交互命令速查（输入「帮助 / help / ?」显示本页）",
        "",
        "知识录入",
        "  录入：<文本> / 添加：<文本>        自然语言抽取实体与关系（如「录入：小明18岁」）",
        "  导入jsonld文件：<路径>              导入 JSON-LD 文件（简写：导入文件：）",
        "  导入jsonld内容：<JSON>              直接导入 JSON-LD 内容（简写：导入内容：）",
        "  导入rdf文件：<路径>                 导入 Turtle/N-Triples/N-Quads/TriG/RDF/XML",
        "  导出rdf文件：<路径>                 导出默认图（按扩展名，支持 jsonld；简写：导出文件：）",
        "  撤销导入 / 导入回执 / 导入历史       导入回执与可撤销导入栈",
        "  情景 / 对话情景                      把编年史编译成可查询的情景理论（CST）",
        "  情景历史：<问题>                     同一问题在各情景下的回答与修订",
        "  情景归因：<问题>                     改口的候选事件原因（无候选就明说）",
        "  情景反事实：<问题>                   撤销最近语义事件后回答是否翻转（可弃权）",
        "  情景查询：<Datalog原子>              在情景理论上执行查询，如 answered_at(S,Q,A,P)",
        "",
        "查询与推理",
        "  直接输入自然语言                    如「小明多大？」「张三和李四什么关系？」",
        "  列出所有人的身高 / 哪些实体          查询板：集合查询后用「第二个 / 下一页 / 只看… / 按…排序 / 跟…比 / 依据」继续编辑",
        "  查询板 / 查询状态                    查看当前查询程序、结果行数与知识版本",
        "  查询剧本：<file.toml>              回放查询板剧本（每轮断言算子/行数/页码/选中/证据），自检交互行为",
        "  SPARQL：<查询>                      执行任意 SPARQL 1.1 查询（SELECT/ASK/CONSTRUCT）",
        "  判断：<陈述> / 蕴含：<陈述>         形式化语义 + 语义蕴含校验",
        "  证明：<目标> / 推理链：<目标>        归结证明与后向链推理",
        "  为什么<陈述> / 溯因：<陈述>         证明失败前沿生成候选假设",
        "  如果…那么… / 假设：…                 反事实推演；无「那么」时进入跨轮假设世界",
        "  退出假设 / 记住这个假设 / 假设世界    丢弃假设 / 并入会话事实 / 查看当前假设",
        "  默认推理：<陈述> / 概率推理：…       非单调常识与加权可能世界",
        "  规则：<如果X的P是Y，那么X的Q是Y>    教一条 Horn 规则（占位符自动变量化）",
        "  模式路由 / 查询模式库                 统一模式库（配置查询+NLU 意图+教学模式）与体检",
        "  模式关系 / 模式关系图                 泛化/冗余/冲突落成的图边与图体检查询",
        "  模式落图 / 同步模式图                 把模式库、关系边与模式经验同步到命名图",
        "  模式匹配：<句子>                     查看命中模式、绑定与落选原因",
        "  模式经验 / 经验台账                  查看命中/未命中的可检索理解经验",
        "  模式联想：<句子>                     用相似成功经验泛化候选骨架（只确认，不直接回答）",
        "  确认联想：<句子>                     用户确认候选骨架，转正为可路由模式并落图",
        "  是 / 不是                            直接结算上一轮模式联想；否定记入负向反馈",
        "  忘经验：<短 id 或原句>               删除单条经验（撤销误确认/误否定）",
        "  清空模式经验                         清空经验台账并同步命名图",
        "  教模式：名 | 要素… | 路由 | 优先级 | 守卫  教一条模式（可选上下文守卫，落图持久化）",
        "  锻造模式：名 | 正例(多个用;分隔) | 反例(可空，多个用;分隔) | 路由 | 优先级 | 守卫",
        "  确认锻造 <id> / 否认锻造 <id> / 退役锻造 <id> / 遗忘锻造 <id>  转正、记反驳、停用、删除",
        "  锻造台账                            例子、正则、探针、命中/反驳与生命周期",
        "  锻造探针 [id]                       重跑正例/反例/逆生成探针，发现悄悄失效的规则",
        "  忘模式：<名>                         遗忘一条教学模式",
        "  记住：<事实> / 忘记：<事实>          会话事实/否定断言与撤回（否定只阻断路径，不写图）",
        "  改口：<事实>                        显式更正会话槽（记录 before/after，可撤销，不写图）",
        "  槽账本 / 改口史：<实体>的<属性>      查看当前赢家、压制值与逐次改口史",
        "  更正：<实体>的<属性>从<旧>改为<新>  删除旧值、写入新值，并记录可追踪的更正事件",
        "  修订预演：<实体>|<属性>|<旧>|<新>   提交前计算影响与冲突（只计算，不写图）",
        "  修订回执 / 撤销修订                查看最近写入回执 / 按逆操作恢复最新一笔",
        "  规则预演：<如果…那么…>              教规则前检查重复/不安全变量/功能冲突与影响",
        "  信念检查 / 信念一致性              功能谓词多值、定义域/值域与规则形状体检",
        "  历史可信度：<问题>                   沿支持图查看回答是可信/可疑/失效/未评估",
        "",
        "台账与报告",
        "  对话记忆 / 帧记忆 / 对话本体          对话状态、挂账帧与策略程序",
        "  追踪报告 / 指代轨迹 / 问据栈          分层追踪：问据栈、指代绑定与逐轮决策",
        "  追踪为什么 <编号>                     沿追踪图回溯某决策的依据链",
        "  身份台账 / 谓词报告 / 律典 / 类型体系  实体身份、谓词画像、语义定律与类型格",
        "  绑定账本                              语境指称绑定：谁的说法指哪个实体（CRD）",
        "  关联规则 / 推理发现 / 全息状态        联想规则、挖掘发现与全息记忆",
        "  补全回合：<实体>                     打开一次学习回合：收集候选、确认归纳、本批闭包",
        "  确认补全：<序号> / 否定补全：<序号>   裁定候选；确认事实只进会话，不写图谱",
        "  补全规则 / 补全报告                   已归纳规则与本回合候选（默认本批作用域）",
        "  补全提问 / 下一个问题                 按效用选一个候选追问（重复提问会降权）",
        "  补全测试 / 补全重放                   运行期裁决回归；重放会把违规规则隔离",
        "  别问了 / 允许提问                     长期静音或恢复主动追问（随会话持久化）",
        "  知识层次 / 层次清单 / 会话镜像        四层知识容器：基础/会话/推测/假设与摘要",
        "  晋升补全规则：<id>|<作用域>           命名空间留在回合内；只有「全局」进入全局归结",
        "  退役补全规则：<id> / 恢复补全规则：<id>  停用或恢复规则；已确认的会话事实保留",
        "  对齐报告 / 对齐候选：<术语>          本体对齐、候选格与答案等价性",
        "  覆盖报告 / 质量护照                   覆盖缺口与数据质量",
        "  关系获取：S 的 P / 获取台账         以问代抽：槽位追问、会话账本、覆盖层与诱导模式",
        "  获取缺口 / 获取模式 / 模式扫描：…    形状缺口转追问、诱导模式库与文本扫描",
        "  确认获取 / 采纳获取 / 模式扫描：…    确认、写入会话事实集、用诱导模式扫描新文本",
        "  形状报告 / 形状校验                  必备属性、基数、类型与定义域/值域校验",
        "  形状画像：<实体>                     形状完整画像：已知/缺失/冲突/未知 + 一次缺口追问",
        "  裁定：<实体> 的 <谓词> = <值>         会话内收敛冲突（不写图谱）",
        "  下一个形状问题 / 形状议程             按严重度继续追问，或查看当前形状缺口议程",
        "  形状对话报告 / 形状对话重置           会话补全进度与记忆状态",
        "  对比报告 / 待定推测 / 性能报告        神经符号双通道、校准与运行统计",
        "  外链台账 / 图账本                     外部知识留册与命名图物化视图",
        "  融合 / 融合应用 / 裁决：谓词=来源     多来源融合视图、物化与显式裁决",
        "  配置契约 / 配置体检                   config.toml 引用/能力/阈值静态检查",
        "",
        "证明与视图",
        "  证明图：<目标> / 知识子图：<实体>     证明树 / 邻域子图",
        "  证明日志 / 证明编号：<id>             回放带稳定 id 的证明凭证",
        "  一致性 / 改口检查                     同题异答、过期回答与未答问题",
        "  时间线 / 编年史 / 回放 <N>            对话编年史审计与重建",
        "  相似实体：<实体> / 介绍：<实体>       相似度排序与生成式介绍",
        "  证据检索：<问题>                    带约束与上界剪枝的证据排序及检索回执",
        "",
        "其他",
        "  帮助 / help / ?                      显示本页",
        "  quit / 退出                          退出系统",
    ]
    .join("\n")
}

/// 解析「同义链接：别名 = 规范实体」的两端（支持半角/全角等号）。
fn parse_alias_link(body: &str) -> Option<(&str, &str)> {
    for separator in ['=', '＝'] {
        if let Some((alias, target)) = body.split_once(separator) {
            let alias = alias.trim();
            let target = target.trim();
            if !alias.is_empty() && !target.is_empty() {
                return Some((alias, target));
            }
        }
    }
    None
}

/// 本轮对话实际用到的边：最近一轮对话实例的焦点实体 + 所问属性。
/// 无属性时退化为「焦点实体触达的边」，让用户至少看到话题邻域。
fn latest_used_edges(
    dialogue_manager: &DialogueManager,
    edges: &[(String, String, String)],
) -> std::collections::HashSet<(String, String, String)> {
    let Some(turn) = dialogue_manager.turns().last() else {
        return std::collections::HashSet::new();
    };
    let entity = turn
        .entity
        .as_deref()
        .or_else(|| dialogue_manager.focus_entity());
    let attribute = turn.attribute.as_deref();
    edges
        .iter()
        .filter(|(subject, predicate, object)| {
            let touches = entity
                .map(|entity| subject == entity || object == entity)
                .unwrap_or(false);
            match attribute {
                Some(attribute) => predicate == attribute && touches,
                None => touches,
            }
        })
        .cloned()
        .collect()
}

async fn try_symbolic_reasoning(
    symbolic_reasoner: &mut SymbolicReasoner,
    contrast_arbiter: &mut ContrastArbiter,
    oxigraph_manager: &OxigraphManager,
    input: &str,
    query_entities: &[String],
) -> Option<String> {
    if !query_entities.is_empty() {
        match oxigraph_manager.get_relevant_facts(query_entities, 2).await {
            Ok(facts) => symbolic_reasoner.observe_facts(&facts),
            Err(error) => {
                symbolic_reasoner.note_engine_fault(format!("知识库读取相关事实失败：{}", error))
            }
        }
    }
    // 陈述句优先：形式化语义 + 语义蕴含判定（校验用户断言）；
    // 判为"未知"时追加溯因分析——后向链失败前沿生成候选假设（可解释的推测）；
    // 最后统一做神经符号对比（联想通道 vs 符号通道），输出仲裁结论并写入会话记忆
    if !contains_question_marker(input) {
        let report = symbolic_reasoner.entailment_report(input);
        if !report.starts_with("形式化失败") {
            let mut combined = report;
            if combined.contains("未知（证据不足）") {
                if let Some(abduction) = symbolic_reasoner.abduce_statement(input) {
                    combined.push('\n');
                    combined.push_str(&abduction);
                }
                // 默认逻辑（可反驳常识）补充：把「通常…除非…」知识用于陈述校验
                if let Some(supplement) = symbolic_reasoner.default_logic_supplement(input) {
                    combined.push('\n');
                    combined.push_str(&supplement);
                }
            }
            let case = contrast_arbiter.contrast_statement(input, symbolic_reasoner);
            combined.push('\n');
            combined.push_str(&contrast_arbiter.render_case_brief(&case));
            return Some(combined);
        }
    }
    if let Some(answer) = symbolic_reasoner.answer(input) {
        // 图谱无记录时，用溯因假设生成候选答案（LLM 式的"猜测 + 理由"），
        // 再追加默认逻辑（可反驳常识「通常…除非…」）补充证据
        if answer.contains("没有记录")
            || answer.contains("没有相关记录")
            || answer.contains("数量为 0")
        {
            let mut combined = answer;
            if let Some(abduction) = symbolic_reasoner.hypothesis_answer(input) {
                combined.push('\n');
                combined.push_str(&abduction);
            }
            if let Some(supplement) = symbolic_reasoner.default_logic_supplement(input) {
                combined.push('\n');
                combined.push_str(&supplement);
            }
            return Some(combined);
        }
        return Some(answer);
    }
    // 开放问句无直接答案：尝试溯因生成候选假设
    if let Some(abduction) = symbolic_reasoner.hypothesis_answer(input) {
        return Some(abduction);
    }
    // 仍无结果时：默认逻辑（非单调常识）补充证据，避免结果抽取断层
    if let Some(supplement) = symbolic_reasoner.default_logic_supplement(input) {
        return Some(supplement);
    }
    // 逻辑问答失败时，优先用 OWL+DCG 话语引擎做连贯叙述，
    // 再回退到"知识图谱即语言模型"的统计生成式介绍
    let entity = symbolic_reasoner.best_linked_entity(input)?;
    symbolic_reasoner
        .narrate(&entity)
        .or_else(|| symbolic_reasoner.generate_about(&entity))
}

/// 同一轮内区分「查无结果」与「引擎故障」：故障显式回执并写入会话故障状态。
fn run_prolog_query(
    prolog_engine: &mut PrologEngine,
    symbolic_reasoner: &mut SymbolicReasoner,
    prolog_query: &str,
    query_type: &str,
    response_formatter: &ResponseFormatter,
) -> String {
    match prolog_engine.query_checked(prolog_query) {
        Ok(results) => response_formatter.format_prolog_results(&results, query_type),
        Err(error) => {
            symbolic_reasoner.note_engine_fault(error.clone());
            format!("推理引擎故障（不是查无记录）：{}", error)
        }
    }
}

fn contains_question_marker(text: &str) -> bool {
    // 疑问词识别走虚词算符表（含同形守卫）
    know_prolog::function_words::contains_question_marker(text)
}

/// 判断输入是否以系统命令开头：澄清进行中时遇到命令应让位给常规处理
/// 插件声明式兜底：返回（可直接打印的文本，贡献路径）。
fn plugin_policy_text(
    runtime: &mut know_prolog::plugin::PluginRuntime,
    input: &str,
) -> Option<(String, String)> {
    let (text, plan) = runtime.policy_reply(input)?;
    Some((runtime.style_wrap(&text), plan))
}

fn is_command_like(input: &str) -> bool {
    const COMMANDS: &[&str] = &[
        "SPARQL",
        "sparql",
        "Sparql",
        "查询SPARQL",
        "查询sparql",
        "导出rdf文件",
        "导出rdf",
        "导出文件",
        "export rdf",
        "全息",
        "对话本体",
        "对话状态",
        "追踪报告",
        "追踪状态",
        "追踪图",
        "trace",
        "指代轨迹",
        "指代报告",
        "问据栈",
        "问据报告",
        "QUD",
        "追踪为什么",
        "查询板",
        "查询状态",
        "工作记忆",
        "查询剧本",
        "运行剧本",
        "溯因",
        "假设生成",
        "为什么",
        "证明",
        "归结",
        "推理链",
        "寻找证明",
        "场景",
        "联合解释",
        "解释场景",
        "采纳假设",
        "确认假设",
        "待定假设",
        "假设列表",
        "候选假设",
        "介绍",
        "生成介绍",
        "叙述",
        "讲述",
        "详细叙述",
        "复述",
        "教本体",
        "本体教学",
        "续写",
        "接龙",
        "补全",
        "补全回合",
        "确认补全",
        "否定补全",
        "驳回补全",
        "补全规则",
        "补全报告",
        "补全测试",
        "补全回归",
        "知识回归",
        "补全测试报告",
        "补全重放",
        "重放补全测试",
        "补全测试重放",
        "下一个问题",
        "该问什么",
        "主动提问",
        "别问了",
        "不要追问",
        "静音补全提问",
        "停止提问",
        "允许提问",
        "恢复提问",
        "解除静音",
        "晋升补全规则",
        "退役补全规则",
        "补全规则退役",
        "恢复补全规则",
        "补全规则恢复",
        "知识层次",
        "层次清单",
        "容器清单",
        "会话镜像",
        "回答层次",
        "关联规则",
        "关联联想",
        "联想",
        "推理发现",
        "推导发现",
        "对齐报告",
        "本体对齐",
        "对齐候选",
        "推理规则",
        "链接预测规则",
        "实体链接",
        "提及账本",
        "提及格",
        "格解码",
        "准入",
        "收录",
        "身份台账",
        "身份注册表",
        "绑定账本",
        "指称账本",
        "语境账本",
        "身份报告",
        "确认身份",
        "记住身份",
        "拒绝身份",
        "忘记别名",
        "合并身份",
        "铸造检查",
        "识别实体",
        "判断",
        "蕴含",
        "校验",
        "形式化",
        "相似实体",
        "语义相似",
        "证据检索",
        "检索证据",
        "对话记忆",
        "帧记忆",
        "话语帧",
        "挂账",
        "如果",
        "假如",
        "反事实",
        "澄清",
        "结构联想",
        "图游走",
        "话题延伸",
        "软推理",
        "加权推理",
        "因果",
        "干预",
        "反事实干预",
        "do：",
        "do:",
        "梯度归因",
        "可微推理",
        "因果分词",
        "分词：",
        "分词:",
        "概率推理",
        "可能世界",
        "默认推理",
        "默认逻辑",
        "教学：",
        "教学:",
        "教我",
        "概率知识",
        "ProbLog",
        "probLog",
        "规则",
        "教规则",
        "学习规则",
        "记住",
        "记录",
        "断言",
        "更正",
        "纠正",
        "修正",
        "改口",
        "更正记忆",
        "纠正记忆",
        "槽账本",
        "槽位账本",
        "改口账本",
        "改口史",
        "槽历史",
        "忘记",
        "遗忘",
        "撤回",
        "撤销导入",
        "导入回执",
        "导入历史",
        "导入日志",
        "撤销历史",
        "外链",
        "同义链接",
        "配置契约",
        "配置检查",
        "配置体检",
        "config-lint",
        "性能报告",
        "性能台账",
        "运行统计",
        "派生台账",
        "版本台账",
        "增量报告",
        "图账本",
        "派生图账本",
        "物化推理",
        "推理物化",
        "物化派生",
        "会话规则",
        "信念检查",
        "信念一致性",
        "修订回执",
        "会话修订",
        "撤销修订",
        "对比",
        "校准",
        "仲裁",
        "编年史",
        "对话编年史",
        "对话档案",
        "档案校验",
        "校验编年史",
        "回放",
        "聊过",
        "查档案",
        "时间线",
        "对话时间线",
        "证明图",
        "证明树",
        "画证明",
        "证明日志",
        "证明档案",
        "证明记录",
        "证明编号",
        "一致性",
        "回答一致性",
        "改口检查",
        "改口",
        "日志视图",
        "情景",
        "对话情景",
        "情景理论",
        "情景历史",
        "情景归因",
        "情景反事实",
        "情景查询",
        "历史可信度",
        "回答可信度",
        "信任报告",
        "可信度",
        "知识子图",
        "子图",
        "邻域",
        "覆盖报告",
        "关系获取",
        "获取关系",
        "获取缺口",
        "关系缺口",
        "追问缺口",
        "缺口追问",
        "问答获取",
        "获取台账",
        "获取报告",
        "关系获取报告",
        "问答获取报告",
        "获取状态",
        "获取模式",
        "关系获取模式",
        "诱导模式",
        "模式库",
        "获取覆盖",
        "会话覆盖层",
        "覆盖层",
        "确认获取",
        "确认关系获取",
        "否认获取",
        "否认关系获取",
        "拒绝获取",
        "撤回获取",
        "撤销获取",
        "采纳获取",
        "采纳关系获取",
        "提交获取",
        "模式扫描",
        "退役模式",
        "形状报告",
        "形状校验",
        "SHACL报告",
        "shacl报告",
        "shapes",
        "形状画像",
        "对话画像",
        "形状对话",
        "形状议程",
        "形状缺口议程",
        "下一个形状问题",
        "下一个缺口问题",
        "裁定",
        "冲突裁定",
        "会话裁定",
        "质量报告",
        "质量护照",
        "质量总览",
        "护照总览",
        "类型体系",
        "类型报告",
        "类型格",
        "覆盖检查",
        "虚词表",
        "功能词表",
        "虚词报告",
        "算符表",
        // 谓词画像 / 身份裁决
        "谓词报告",
        "谓词画像",
        "谓词注册表",
        "谓词列表",
        "律典",
        "定律台账",
        "语义定律",
        "law-report",
        "登记别名",
        "登记逆谓词",
        "功能谓词",
        "取消功能谓词",
        "确认谓词",
        "拒绝谓词",
        // 对比记忆的追问与待定推测
        "待定推测",
        // 会话规则库别名
        "后向链规则",
        // 澄清子命令（澄清进行中不应被当作回答）
        "开始澄清",
        "停止澄清",
        "结束澄清",
        "退出澄清",
        "提问确认",
        // 对话续说 / 换角度
        "接着说",
        "继续说",
        "继续介绍",
        "继续讲",
        "接着讲",
        "换个角度说",
        "换个说法",
        "换个角度",
        "换一个角度",
        "重新介绍",
        "讲讲",
        "说说",
        // 撤销上次导入
        "撤销上次导入",
        // 声明式插件
        "插件",
        "启用插件",
        "停用插件",
        "禁用插件",
        "为什么这么答",
        // 统一模式库（Pattern IR）
        "模式路由",
        "查询模式库",
        "统一模式库",
        "模式关系",
        "查询模式关系",
        "模式关系图",
        "模式图体检",
        "模式落图",
        "同步模式图",
        "保存模式库",
        "模式经验",
        "模式经验报告",
        "经验台账",
        "模式联想：",
        "模式联想:",
        "模式联想 ",
        "案例联想：",
        "案例联想:",
        "确认联想：",
        "确认联想:",
        "确认联想 ",
        "采纳联想：",
        "采纳联想:",
        "采纳联想 ",
        "案例转正：",
        "案例转正:",
        "案例转正 ",
        "忘经验：",
        "忘经验:",
        "忘经验 ",
        "遗忘经验：",
        "遗忘经验:",
        "清空模式经验",
        "遗忘模式经验",
        "模式匹配：",
        "模式匹配:",
        "模式匹配 ",
        "匹配模式：",
        "匹配模式:",
        "教模式：",
        "教模式:",
        "教模式 ",
        "教学模式：",
        "教学模式:",
        "锻造模式：",
        "锻造模式:",
        "锻造模式 ",
        "锻造模式",
        "模式锻造：",
        "模式锻造:",
        "模式锻造",
        "确认锻造：",
        "确认锻造:",
        "确认锻造 ",
        "采纳锻造：",
        "采纳锻造:",
        "否认锻造：",
        "否认锻造:",
        "否认锻造 ",
        "否决锻造：",
        "否决锻造:",
        "退役锻造：",
        "退役锻造:",
        "退役锻造 ",
        "停用锻造：",
        "停用锻造:",
        "遗忘锻造：",
        "遗忘锻造:",
        "遗忘锻造 ",
        "删除锻造：",
        "删除锻造:",
        "锻造台账",
        "锻造报告",
        "锻造状态",
        "模式锻造台账",
        "锻造探针",
        "复核锻造",
        "锻造探针报告",
        "锻造探针：",
        "锻造探针:",
        "锻造探针 ",
        "复核锻造：",
        "复核锻造:",
        "忘模式：",
        "忘模式:",
        "忘模式 ",
        "遗忘模式：",
        "遗忘模式:",
        // 帮助与退出
        "帮助",
        "help",
        "?",
        "退出",
        "exit",
    ];
    COMMANDS.iter().any(|command| input.starts_with(command))
}

/// 帧澄清(input) 的一步处理结果：绑定后重放 / 本轮已消化 / 放行常规管线
enum FrameClarificationStep {
    Replay(String),
    Handled,
    Pass,
}

/// 帧澄清进行中的短答处理：序号/候选名/已知实体名 → 确定性绑定；
/// 越界 → 重问（零状态变化）；取消词 → 释放；命令或非回答 → 放行。
fn handle_frame_clarification(
    reasoner: &mut SymbolicReasoner,
    input: &str,
) -> FrameClarificationStep {
    if !reasoner.frame_clarification_active() || is_command_like(input) {
        return FrameClarificationStep::Pass;
    }
    match reasoner.answer_frame_clarification(input) {
        Some(ClarificationOutcome::Bound { question, note }) => {
            println!("（{}）", note);
            FrameClarificationStep::Replay(question)
        }
        Some(ClarificationOutcome::Cancelled { note }) => {
            println!("（{}）", note);
            println!();
            FrameClarificationStep::Handled
        }
        Some(ClarificationOutcome::Reprompt { prompt }) => {
            println!("{}", prompt);
            println!();
            FrameClarificationStep::Handled
        }
        None => FrameClarificationStep::Pass,
    }
}

/// 解析干预命令的后缀 `=真/假`（默认 do(…=假)：移除该事实再观察下游变化）
fn parse_intervention_value(target: &str) -> (&str, bool) {
    for (suffix, value) in [
        ("=真", true),
        ("=对", true),
        ("=true", true),
        ("=1", true),
        ("=假", false),
        ("=错", false),
        ("=false", false),
        ("=0", false),
    ] {
        if let Some(stripped) = target.strip_suffix(suffix) {
            return (stripped.trim(), value);
        }
    }
    (target, false)
}

fn strip_any_prefix<'a>(input: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    prefixes
        .iter()
        .find_map(|prefix| input.strip_prefix(prefix))
}

/// 当前对话状态快照：供模式守卫、经验检索与案例联想共用同一口径。
fn dialogue_pattern_context(
    dialogue_manager: &DialogueManager,
    symbolic_reasoner: &SymbolicReasoner,
) -> pattern_ir::PatternContext {
    let focus = dialogue_manager.focus_entity().map(str::to_string);
    pattern_ir::PatternContext {
        focus_type: focus
            .as_deref()
            .and_then(|entity| symbolic_reasoner.context().focus_type(entity))
            .map(str::to_string),
        focus,
        last_move: dialogue_manager
            .turns()
            .last()
            .map(|turn| turn.kind.label().to_string()),
        pending_clarify: dialogue_manager.pending_reference().is_some(),
        turn: dialogue_manager.turn_count(),
    }
}

/// 用相似成功经验为一句输入生成候选骨架（确认前不参与路由）。
fn suggest_case_for_text(
    text: &str,
    config: &know_prolog::SystemConfig,
    synonyms: &std::collections::HashMap<String, String>,
    episodes: &pattern_ir::PatternEpisodeLog,
    context: &pattern_ir::PatternContext,
) -> Option<pattern_ir::CaseSuggestion> {
    let tagged = preprocess(text, &config.nlp);
    let normalized = normalize_words(tagged, synonyms);
    episodes.suggest_case_with_context(&normalized, context, 0.2)
}

/// 统一模式库路由：返回（查询类型、实体、命中模式解释行）。
///
/// 路由语义与既有 `analyze_query_intent(&words, &rules)` 一致（同排序、同匹配、
/// 同实体提取），额外给出「命中模式」的人类可读说明；无命中返回 `None`。
/// 每次尝试都会写入 [`pattern_ir::PatternEpisodeLog`]：成功记下命中模式与绑定，
/// 失败也留下可检索的情境经验（不改变路由结果）。
fn route_query_with_patterns(
    library: &PatternLibrary,
    input: &str,
    words: &[know_prolog::nlp::TaggedWord],
    episodes: &mut pattern_ir::PatternEpisodeLog,
    context: &pattern_ir::PatternContext,
) -> Option<(String, Vec<String>, Option<String>)> {
    let Some(matched) = library
        .match_route_with_context(words, context)
        .or_else(|| library.match_fallback_with_context(words, context))
    else {
        // 失败也要留下可行动的粒度：记录最近受阻模式停在哪一要素。
        let frontier_report = frontier::analyze_tokens(input, words, library, context);
        let snapshot = frontier_report.closest_blocked().and_then(|row| {
            row.failure
                .as_ref()
                .map(|failure| pattern_ir::EpisodeFrontier {
                    pattern_id: row.id.clone(),
                    pattern_name: row.name.clone(),
                    query_type: row.query_type.clone(),
                    coverage: row.coverage(),
                    failure: failure.clone(),
                })
        });
        episodes.record_unmatched_with_frontier(input, words, context, snapshot);
        return None;
    };
    episodes.record_match_with_context(input, words, &matched, context);
    let guard = matched
        .entry
        .guard
        .as_ref()
        .map(|guard| format!("，守卫：{}", guard.render()))
        .unwrap_or_default();
    let note = format!(
        "{}（{}，优先级 {}，来源 {}{}）",
        matched.entry.name,
        matched.entry.render(),
        matched.entry.priority,
        matched.entry.source,
        guard
    );
    Some((
        matched.entry.query_type.clone(),
        matched.entities.clone(),
        Some(note),
    ))
}

/// 解析「左 = 右」形式的谓词登记参数（支持 = / ＝ / =>）
fn parse_predicate_pair(spec: &str) -> Option<(String, String)> {
    let spec = spec.trim();
    for separator in ["=>", "=", "＝"] {
        if let Some((left, right)) = spec.split_once(separator) {
            let left = left.trim();
            let right = right.trim();
            if !left.is_empty() && !right.is_empty() {
                return Some((left.to_string(), right.to_string()));
            }
        }
    }
    None
}

/// 解析更正命令：`实体|属性|旧值|新值` 或 `实体 的 属性 从 旧值 改为 新值`。
///
/// 只认「有旧值」的更正：修正一条不知道旧值的陈述不是更正，而是无条件覆盖。
fn parse_correction_spec(target: &str) -> Option<(String, String, String, String)> {
    let trimmed = target
        .trim()
        .trim_matches(|ch: char| matches!(ch, '。' | '！' | '!' | '；' | ';' | '，' | ','));
    if trimmed.is_empty() {
        return None;
    }
    for separator in ['|', '｜'] {
        if trimmed.contains(separator) {
            let parts: Vec<&str> = trimmed
                .split(separator)
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .collect();
            if parts.len() == 4 {
                return Some((
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                    parts[3].to_string(),
                ));
            }
        }
    }
    let body = trimmed
        .strip_prefix("把")
        .or_else(|| trimmed.strip_prefix("将"))
        .unwrap_or(trimmed)
        .trim();
    for marker in ["改为", "改成", "更正为", "修正为", "变为", "换成"] {
        let Some(position) = body.find(marker) else {
            continue;
        };
        let left = body[..position].trim();
        let new_value = body[position + marker.len()..]
            .trim()
            .trim_matches(|ch: char| matches!(ch, '。' | '！' | '!' | '；' | ';' | '，' | ','));
        if new_value.is_empty() {
            continue;
        }
        for old_marker in ["从", "由", "原值为", "原值", "原本为", "原本"] {
            let Some(index) = left.rfind(old_marker) else {
                continue;
            };
            let head = left[..index].trim();
            let old_value = left[index + old_marker.len()..]
                .trim()
                .trim_start_matches([':', '：', '='])
                .trim();
            if old_value.is_empty() {
                continue;
            }
            let head = head.strip_suffix('是').unwrap_or(head).trim();
            if let Some((entity, predicate)) = head.rsplit_once('的') {
                let entity = entity.trim().trim_start_matches(['把', '将']).trim();
                let predicate = predicate.trim();
                if !entity.is_empty() && !predicate.is_empty() {
                    return Some((
                        entity.to_string(),
                        predicate.to_string(),
                        old_value.to_string(),
                        new_value.to_string(),
                    ));
                }
            }
        }
    }
    None
}

/// 中文句子 → 关系三元组：只取确定事实、二元、无占位符的原子，
/// 跳过「是」断言（类型断言不落成关系）。
/// 两条抽取通道合并去重：
///   1. 模式解析器 `parse_lessons`（「记住：」同款）；
///   2. DCG 文法抽取（被字句/把字句/并列/所有格换向/词库归一）。
fn extract_relationships_from_text(
    text: &str,
    words: &[know_prolog::nlp::TaggedWord],
    grammar_extractor: &GrammarExtractor,
) -> Vec<KnowledgeItem> {
    use know_prolog::chinese_reasoning_nlu::{parse_lessons, LessonKind};
    let outcome = parse_lessons(text);
    let mut seen: std::collections::BTreeSet<(String, String, String)> =
        std::collections::BTreeSet::new();
    let mut items = Vec::new();
    let mut push_triple = |from: String, predicate: String, to: String| {
        if predicate == "是" || from.is_empty() || to.is_empty() || from == to {
            return;
        }
        let key = (from.clone(), predicate.clone(), to.clone());
        if !seen.insert(key) {
            return;
        }
        items.push(KnowledgeItem::Relationship {
            from_label: String::new(),
            from_id: from,
            to_label: String::new(),
            to_id: to,
            rel_type: predicate,
            properties: std::collections::HashMap::new(),
        });
    };

    for lesson in outcome.lessons {
        if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
            continue;
        }
        let terms = &lesson.head.terms;
        if terms.len() != 2 {
            continue;
        }
        push_triple(
            terms[0].name().trim().to_string(),
            lesson.head.predicate.to_string(),
            terms[1].name().trim().to_string(),
        );
    }

    for triple in grammar_extractor.extract_from_tagged(words) {
        push_triple(triple.subject, triple.predicate, triple.object);
    }

    items
}

/// 把谓词画像注册表的词面同步到符号推理器：
/// 正向词面（别名 → 槽位）走别名通道（对齐同义词 + 生成词面表），
/// 逆向词面只登记查询换向（不改生成叙事词面，避免叙事错向）。
fn sync_predicate_pairs(reasoner: &mut SymbolicReasoner, oxigraph_manager: &OxigraphManager) {
    // 别名/换向词面与画像律典走同一注入入口（服务启动复用同一实现）
    reasoner.align_predicate_pairs(&oxigraph_manager.predicate_query_pairs());
    // 谓词画像的显式声明同步进律典：传递/对称/互逆律从此进入查询选路。
    reasoner.configure_laws(&oxigraph_manager.predicate_registry_snapshot());
}

/// 把实体身份注册表里已确认的持久别名同步进链接器（别名与规范名同权）。
fn sync_entity_aliases(reasoner: &mut SymbolicReasoner, oxigraph_manager: &OxigraphManager) {
    reasoner.align_entity_aliases(&oxigraph_manager.confirmed_entity_aliases());
}

/// 把全息命令参数切成若干词项：支持空白/中英文逗号/冒号分隔
fn parse_holo_terms(target: &str, expected: usize) -> Option<Vec<String>> {
    let terms: Vec<String> = target
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '，' | ',' | '：' | ':' | '、'))
        .filter(|term| !term.is_empty())
        .map(|term| term.to_string())
        .collect();
    if terms.len() == expected {
        Some(terms)
    } else {
        None
    }
}

/// 解析全息检索的两元组：支持「主体 谓词」与「主语的谓词」两种写法
fn parse_holo_pair(target: &str) -> Option<(String, String)> {
    if let Some(terms) = parse_holo_terms(target, 2) {
        return Some((terms[0].clone(), terms[1].clone()));
    }
    let text = target.trim();
    if let Some(index) = text.find('的') {
        let subject = text[..index].trim();
        let predicate = text[index + '的'.len_utf8()..].trim();
        if !subject.is_empty() && !predicate.is_empty() {
            return Some((subject.to_string(), predicate.to_string()));
        }
    }
    None
}

/// 渲染全息召回候选（软召回分数 + 硬核查标记）
fn render_holo_hits(title: &str, hits: &[HoloHit]) -> String {
    if hits.is_empty() {
        return format!(
            "{}：全息图未召回候选（实体/谓词未登记，或该结构在知识中不存在）。",
            title
        );
    }
    let verified = hits.iter().filter(|hit| hit.verified).count();
    let mut out = format!(
        "{}（候选 {} 个，通过三元组核查 {} 个）：",
        title,
        hits.len(),
        verified
    );
    for (index, hit) in hits.iter().enumerate() {
        let mark = if hit.verified {
            "已核查✓"
        } else {
            "未登记（串扰/推测，不入结论）"
        };
        out.push_str(&format!(
            "\n  {}. {}  余弦 {:.3}  [{}]",
            index + 1,
            hit.symbol,
            hit.score,
            mark
        ));
    }
    out
}

/// 渲染全息类比候选（关系迁移路径 + 核查标记）
fn render_holo_analogy(a: &str, b: &str, c: &str, hits: &[HoloAnalogyHit]) -> String {
    if hits.is_empty() {
        return format!(
            "全息类比 {}:{} :: {}:? 未产生候选（需先登记一组对应事实，关系才能被解缠迁移）。",
            a, b, c
        );
    }
    let mut out = format!(
        "全息类比 {}:{} :: {}:?（关系解缠 → 迁移绑定 → 净噪）：",
        a, b, c
    );
    for (index, hit) in hits.iter().enumerate() {
        let via: Vec<String> = hit
            .via
            .iter()
            .map(|(predicate, score)| format!("{}（{:.2}）", predicate, score))
            .collect();
        let mark = if hit.verified {
            "已核查✓"
        } else {
            "未登记（全息推测，供参考）"
        };
        out.push_str(&format!(
            "\n  {}. {}  综合分 {:.3}  关系通道：{}  [{}]",
            index + 1,
            hit.symbol,
            hit.score,
            via.join("、"),
            mark
        ));
    }
    out
}

/// 解析中文数词（1..=99）：十一、二十、二十三、九十九
fn parse_chinese_numeral(text: &str) -> Option<usize> {
    let digit = |ch: char| -> Option<usize> {
        Some(match ch {
            '一' => 1,
            '二' | '两' => 2,
            '三' => 3,
            '四' => 4,
            '五' => 5,
            '六' => 6,
            '七' => 7,
            '八' => 8,
            '九' => 9,
            _ => return None,
        })
    };
    let chars: Vec<char> = text.chars().collect();
    match chars.as_slice() {
        ['十'] => Some(10),
        [single] => digit(*single),
        ['十', ones] => Some(10 + digit(*ones)?),
        [tens, '十'] => Some(digit(*tens)? * 10),
        [tens, '十', ones] => Some(digit(*tens)? * 10 + digit(*ones)?),
        _ => None,
    }
}

/// 解析「采纳假设 N」中的编号：支持阿拉伯数字与「一、二、三…十、十二、二十三」
fn parse_hypothesis_index(text: &str) -> Option<usize> {
    let trimmed = text.trim().trim_start_matches(['：', ':', ' ', '第']);
    let trimmed = trimmed.trim_end_matches('条').trim();
    if let Ok(index) = trimmed.parse::<usize>() {
        return (index > 0).then_some(index);
    }
    let index = parse_chinese_numeral(trimmed)?;
    (index > 0).then_some(index)
}

/// 解析歧义义项选择输入（1 基编号）：返回候选表 0 基下标。
///
/// 空输入 / 非数字 / 越界一律返回 `None`（调用方回退权重最高项）。
fn parse_ambiguous_choice(text: &str, options: usize) -> Option<usize> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(index) if (1..=options).contains(&index) => Some(index - 1),
        _ => None,
    }
}

/// 歧义实体的人工选择：提示用户按编号选择义项。
///
/// 仅在交互模式调用；空输入 / 非法输入 / EOF 均回退到权重最高项（第 1 项），
/// 返回候选表下标。
fn prompt_ambiguous_selection(
    entity: &str,
    candidates: &[(String, u64)],
    max_options: usize,
) -> usize {
    let options = candidates.len().min(max_options.max(1));
    print!(
        "请选择「{}」的义项编号 [1-{}]（直接回车选权重最高项）: ",
        entity, options
    );
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() {
        return 0;
    }
    match parse_ambiguous_choice(&line, options) {
        Some(index) => index,
        None => {
            if !line.trim().is_empty() {
                println!("输入「{}」无效，使用权重最高项。", line.trim());
            }
            0
        }
    }
}

/// 对查询实体做歧义解析：自动模式结合整句上下文选择义项，
/// 人工模式在交互界面提示用户按编号选择。
///
/// 无候选或调用失败时保留原实体名，保证返回结果与输入实体一一对应。
async fn resolve_query_entities_with_disambiguation(
    input: &str,
    query_entities: &[String],
    api_config: &know_prolog::ApiConfig,
    adapter: &LocalSemanticAdapter,
) -> Vec<String> {
    println!("正在解析实体歧义...");
    let mut resolved_entities = Vec::new();
    let mut found_ambiguous = false;

    for entity in query_entities {
        match adapter.get_ambiguous_entities(entity).await {
            Ok(ambiguous_response) => {
                if ambiguous_response.data.len() > 1 {
                    found_ambiguous = true;
                    println!("发现实体 '{}' 有多种含义:", entity);
                    for (i, (entity_name, weight)) in ambiguous_response
                        .data
                        .iter()
                        .take(api_config.max_ambiguous_entities)
                        .enumerate()
                    {
                        println!("  {}. {} (权重: {})", i + 1, entity_name, weight);
                    }

                    let chosen_entity = if api_config.auto_select_best_entity {
                        let index =
                            select_candidate_by_context(&ambiguous_response.data, Some(input))
                                .unwrap_or(0);
                        let (best_entity, _) = &ambiguous_response.data[index];
                        println!("自动选择（结合上下文）: {}", best_entity);
                        best_entity.clone()
                    } else {
                        let index = prompt_ambiguous_selection(
                            entity,
                            &ambiguous_response.data,
                            api_config.max_ambiguous_entities,
                        );
                        let (chosen, _) = &ambiguous_response.data[index];
                        println!("已选择: {}", chosen);
                        chosen.clone()
                    };
                    resolved_entities.push(chosen_entity.clone());

                    match adapter.get_simplified_knowledge(&chosen_entity).await {
                        Ok(knowledge) => {
                            if let Some(desc) = knowledge.get("description") {
                                if !desc.is_empty() {
                                    println!("  描述: {}", desc);
                                }
                            }
                            if let Some(tags) = knowledge.get("tags") {
                                if !tags.is_empty() {
                                    println!("  标签: {}", tags);
                                }
                            }
                        }
                        Err(e) => eprintln!("获取实体知识时出错: {}", e),
                    }
                } else if ambiguous_response.data.len() == 1 {
                    if let Some((entity_name, _)) = ambiguous_response.data.first() {
                        resolved_entities.push(entity_name.clone());
                    }
                } else {
                    resolved_entities.push(entity.clone());
                }
            }
            Err(e) => {
                eprintln!("查询实体 '{}' 的歧义信息时出错: {}", entity, e);
                resolved_entities.push(entity.clone());
            }
        }
    }

    if found_ambiguous {
        println!("─────────────────────────────────");
    }
    resolved_entities
}

async fn continue_original_query(
    config: &know_prolog::SystemConfig,
    oxigraph_manager: &OxigraphManager,
    prolog_engine: &mut PrologEngine,
    response_formatter: &ResponseFormatter,
    local_semantic_adapter: &LocalSemanticAdapter,
    external_knowledge_service: &ExternalKnowledgeService,
    symbolic_reasoner: &mut SymbolicReasoner,
    contrast_arbiter: &mut ContrastArbiter,
    plugin_runtime: &mut know_prolog::plugin::PluginRuntime,
    input: &str,
    query_type: &str,
    entities: &[String],
    precomputed_query_entities: &[String],
) {
    println!("查询类型: {}, 实体: {:?}", query_type, entities);

    // 提取查询中的实体
    let mut seen_entities = std::collections::HashSet::new();
    let mut query_entities: Vec<String> =
        Vec::with_capacity(precomputed_query_entities.len() + entities.len());
    for entity in precomputed_query_entities.iter().chain(entities.iter()) {
        if seen_entities.insert(entity.as_str()) {
            query_entities.push(entity.clone());
        }
    }

    // 使用歧义解析功能（如果启用）
    if let Some(api_config) = &config.api {
        if api_config.enable_entity_disambiguation {
            let resolved_entities = resolve_query_entities_with_disambiguation(
                input,
                &query_entities,
                api_config,
                local_semantic_adapter,
            )
            .await;

            let final_entities = if !resolved_entities.is_empty() {
                resolved_entities
            } else {
                entities.to_vec()
            };

            let max_hops = 2;
            match oxigraph_manager
                .get_relevant_facts(&final_entities, max_hops)
                .await
            {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);

                    if let Some(answer) = oxigraph_manager
                        .get_direct_answer(query_type, &final_entities)
                        .await
                    {
                        println!("{}", answer);
                    } else if let Some(prolog_query) =
                        PrologEngine::generate_prolog_query(query_type, &final_entities)
                    {
                        println!("Prolog查询: {}", prolog_query);

                        let response = run_prolog_query(
                            prolog_engine,
                            symbolic_reasoner,
                            &prolog_query,
                            query_type,
                            response_formatter,
                        );
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                    if let Some(answer) = try_symbolic_reasoning(
                        symbolic_reasoner,
                        contrast_arbiter,
                        oxigraph_manager,
                        input,
                        precomputed_query_entities,
                    )
                    .await
                    {
                        println!("{}", answer);
                    }
                }
            }
        } else {
            // 歧义解析功能被禁用
            let max_hops = 2;
            match oxigraph_manager
                .get_relevant_facts(&query_entities, max_hops)
                .await
            {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);

                    if let Some(answer) = oxigraph_manager
                        .get_direct_answer(query_type, entities)
                        .await
                    {
                        println!("{}", answer);
                    } else if let Some(prolog_query) =
                        PrologEngine::generate_prolog_query(query_type, entities)
                    {
                        println!("Prolog查询: {}", prolog_query);

                        let response = run_prolog_query(
                            prolog_engine,
                            symbolic_reasoner,
                            &prolog_query,
                            query_type,
                            response_formatter,
                        );
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                    if let Some(answer) = try_symbolic_reasoning(
                        symbolic_reasoner,
                        contrast_arbiter,
                        oxigraph_manager,
                        input,
                        precomputed_query_entities,
                    )
                    .await
                    {
                        println!("{}", answer);
                    }
                }
            }
        }
    } else {
        // 没有API配置
        let max_hops = 2;
        match oxigraph_manager
            .get_relevant_facts(&query_entities, max_hops)
            .await
        {
            Ok(facts) => {
                println!("加载了 {} 个相关事实", facts.len());
                prolog_engine.load_relevant_facts(facts);

                if let Some(answer) = oxigraph_manager
                    .get_direct_answer(query_type, entities)
                    .await
                {
                    println!("{}", answer);
                } else if let Some(prolog_query) =
                    PrologEngine::generate_prolog_query(query_type, entities)
                {
                    println!("Prolog查询: {}", prolog_query);

                    let response = run_prolog_query(
                        prolog_engine,
                        symbolic_reasoner,
                        &prolog_query,
                        query_type,
                        response_formatter,
                    );
                    println!("{}", response);
                } else {
                    println!("无法生成有效的Prolog查询");
                }
            }
            Err(e) => {
                println!("{}: {}", config.messages.process_error, e);

                if let Some(answer) = try_symbolic_reasoning(
                    symbolic_reasoner,
                    contrast_arbiter,
                    oxigraph_manager,
                    input,
                    precomputed_query_entities,
                )
                .await
                {
                    println!("{}", answer);
                } else {
                    // 本地查询失败时，先给插件声明式兜底一次机会，再尝试外部知识API
                    println!("本地查询失败，尝试外部知识源...");

                    if let Some((text, _plan)) = plugin_policy_text(plugin_runtime, input) {
                        println!("{}", text);
                    } else if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                    } else {
                        match external_knowledge_service
                            .answer_question_with_entities(input, &query_entities)
                            .await
                        {
                            Ok(outcome) => match external_outcome_text(&outcome) {
                                Some(text) => println!("{}", text),
                                None => println!("很抱歉，我无法找到相关信息。"),
                            },
                            Err(e) => {
                                println!("查询外部知识时出错：{}。", e);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn store_override_isolates_relative_dialogue_state() {
        let mut path = Some("oxigraph_data/dialogue_state.json".to_string());
        isolate_dialogue_state_path(&mut path, "/tmp/kp-alt");
        assert_eq!(
            path.as_deref(),
            Some("/tmp/kp-alt/dialogue_state.json"),
            "相对状态路径必须跟随 --store"
        );

        let mut absolute = Some("/var/lib/kp/state.json".to_string());
        isolate_dialogue_state_path(&mut absolute, "/tmp/kp-alt");
        assert_eq!(
            absolute.as_deref(),
            Some("/var/lib/kp/state.json"),
            "绝对路径是用户显式指定，不得改写"
        );

        let mut missing = None;
        isolate_dialogue_state_path(&mut missing, "/tmp/kp-alt");
        assert_eq!(missing, None);
    }

    #[test]
    fn ask_cli_accepts_question_and_stdin_modes() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "ask",
            "张三的师傅是谁",
            "--json",
            "--strict",
        ])
        .expect("ask 参数应可解析");
        match cli.command {
            Some(Commands::Ask {
                question,
                json,
                strict,
            }) => {
                assert_eq!(question.as_deref(), Some("张三的师傅是谁"));
                assert!(json);
                assert!(strict);
            }
            _ => panic!("应解析为 Ask 子命令"),
        }
        let cli = Cli::try_parse_from(["knowprolog", "ask"]).expect("无问句时应读 stdin");
        assert!(matches!(
            cli.command,
            Some(Commands::Ask {
                question: None,
                json: false,
                strict: false
            })
        ));
        for terminator in ["quit", "exit", "退出", "q", " 退出 "] {
            assert!(is_ask_terminator(terminator), "{}", terminator);
        }
        assert!(!is_ask_terminator("为什么退出假设"));
    }

    #[test]
    fn config_override_is_global_and_defaults_to_config_toml() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "--config",
            "conf/alt.toml",
            "ask",
            "张三的师傅是谁",
        ])
        .expect("全局 --config 应可解析");
        assert_eq!(cli.config, "conf/alt.toml");
        let default = Cli::try_parse_from(["knowprolog", "ask", "问题"]).unwrap();
        assert_eq!(default.config, "config.toml");
    }

    /// 各命令必须被 is_command_like 识别，避免澄清追问时被吞掉
    #[test]
    fn command_like_covers_journal_and_perf_commands() {
        for command in [
            "撤销导入",
            "导入回执",
            "导入历史",
            "导入日志",
            "性能报告",
            "性能台账",
            "配置契约",
            "配置体检",
            "派生台账",
            "增量报告",
            "图账本",
            "规则账本",
            "规则图账本",
            "派生图账本",
            "物化推理",
            "推理物化",
            "物化派生",
            "忘记规则：如果X的师傅是Y，那么X的老师是Y",
            "撤回规则：如果X的师傅是Y，那么X的老师是Y",
            "虚词表",
            "功能词表",
            "虚词报告",
            "算符表",
            "提及账本",
            "外链台账",
            "情景",
            "对话情景",
            "情景理论",
            "情景：12",
            "情景历史：姚明的身高是多少",
            "情景归因：姚明的身高是多少",
            "情景反事实：姚明的身高是多少",
            "情景查询：answered_at(S,Q,A,P)",
            "外链查看：法国",
            "外链刷新：法国",
            "外链删除：法国",
            "同义链接：姚明 = ownthink:姚明",
            "帧记忆",
            "话语帧",
            "挂账",
            "追踪报告",
            "追踪状态",
            "追踪图",
            "指代轨迹",
            "问据栈",
            "追踪为什么 4",
            "查询剧本：examples/query_board_scenario.toml",
            "运行剧本：examples/query_board_scenario.toml",
            "默认推理：麻雀会飞",
            "默认逻辑：",
            "对齐候选：老婆",
            "补全回合：张三",
            "确认补全：1",
            "否定补全：1",
            "补全提问",
            "补全测试",
            "补全重放",
            "别问了",
            "允许提问",
            "晋升补全规则：rule-abc|全局",
            "退役补全规则：rule-abc",
            "恢复补全规则：rule-abc",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    /// 各命令（谓词裁决 / 澄清子命令 / 帮助退出等）都要被 is_command_like 识别
    #[test]
    fn command_like_covers_help_predicate_and_clarification_commands() {
        for command in [
            "帮助",
            "help",
            "?",
            "退出",
            "exit",
            "谓词报告",
            "谓词画像",
            "谓词注册表",
            "谓词列表",
            "登记别名：师父 = 师傅",
            "登记逆谓词：师傅 = 徒弟",
            "功能谓词：出生地",
            "取消功能谓词：出生地",
            "确认谓词：师傅",
            "拒绝谓词：师傅",
            "待定推测",
            "后向链规则",
            "开始澄清",
            "停止澄清",
            "结束澄清",
            "退出澄清",
            "提问确认",
            "接着说",
            "继续说",
            "继续介绍",
            "继续讲",
            "接着讲",
            "换个角度说",
            "换个说法",
            "换个角度",
            "换一个角度",
            "重新介绍",
            "撤销上次导入",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    /// 导入指令解析：全角/半角冒号、README 简写、参数正文零替换
    #[test]
    fn import_directive_parses_colons_aliases_and_keeps_body() {
        assert_eq!(
            parse_import_directive("导入jsonld文件：test_data.jsonld"),
            Some(ImportDirective::JsonldFile("test_data.jsonld".to_string()))
        );
        assert_eq!(
            parse_import_directive("导入jsonld文件: test_data.jsonld"),
            Some(ImportDirective::JsonldFile("test_data.jsonld".to_string()))
        );
        // README 中的简写
        assert_eq!(
            parse_import_directive("导入文件：examples/master_disciple.jsonld"),
            Some(ImportDirective::JsonldFile(
                "examples/master_disciple.jsonld".to_string()
            ))
        );
        assert_eq!(
            parse_import_directive("导入内容：{\"@id\": \"http://example.org/kg/entity/王五\"}"),
            Some(ImportDirective::JsonldContent(
                "{\"@id\": \"http://example.org/kg/entity/王五\"}".to_string()
            ))
        );
        assert_eq!(
            parse_import_directive("import jsonld content: {\"@id\": \"x\"}"),
            Some(ImportDirective::JsonldContent(
                "{\"@id\": \"x\"}".to_string()
            ))
        );
        assert_eq!(
            parse_import_directive("导入rdf文件：data/sample.ttl"),
            Some(ImportDirective::RdfFile("data/sample.ttl".to_string()))
        );
        assert_eq!(
            parse_import_directive("import rdf: data/sample.nt"),
            Some(ImportDirective::RdfFile("data/sample.nt".to_string()))
        );

        // 参数正文里再出现关键词不得被删改
        assert_eq!(
            parse_import_directive("导入文件：导入文件：备份.jsonld"),
            Some(ImportDirective::JsonldFile(
                "导入文件：备份.jsonld".to_string()
            ))
        );

        // 非导入指令（自然语言条目）不误判
        assert_eq!(parse_import_directive("录入：小明18岁"), None);
        assert_eq!(parse_import_directive("导入jsonld文件"), None);
    }

    /// 帮助页覆盖关键入口（录入 / 导入 / 查询 / 退出），避免手册与实现漂移
    #[test]
    fn interactive_help_mentions_key_commands() {
        let help = interactive_help();
        for needle in [
            "录入：",
            "导入jsonld文件：",
            "导入rdf文件：",
            "导出rdf文件：",
            "SPARQL：",
            "为什么",
            "证明：",
            "假设世界",
            "退出假设",
            "追踪报告",
            "改口：",
            "槽账本",
            "帮助",
            "quit",
        ] {
            assert!(help.contains(needle), "帮助页应包含 {}", needle);
        }
    }

    /// SPARQL 指令解析：全角/半角冒号、大小写、空格形式与正文保真
    #[test]
    fn sparql_command_parses_colons_case_and_keeps_body() {
        let select = "SELECT ?s WHERE { ?s ?p ?o }";
        assert_eq!(
            parse_sparql_command(&format!("SPARQL：{}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("sparql:{}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("Sparql {}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("sparql {}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("查询SPARQL：{}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("查询SPARQL:{}", select)),
            Some(select)
        );
        // 半角冒号的短前缀不得按全角长度切分而吃掉正文（含多字节字符）
        assert_eq!(
            parse_sparql_command("查询SPARQL:SELECT ?s WHERE { ?s <http://a/中> }"),
            Some("SELECT ?s WHERE { ?s <http://a/中> }")
        );
        assert_eq!(parse_sparql_command("sparql："), Some(""));
        assert_eq!(parse_sparql_command("小明多大？"), None);
        // 正文里的冒号与关键词不被删改
        assert_eq!(
            parse_sparql_command("SPARQL：SELECT ?s WHERE { ?s <http://a/b> \"x:y\" }"),
            Some("SELECT ?s WHERE { ?s <http://a/b> \"x:y\" }")
        );
        assert!(is_command_like("SPARQL：SELECT ?s WHERE { ?s ?p ?o }"));
        assert!(is_command_like("sparql SELECT ?s WHERE { ?s ?p ?o }"));
    }

    /// 导出指令解析：全角/半角冒号、简写与路径保真；非指令不误判
    #[test]
    fn export_directive_parses_colons_aliases_and_keeps_path() {
        assert_eq!(
            parse_export_directive("导出rdf文件：out/data.ttl"),
            Some(ExportDirective::RdfFile("out/data.ttl".to_string()))
        );
        assert_eq!(
            parse_export_directive("导出文件: backup.jsonld"),
            Some(ExportDirective::RdfFile("backup.jsonld".to_string()))
        );
        assert_eq!(
            parse_export_directive("export rdf: dump.nt"),
            Some(ExportDirective::RdfFile("dump.nt".to_string()))
        );
        assert_eq!(
            parse_export_directive("导出rdf：导出文件：怪名.ttl"),
            Some(ExportDirective::RdfFile("导出文件：怪名.ttl".to_string()))
        );
        assert_eq!(parse_export_directive("导出rdf文件"), None);
        assert_eq!(parse_export_directive("录入：小明18岁"), None);
        assert!(is_command_like("导出文件：dump.ttl"));
    }

    /// 「同义链接」命令的解析：半角/全角等号、缺端拒绝
    #[test]
    fn alias_link_parses_both_equals_forms() {
        assert_eq!(
            parse_alias_link("姚明 = ownthink:姚明"),
            Some(("姚明", "ownthink:姚明"))
        );
        assert_eq!(
            parse_alias_link("姚明＝ownthink:姚明"),
            Some(("姚明", "ownthink:姚明"))
        );
        assert_eq!(parse_alias_link("只有一头 = "), None);
        assert_eq!(parse_alias_link("没有等号"), None);
    }

    /// 外链候选实体：预计算在前、去重保序
    #[test]
    fn external_candidates_merge_dedup_and_order() {
        let entities = vec!["B".to_string(), "D".to_string()];
        let precomputed = vec!["A".to_string(), "B".to_string()];
        assert_eq!(
            external_candidates(&entities, &precomputed),
            vec!["A", "B", "D"]
        );
    }

    /// 跨重启撤销：导入 → 日志落盘 → 重新打开存储 → 恢复日志 → 撤销归零
    #[tokio::test]
    async fn journal_survives_restart_and_undoes_import() {
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().to_str().unwrap().to_string();
        let journal_path = format!("{}/import_journal.json", store_path);

        {
            let manager =
                OxigraphManager::with_perf_ledger(&store_path, PerfLedger::shared()).unwrap();
            let items = vec![KnowledgeItem::Relationship {
                from_label: "人物".to_string(),
                from_id: "张三".to_string(),
                to_label: "人物".to_string(),
                to_id: "李四".to_string(),
                rel_type: "师傅".to_string(),
                properties: HashMap::new(),
            }];
            let receipt = manager.save_knowledge_with_receipt(items).await.unwrap();
            assert!(receipt.can_undo());

            let mut journal = ImportJournal::new();
            journal.push(receipt.clone());
            journal.save(&journal_path).unwrap();
            let snapshot = manager.load_graph_snapshot().await.unwrap();
            assert!(!snapshot.is_empty());
            receipt
        };

        // 模拟进程重启：新开存储 + 从信封恢复日志
        let manager = OxigraphManager::with_perf_ledger(&store_path, PerfLedger::shared()).unwrap();
        let JournalLoad::Loaded(mut journal) = ImportJournal::load(&journal_path) else {
            panic!("应能从 ContextImage 信封恢复导入日志");
        };
        let restored = journal.pop().expect("栈顶应有一笔可撤销导入");
        let removed = manager.undo_import(&restored).await.unwrap();
        assert_eq!(removed, restored.actions.len());
        assert_eq!(journal.len(), 0);
        let snapshot = manager.load_graph_snapshot().await.unwrap();
        assert!(
            snapshot.is_empty(),
            "撤销后图谱应回到导入前状态，实际剩余 {} 条",
            snapshot.len()
        );
    }

    /// 服务会话的知识装载必须含关系事实：只有实体会让关系问法静默落空。
    #[tokio::test]
    async fn serve_facts_include_relations_alongside_entities() {
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().to_str().unwrap().to_string();
        let manager = OxigraphManager::with_perf_ledger(&store_path, PerfLedger::shared()).unwrap();
        let items = vec![
            KnowledgeItem::Entity {
                label: "人物".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "张三".to_string()),
                    ("age".to_string(), "25".to_string()),
                ]),
            },
            KnowledgeItem::Relationship {
                from_label: "人物".to_string(),
                from_id: "张三".to_string(),
                to_label: "人物".to_string(),
                to_id: "李四".to_string(),
                rel_type: "师傅".to_string(),
                properties: HashMap::new(),
            },
        ];
        manager.save_knowledge(items).await.unwrap();

        let facts = serve_facts(&manager).await.unwrap();
        assert!(
            facts
                .iter()
                .any(|fact| matches!(fact, PrologFact::Entity { .. })),
            "实体事实应保留：{:?}",
            facts
        );
        assert!(
            facts.iter().any(
                |fact| matches!(fact, PrologFact::Relation { rel_type, .. } if rel_type == "师傅")
            ),
            "关系事实必须随服务启动一并装载：{:?}",
            facts
        );
    }

    /// 帧澄清粘合层：命令/无关输入放行且零状态变化；序号绑定后返回重放问句
    #[test]
    fn frame_clarification_step_binds_replays_and_passes_through() {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226cm".to_string());
        yao.insert("体重".to_string(), "140kg".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190cm".to_string());
        ye.insert("体重".to_string(), "85kg".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人".to_string(),
                properties: ye,
            },
        ];
        let mut reasoner =
            SymbolicReasoner::new(know_prolog::ReasoningConfig::default(), HashMap::new());
        reasoner.observe_facts(&facts);

        // 无源可填的洞 → 澄清（候选人名序：叶莉、姚明）
        reasoner.contextualize("体重是多少");
        let prompt = reasoner
            .start_frame_clarification("体重是多少")
            .expect("应开启帧澄清");
        assert!(
            prompt.contains("叶莉") && prompt.contains("姚明"),
            "{}",
            prompt
        );

        // 命令让位：放行给常规命令通道，澄清保持
        assert!(matches!(
            handle_frame_clarification(&mut reasoner, "帧记忆"),
            FrameClarificationStep::Pass
        ));
        // 非回答自然语言：放行，澄清保持
        assert!(matches!(
            handle_frame_clarification(&mut reasoner, "北京是哪个省的"),
            FrameClarificationStep::Pass
        ));
        assert!(reasoner.frame_clarification_active());

        // 序号短答：确定性绑定 → 返回重放问句（将由主循环走常规问答）
        match handle_frame_clarification(&mut reasoner, "1") {
            FrameClarificationStep::Replay(question) => {
                assert_eq!(question, "叶莉的体重是什么");
            }
            _ => panic!("序号应绑定并重放"),
        }
        assert!(!reasoner.frame_clarification_active(), "绑定后澄清应关闭");
        let answer = reasoner.answer("叶莉的体重是什么").expect("重放应可作答");
        assert!(answer.contains("85kg"), "{}", answer);
    }

    /// 歧义义项选择解析：合法编号映射为 0 基下标；空/非数字/越界回退 None
    #[test]
    fn ambiguous_choice_parses_valid_index_and_rejects_invalid() {
        assert_eq!(parse_ambiguous_choice("1", 5), Some(0));
        assert_eq!(parse_ambiguous_choice(" 3 ", 5), Some(2));
        assert_eq!(parse_ambiguous_choice("5", 5), Some(4));
        assert_eq!(parse_ambiguous_choice("0", 5), None, "编号从 1 开始");
        assert_eq!(parse_ambiguous_choice("6", 5), None, "超出候选数量");
        assert_eq!(parse_ambiguous_choice("", 5), None, "空输入回退");
        assert_eq!(parse_ambiguous_choice("苹果", 5), None, "非数字回退");
        assert_eq!(parse_ambiguous_choice("2", 1), None, "单个候选时越界");
    }

    #[test]
    fn correction_spec_accepts_pipe_and_chinese_forms() {
        assert_eq!(
            parse_correction_spec("姚明|height|226|229"),
            Some((
                "姚明".to_string(),
                "height".to_string(),
                "226".to_string(),
                "229".to_string()
            ))
        );
        assert_eq!(
            parse_correction_spec("姚明的身高从226改为229"),
            Some((
                "姚明".to_string(),
                "身高".to_string(),
                "226".to_string(),
                "229".to_string()
            ))
        );
        assert_eq!(
            parse_correction_spec("把姚明的身高由 226 改成 229。"),
            Some((
                "姚明".to_string(),
                "身高".to_string(),
                "226".to_string(),
                "229".to_string()
            ))
        );
        assert_eq!(parse_correction_spec("姚明的身高是229"), None);
        assert_eq!(parse_correction_spec("姚明|height|226"), None);
    }

    #[test]
    fn paths_cli_supports_between_chain_and_common_modes() {
        let between = Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--to",
            "王五",
            "--max-hops",
            "3",
        ])
        .expect("两点路径参数应可解析");
        match between.command {
            Some(Commands::Paths {
                to,
                via,
                common_with,
                max_hops,
                ..
            }) => {
                assert_eq!(to.as_deref(), Some("王五"));
                assert!(via.is_empty());
                assert!(common_with.is_none());
                assert_eq!(max_hops, 3);
            }
            _ => panic!("应解析为 Paths 子命令"),
        }

        let chain = Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--via",
            "师傅",
            "--via",
            "师傅",
        ])
        .expect("谓词链参数应可解析");
        match chain.command {
            Some(Commands::Paths { to, via, .. }) => {
                assert!(to.is_none());
                assert_eq!(via, vec!["师傅".to_string(), "师傅".to_string()]);
            }
            _ => panic!("应解析为 Paths 子命令"),
        }

        let common = Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--common-with",
            "李四",
        ])
        .expect("共同邻居参数应可解析");
        match common.command {
            Some(Commands::Paths { common_with, .. }) => {
                assert_eq!(common_with.as_deref(), Some("李四"));
            }
            _ => panic!("应解析为 Paths 子命令"),
        }

        assert!(Cli::try_parse_from(["knowprolog", "paths", "--from", "张三"]).is_err());
        assert!(Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--to",
            "王五",
            "--via",
            "师傅",
        ])
        .is_err());
    }

    #[test]
    fn forge_cli_parses_repeated_examples_and_activation() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "forge",
            "--name",
            "女儿问法",
            "--route",
            "daughter_query",
            "--positive",
            "姚明的女儿是谁",
            "--positive",
            "姚明的闺女是谁",
            "--negative",
            "姚明今年几岁",
            "--priority",
            "30",
            "--activate",
            "--json",
        ])
        .expect("锻造参数应可解析");
        match cli.command {
            Some(Commands::Forge {
                name,
                route,
                positives,
                negatives,
                priority,
                activate,
                json,
            }) => {
                assert_eq!(name, "女儿问法");
                assert_eq!(route, "daughter_query");
                assert_eq!(positives.len(), 2);
                assert_eq!(negatives, vec!["姚明今年几岁".to_string()]);
                assert_eq!(priority, 30);
                assert!(activate);
                assert!(json);
            }
            _ => panic!("应解析为 Forge 子命令"),
        }
        assert!(Cli::try_parse_from(["knowprolog", "forge", "--name", "空"]).is_err());
    }

    #[test]
    fn build_kg_cli_parses_feedback_and_apply() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "build-kg",
            "--text",
            "张三拜李四为师。",
            "--min-support",
            "3",
            "--min-purity",
            "750",
            "--limit",
            "12",
            "--feedback",
            "feedback.txt",
            "--apply",
            "--json",
        ])
        .expect("build-kg 参数应可解析");
        match cli.command {
            Some(Commands::BuildKg {
                text,
                min_support,
                min_purity,
                limit,
                feedback,
                apply,
                json,
                ..
            }) => {
                assert_eq!(text.as_deref(), Some("张三拜李四为师。"));
                assert_eq!(min_support, 3);
                assert_eq!(min_purity, 750);
                assert_eq!(limit, 12);
                assert_eq!(feedback.as_deref(), Some("feedback.txt"));
                assert!(apply);
                assert!(json);
            }
            _ => panic!("应解析为 BuildKg 子命令"),
        }
        assert!(Cli::try_parse_from([
            "knowprolog",
            "build-kg",
            "--text",
            "甲",
            "--file",
            "corpus.txt",
        ])
        .is_err());
    }

    #[test]
    fn command_like_covers_trust_and_correction_commands() {
        for command in [
            "更正：姚明的身高从226改为229",
            "纠正：姚明|height|226|229",
            "修正：姚明的身高从226改为229",
            "历史可信度",
            "历史可信度：姚明的身高是多少",
            "可信度：姚明的身高是多少",
            "信任报告",
            "改口：姚明的身高是229cm",
            "改口史：姚明 身高",
            "槽账本",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    #[test]
    fn recovery_confirmation_bridges_into_reasoner_premises() {
        use know_prolog::recovery::{ConfirmationSource, RecoveryState};

        let mut state = RecoveryState::new();
        state.record_confirmed(
            "王五",
            "师傅",
            "赵六",
            "receipt-1",
            ConfirmationSource::Provisional,
        );
        let mut manager = DialogueManager::new();
        let mut snapshot = manager.snapshot();
        snapshot.recovery = state;
        manager.restore(&snapshot);

        let mut reasoner =
            SymbolicReasoner::new(know_prolog::ReasoningConfig::default(), HashMap::new());
        assert_eq!(
            sync_recovery_confirmed_facts(&manager, &mut reasoner),
            1,
            "首次同步应写入一条会话事实"
        );
        assert!(reasoner
            .session_facts()
            .iter()
            .any(|atom| atom.display() == "师傅(王五, 赵六)"));
        assert_eq!(
            sync_recovery_confirmed_facts(&manager, &mut reasoner),
            0,
            "重复同步必须幂等"
        );
        let proof = reasoner.resolution_proof_report("王五的师傅是赵六");
        assert!(!proof.contains("未得证"), "{}", proof);
    }

    /// 统一模式库命令：库视图 / 匹配解释 / 教学模式 / 忘模式都要被识别为命令。
    #[test]
    fn command_like_covers_pattern_library_commands() {
        for command in [
            "模式路由",
            "查询模式库",
            "统一模式库",
            "模式关系",
            "模式关系图",
            "模式落图",
            "同步模式图",
            "模式匹配：法国的首都是哪里",
            "模式经验",
            "经验台账",
            "模式联想：德国的首都在哪里",
            "确认联想：德国的首都在哪里",
            "采纳联想：德国的首都在哪里",
            "忘经验：a1b2c3d4",
            "遗忘经验：法国首都查询",
            "清空模式经验",
            "教模式：老家 | {人} 的 老家 | hometown_query | 20 | 焦点",
            "忘模式：老家",
            "锻造模式：女儿问法 | 姚明的女儿是谁 | 姚明今年几岁 | daughter_query",
            "锻造模式",
            "确认锻造 forge_abc123",
            "否认锻造 abc123",
            "退役锻造 abc123",
            "遗忘锻造 abc123",
            "锻造台账",
            "锻造探针",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }
}
