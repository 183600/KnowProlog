use clap::{Parser, Subcommand, ValueEnum};
use std::collections::BTreeSet;
use std::io::{BufRead, Read, Write};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

// 从库（know_prolog）中导入所需的功能
use know_prolog::{
    answer_plan::StyleDirective,
    batch_teaching::{
        self, append_learned_items, BatchOptions, BatchTeacher, BatchTeachingLedger,
        LedgerLoad as BatchLedgerLoad,
    },
    binding_frames::FrameProgram,
    binding_table::{from_fact_refs, Coverage, TableSource},
    chronicle::{Chronicle, EntityRef, FsyncPolicy},
    config_contract,
    contextual_reference::{
        parse_clarify_act, rewrite_at, BindingLedger, BindingScope, ClarifyAct, ClarifyRequest,
        ContextualResolver,
    },
    contextual_segmentation::{ContextualSegmenter, ReadingKind, ReadingMemory, SegmentOptions},
    conversation_theory::{self, ConversationTheory},
    coref_closure::{self, CorefEvidence},
    create_default_synonyms,
    dense_retrieval::{DenseRetriever, SearchBudget, SearchConstraints},
    derivation::Derivation,
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
    dialogue_uri::{DialogueArtifactIndex, DialogueSession},
    disclosure_fusion::{
        self as disclosure_fusion, AnswerDisclosureConfig, ClaimAtom, FusedClaim,
    },
    embedding_eval::{
        compare_baseline, entity_types_from_prolog_facts, evaluate, triples_from_prolog_facts,
        EvalBaseline, EvalConfig,
    },
    entity_registry::{extract_alias_candidates, EntityRegistry, MintVerdict, NominationKind},
    entity_thread::{render_event_line, EntityThreadDigest, ThreadLookup},
    event_graph::{EventDialogue, EventGraph},
    external_knowledge::{get_direct_answer, AnswerOutcome, ExternalKnowledgeService},
    external_ledger::{ExternalLedger, LedgerLoad},
    extract_query_entities_from_tagged, frontier, fulfillment,
    fusion_arbitration::{arbitrate, ArbitrationInput},
    grammar_extractor::GrammarExtractor,
    history_trust::TrustAssessment,
    holographic_memory::{HoloAnalogyHit, HoloHit},
    identity_eval::{
        compare_baseline as compare_identity_baseline, evaluate_seed as evaluate_identity,
        IdentityEvalBaseline,
    },
    import_frames::{FrameLoad as ImportFrameLoad, ImportFrameLedger},
    import_journal::{ImportJournal, JournalLoad},
    joint_parse::verify_frames,
    knowledge_fusion::{
        Adjudication, FusionPolicy, NamespaceKind, NamespaceRecord, NamespaceStatus, TrustLevel,
    },
    lang_runtime::{self, LangReloadOutcome, LangRuntime},
    law_book::LawBook,
    lex_gate, lexical_memory, lexical_trace,
    link_prediction::{LinkPredictor, Triple as LawTriple},
    load_config_from,
    local_semantic_adapter::LocalSemanticAdapter,
    neuro_symbolic_contrast::ContrastArbiter,
    nlp::{
        normalize_words, normalize_words_sense_aware_with, preprocess, preprocess_annotated,
        TaggedWord,
    },
    nlu::NLUEngine,
    ontology_reasoning::{parse_ontology_lesson, OntologyProgram},
    open_closure::{self, OpenClosureDialogue, OpenClosureSession, OpenClosureTurn},
    owl_annotation::{
        self, AnnotationLedger, GraphObservation, LedgerLoad as OwlLedgerLoad, ProposalOptions,
        ReviewAction,
    },
    ownthink_api::select_candidate_by_context,
    path_memory::{EdgeSource, PathEdge, PathPredicate, TraversalGraph, TraversalOptions},
    pattern_closure::{ClosureBudget, ClosureOp},
    pattern_evolution::{self, EvolutionClock, EvolutionLedger},
    pattern_forge::{self, ForgeRequest, PatternForge},
    pattern_ir::{self, PatternLibrary},
    perf_ledger::PerfLedger,
    plugin::{PluginRegistry, PluginRuntime, PluginSessionState, PluginSettings},
    privacy_flow::{
        self, ConsentScope, DisclosurePrecision, DisclosureRequest, PolicyMode, PrivacyAct,
        PrivacyAdjudicator, PrivacyCategory, PrivacyGraph, PrivacyLedger, PrivacyPolicy,
        SpanDetector,
    },
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
    reference_probe::{self, GroundingLedger, ProbeDecision, ProbeStatus},
    relation_acquisition::{AcquisitionSource, GoalOpen, RelationAcquisition},
    relation_frame::{
        diagnose as diagnose_frames, narrate_frame, FrameExtractor, FrameGate, RelationFrame,
        FRAME_DIALECT,
    },
    release::{self, CheckOptions},
    resolution_engine::SearchStatus,
    rewrite_book::{self, RewriteBook},
    rewrite_retrieval,
    schema_view::SchemaView,
    search_port::{SearchMode, SearchRequest},
    shape_dialogue::ShapeDialogue,
    shape_eval::{
        compare_baseline as compare_shape_eval_baseline, evaluate_seed as evaluate_shape_dialogue,
        ShapeEvalBaseline,
    },
    shape_gate::{RepairAction, ShapeReport, ShapeSet},
    startup_ledger::StartupLedger,
    symbolic_generator::SymbolicGenerator,
    symbolic_reasoner::{FormalClarificationOutcome, LanguagePolicySync, SymbolicReasoner},
    synonym_graph::{SynonymContext, SynonymGraph, SynonymKind, SynonymResolution, SynonymSense},
    variant_closure::{VariantClosure, VariantContext},
    teachable_assertion::{TarResponse, TeachableAssertionReader},
    temporal::{self, TemporalConstraint, TemporalIndex, UnknownTimePolicy, Validity},
    trace_graph::{ReferentTrail, TraceMode},
    type_system::TypeSystem,
    view_render::{
        render_coverage_report, render_proof_ascii, render_proof_mermaid, render_timeline_flow,
        render_timeline_markdown, CoverageInput, KnowledgeSubgraph, TimelineTurn,
    },
    viewpoint,
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
    /// Import JSON-LD knowledge base from file or stdin
    ImportJsonld {
        /// Path to the JSON-LD file; "-" reads from stdin
        file_path: String,
    },
    /// Import an RDF file or stdin (Turtle / N-Triples / N-Quads / TriG /
    /// RDF/XML; format inferred from the extension, or set with --format)
    ImportRdf {
        /// Path to the RDF file (.ttl / .nt / .nq / .trig / .rdf / .xml);
        /// "-" reads from stdin
        file_path: String,
        /// Explicit RDF format (ttl / nt / nq / trig / rdf / xml); required
        /// when reading from stdin
        #[arg(long)]
        format: Option<String>,
    },
    /// Inspect RDF import episode memory: list batches, or show a brief for
    /// one batch (这批 / 上一批 / 第 N 批 / source file / batch id)
    Episodes {
        /// Batch reference; omit to list all recorded batches
        reference: Option<String>,
        /// Print JSON instead of the Chinese brief
        #[arg(long)]
        json: bool,
        /// Export the referenced batch to an RDF file (.ttl / .nt / .nq /
        /// .trig / .rdf / .xml / .jsonld); requires a batch reference
        #[arg(long)]
        export: Option<String>,
    },
    /// Retract a recorded RDF import batch: remove its first-time additions
    /// from the default graph, then drop the batch graph and episode metadata
    RetractBatch {
        /// Batch reference (这批 / 上一批 / 第 N 批 / source file / batch id)
        reference: String,
    },
    /// Batch-import natural-language sentences with the explicitly adopted
    /// batch-teaching rules (base extraction rules still run first)
    BatchImport {
        /// Corpus file (one or more sentences per line)
        file_path: String,
        /// Extract and report without writing to the knowledge store
        #[arg(long)]
        dry_run: bool,
        /// Print the result as JSON
        #[arg(long)]
        json: bool,
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
        /// Attach a bounded formal derivation (proof steps / failure frontier)
        /// to every answered question; answers without a resolution proof say so
        #[arg(long)]
        explain: bool,
        /// When a question cannot be answered, ask the single most discriminative
        /// explanation-space clarification question. Following stdin lines are
        /// treated as answers to that clarification until the session ends.
        #[arg(long)]
        clarify: bool,
    },
    /// Annotate Chinese text with predicate-centered semantic roles and show
    /// the relation-frame projection plus a reversible realization
    Roles {
        /// Sentence to annotate; omit or "-" to read one sentence per line
        text: Option<String>,
        /// Predicate literal for sentences whose schema is not registered
        #[arg(long)]
        predicate: Option<String>,
        /// Print JSON records
        #[arg(long)]
        json: bool,
    },
    /// Analyze dependency spines and compile canonical frames offline; with
    /// --fact and --question, build an in-memory spine index and answer.
    Spine {
        /// Sentence(s) to analyze; omit or "-" to read one line per sentence
        text: Option<String>,
        /// Fact sentence used to build the in-memory index (repeatable)
        #[arg(long = "fact")]
        fact: Vec<String>,
        /// Who/what question answered against the in-memory index
        #[arg(long)]
        question: Option<String>,
        /// Print JSON instead of the Chinese audit view
        #[arg(long)]
        json: bool,
    },
    /// Adjudicate sensitive spans and facts through the POS-typed privacy
    /// flow contract. Audit mode (default) only computes what would happen;
    /// --enforce rewrites/hides values and records the session ledger
    Privacy {
        /// Text to scan for sensitive spans; "-" reads from stdin
        #[arg(long)]
        text: Option<String>,
        /// Fact to adjudicate as SUBJECT|PREDICATE|OBJECT (repeatable)
        #[arg(long = "fact", value_name = "S|P|O")]
        fact: Vec<String>,
        /// File of facts to adjudicate, one SUBJECT|PREDICATE|OBJECT per line
        /// (blank lines and # comments are ignored)
        #[arg(long, value_name = "PATH")]
        facts_file: Option<String>,
        /// Read facts from the knowledge store instead of (or in addition to)
        /// --fact/--facts-file; the store is only opened for this flag
        #[arg(long)]
        from_store: bool,
        /// Keep at most this many store facts (audit summary then covers the
        /// sampled facts only)
        #[arg(long, value_name = "N")]
        max_facts: Option<usize>,
        /// Question text; its interrogative type constrains the replacement
        #[arg(long)]
        question: Option<String>,
        /// Purpose of this disclosure (default: 问答)
        #[arg(long, default_value = "问答")]
        purpose: String,
        /// Requested precision: exact / granular / coarse / type / hidden
        #[arg(long, default_value = "exact")]
        precision: String,
        /// Override the k-anonymity threshold
        #[arg(long)]
        k: Option<usize>,
        /// Enforce the policy instead of audit/shadow mode
        #[arg(long)]
        enforce: bool,
        /// Data subject used by natural-language policy acts (default: 我)
        #[arg(long, default_value = "我")]
        subject: String,
        /// Privacy speech act in natural language, e.g. 「别告诉别人我的电话」
        /// or 「我同意你告诉别人我的邮箱」(repeatable; processed in order)
        #[arg(long = "utterance")]
        utterance: Vec<String>,
        /// Grant session consent as SUBJECT|CATEGORY|PURPOSE (repeatable)
        #[arg(long = "grant", value_name = "S|CATEGORY|PURPOSE")]
        grant: Vec<String>,
        /// Grant one-time consent as SUBJECT|CATEGORY|PURPOSE (repeatable)
        #[arg(long = "grant-once", value_name = "S|CATEGORY|PURPOSE")]
        grant_once: Vec<String>,
        /// Revoke consent as SUBJECT|CATEGORY|PURPOSE (repeatable)
        #[arg(long = "revoke", value_name = "S|CATEGORY|PURPOSE")]
        revoke: Vec<String>,
        /// Forget all facts about a subject, or SUBJECT|CATEGORY (repeatable)
        #[arg(long = "forget", value_name = "SUBJECT[|CATEGORY]")]
        forget: Vec<String>,
        /// Policy file (TOML or JSON); defaults are used when omitted
        #[arg(long)]
        policy: Option<String>,
        /// Session ledger JSON to load and save
        #[arg(long)]
        session: Option<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Explore transitive closure over open predicates in a session ledger:
    /// observe text/triples, ask "all", confirm or reject a transitive
    /// hypothesis, and get witness-carrying answers (never writes the KB)
    OpenClosure {
        /// Natural-language sentence to observe (repeatable)
        #[arg(long = "text")]
        text: Vec<String>,
        /// Fact to observe as SUBJECT|PREDICATE|OBJECT (prefix ! for explicit
        /// negation; repeatable)
        #[arg(long = "fact", value_name = "S|P|O")]
        fact: Vec<String>,
        /// Query as SUBJECT|PREDICATE or Chinese 「X 的所有 Y 是谁？」 (repeatable)
        #[arg(long = "ask", value_name = "QUERY")]
        ask: Vec<String>,
        /// Confirm a predicate as transitive in this session (repeatable)
        #[arg(long = "confirm", value_name = "PREDICATE")]
        confirm: Vec<String>,
        /// Reject a predicate as transitive in this session (repeatable)
        #[arg(long = "reject", value_name = "PREDICATE")]
        reject: Vec<String>,
        /// Retract an observed fact as SUBJECT|PREDICATE|OBJECT (repeatable)
        #[arg(long = "retract", value_name = "S|P|O")]
        retract: Vec<String>,
        /// Dialogue turn processed in order: statement, question,
        /// confirmation/rejection, or 「为什么」 (repeatable)
        #[arg(long = "turn", value_name = "TEXT")]
        turn: Vec<String>,
        /// Session ledger JSON to load and save (state only; the knowledge
        /// store is never opened or written)
        #[arg(long)]
        session: Option<String>,
        /// Maximum closure depth
        #[arg(long, default_value_t = 8)]
        max_depth: usize,
        /// Maximum distinct closure nodes
        #[arg(long, default_value_t = 512)]
        max_nodes: usize,
        /// Maximum neighbor expansions
        #[arg(long, default_value_t = 20_000)]
        max_expansions: usize,
        /// Print the ledger and answers as JSON
        #[arg(long)]
        json: bool,
    },
    /// Inspect a coreference closure from JSON evidence: identity classes,
    /// ambiguity/type guards, and optional relation closure over the quotient
    Coref {
        /// JSON evidence file; "-" reads from stdin; omit when using artifacts
        #[arg(value_name = "EVIDENCE")]
        file_path: Option<String>,
        /// Entity registry JSON to absorb (aliases, merges, types)
        #[arg(long, value_name = "FILE")]
        registry: Option<String>,
        /// Variant closure JSON to absorb (metonymy stays non-identity)
        #[arg(long, value_name = "FILE")]
        variants: Option<String>,
        /// Binding ledger snapshot JSON to absorb
        #[arg(long, value_name = "FILE")]
        bindings: Option<String>,
        /// Referent trail JSON to absorb
        #[arg(long, value_name = "FILE")]
        trail: Option<String>,
        /// Resolve one surface and show its canonical identity + witness
        #[arg(long)]
        resolve: Option<String>,
        /// Relation closure start surface (evaluated on the identity quotient)
        #[arg(long)]
        closure: Option<String>,
        /// Restrict the relation closure to one predicate
        #[arg(long)]
        predicate: Option<String>,
        /// Closure operator: "plus" (default), "star", or "exact:N"
        #[arg(long, default_value = "plus")]
        operator: String,
        /// Transitive law id required by plus/star closures
        #[arg(long)]
        law: Option<String>,
        /// Print JSON instead of the Chinese audit view
        #[arg(long)]
        json: bool,
    },
    /// Inspect type memory: type profiles, members or a pairwise comparison
    Types {
        /// Type to inspect (canonical name or localized label)
        #[arg(long = "type", value_name = "TYPE")]
        type_name: Option<String>,
        /// Compare two types, e.g. --compare 人物 组织
        #[arg(
            long,
            num_args = 2,
            value_names = ["LEFT", "RIGHT"],
            conflicts_with = "type_name"
        )]
        compare: Option<Vec<String>>,
        /// List members instead of the profile (requires --type; use --offset/--limit to page)
        #[arg(long, requires = "type_name")]
        members: bool,
        /// Zero-based member offset (only with --members)
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Maximum members per page (only with --members)
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Print the profile/page/comparison as JSON
        #[arg(long)]
        json: bool,
    },
    /// Inspect or extend the sense-aware synonym graph; resolve a surface or
    /// realize a canonical name with a reversible user-facing surface
    Synonyms {
        /// Surface to resolve (surface -> canonical)
        #[arg(long)]
        surface: Option<String>,
        /// Canonical name to realize back to a user-facing surface
        #[arg(long)]
        canonical: Option<String>,
        /// Synonym layer: entity / predicate / type
        #[arg(long, default_value = "entity")]
        kind: String,
        /// Expected type used as a sense constraint
        #[arg(long = "type")]
        expected_type: Option<String>,
        /// Predicate used as a domain/range context
        #[arg(long)]
        predicate: Option<String>,
        /// Explicit domain constraint
        #[arg(long)]
        domain: Option<String>,
        /// Explicit range constraint
        #[arg(long)]
        range: Option<String>,
        /// Session echo surface for reverse generation (repeatable)
        #[arg(long = "echo")]
        echo: Vec<String>,
        /// Confirm SURFACE -> CANONICAL in the graph
        #[arg(long)]
        confirm: bool,
        /// Refute SURFACE -> CANONICAL in the graph
        #[arg(long)]
        refute: bool,
        /// Merge config assets (profiles, type aliases, flat synonyms) first
        #[arg(long)]
        seed_config: bool,
        /// Print JSON
        #[arg(long)]
        json: bool,
    },
    /// Resolve a name through the anchored variant closure: follow typed,
    /// timed alias/synonym paths to one registered identity, or realize a
    /// reversible surface for a canonical identity (read-only)
    Variants {
        /// Surface to resolve (surface -> anchored identity)
        #[arg(long, conflicts_with = "canonical")]
        surface: Option<String>,
        /// Canonical identity to realize back to a reversible surface
        #[arg(long)]
        canonical: Option<String>,
        /// Expected type used as an anchor guard
        #[arg(long = "type")]
        expected_type: Option<String>,
        /// Predicate used as a sense/domain context
        #[arg(long)]
        predicate: Option<String>,
        /// Explicit domain constraint
        #[arg(long)]
        domain: Option<String>,
        /// Explicit range constraint
        #[arg(long)]
        range: Option<String>,
        /// Reference year for time-windowed variant edges
        #[arg(long)]
        year: Option<i32>,
        /// Session echo surface for reverse generation (repeatable)
        #[arg(long = "echo")]
        echo: Vec<String>,
        /// Also try anchored morphological rules (honorific/suffix/initials)
        #[arg(long)]
        morphology: bool,
        /// Include proposed synonym entries as candidate edges
        #[arg(long)]
        include_pending: bool,
        /// Seed registry profiles and configured synonyms first
        #[arg(long)]
        seed_config: bool,
        /// Print the result as JSON
        #[arg(long)]
        json: bool,
    },
    /// Explain contextual segmentation: symbolic subword evidence, lattice
    /// paths, readings, ambiguity margin and clarification (read-only; opens
    /// the knowledge store for lexicon and graph context)
    Segment {
        /// Text to segment; omit to read stdin
        text: Option<String>,
        /// Focus term used to build the context vector (repeatable)
        #[arg(long = "context")]
        context: Vec<String>,
        /// Expected entity type used as a soft schema constraint (repeatable)
        #[arg(long = "expect")]
        expected_types: Vec<String>,
        /// Expected predicate used as a soft context feature
        #[arg(long)]
        predicate: Option<String>,
        /// Ambiguity margin threshold for clarification
        #[arg(long, default_value_t = 0.05)]
        margin: f64,
        /// Read/write the contextual reading memory JSON at this path
        #[arg(long)]
        memory: Option<String>,
        /// Confirm SURFACE=CANONICAL in the current context (repeatable)
        #[arg(long = "confirm", value_name = "SURFACE=CANONICAL")]
        confirm: Vec<String>,
        /// Correct SURFACE=REJECTED>ACCEPTED in the current context (repeatable)
        #[arg(long = "correct", value_name = "SURFACE=REJECTED>ACCEPTED")]
        correct: Vec<String>,
        /// Print the result as JSON
        #[arg(long)]
        json: bool,
    },
    /// Frame-constrained joint parsing: frame spans and predicate domain/range
    /// feed back into segmentation, then type adjudication and reading
    /// normalization converge (read-only; opens the store for type evidence)
    JointParse {
        /// Text to parse; omit to read stdin
        text: Option<String>,
        /// Focus term used to seed the joint context (repeatable)
        #[arg(long = "context")]
        context: Vec<String>,
        /// Expected entity type used as a soft schema constraint (repeatable)
        #[arg(long = "expect")]
        expected_types: Vec<String>,
        /// Expected predicate used as a soft context feature
        #[arg(long)]
        predicate: Option<String>,
        /// Ambiguity margin threshold for clarification
        #[arg(long, default_value_t = 0.05)]
        margin: f64,
        /// Print the joint parse as JSON
        #[arg(long)]
        json: bool,
    },
    /// Verify a generated candidate by re-reading it: every expected fact must
    /// come back with the same subject, predicate, object, polarity and modality
    JointVerify {
        /// Candidate sentence to re-read; omit to read stdin
        text: Option<String>,
        /// Expected fact as SUBJECT|PREDICATE|OBJECT (prefix ! for negation; repeatable)
        #[arg(long = "fact", value_name = "S|P|O", required = true)]
        fact: Vec<String>,
        /// Focus term used to seed the joint context (repeatable)
        #[arg(long = "context")]
        context: Vec<String>,
        /// Expected entity type used as a soft schema constraint (repeatable)
        #[arg(long = "expect")]
        expected_types: Vec<String>,
        /// Expected predicate used as a soft context feature
        #[arg(long)]
        predicate: Option<String>,
        /// Print the parse and verification as JSON
        #[arg(long)]
        json: bool,
    },
    /// Generate a candidate from expected facts with the existing generator,
    /// then re-read it through joint parsing and report whether the same
    /// frames come back
    JointGenerate {
        /// Expected fact as SUBJECT|PREDICATE|OBJECT (prefix ! for negation; repeatable)
        #[arg(long = "fact", value_name = "S|P|O", required = true)]
        fact: Vec<String>,
        /// Focus term used to seed the joint context (repeatable)
        #[arg(long = "context")]
        context: Vec<String>,
        /// Expected entity type used as a soft schema constraint (repeatable)
        #[arg(long = "expect")]
        expected_types: Vec<String>,
        /// Expected predicate used as a soft context feature
        #[arg(long)]
        predicate: Option<String>,
        /// Print the candidate and verification as JSON
        #[arg(long)]
        json: bool,
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
        /// Also print the canonical query identity and generated-question readback
        #[arg(long)]
        identity: bool,
        /// Print the trace/report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Compile the bidirectional semantic rule pack, preview extraction or
    /// realization, and verify round-trip consistency (offline; no store)
    Grammar {
        /// Natural-language text to read with the configured rule pack
        #[arg(conflicts_with = "fact")]
        text: Option<String>,
        /// Fact to realize as "subject|predicate|value" (or whitespace separated)
        #[arg(long, conflicts_with = "text")]
        fact: Option<String>,
        /// Re-read every realized sentence and require the same fact back
        #[arg(long)]
        roundtrip: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Inspect or extend the persistent predicate registry: observed
    /// predicates, pending admissions, aliases, inverses and functional marks
    Predicates {
        /// Print the registry and the applied edits as JSON
        #[arg(long)]
        json: bool,
        /// Confirm a pending predicate (repeatable)
        #[arg(long = "confirm", value_name = "PREDICATE")]
        confirm: Vec<String>,
        /// Reject a pending predicate and remove its triples (repeatable)
        #[arg(long = "reject", value_name = "PREDICATE")]
        reject: Vec<String>,
        /// Register an alias: ALIAS=CANONICAL (repeatable)
        #[arg(long = "alias", value_name = "ALIAS=CANONICAL")]
        alias: Vec<String>,
        /// Register inverse predicates: LEFT=RIGHT (repeatable)
        #[arg(long = "inverse", value_name = "LEFT=RIGHT")]
        inverse: Vec<String>,
        /// Mark predicates as functional, i.e. one value per subject (repeatable)
        #[arg(long = "functional", value_name = "PREDICATE")]
        functional: Vec<String>,
        /// Clear the functional mark (repeatable)
        #[arg(long = "non-functional", value_name = "PREDICATE")]
        non_functional: Vec<String>,
    },
    /// Inspect or extend the persistent lexical memory: POS evidence and
    /// classifier→type expectations stored in their own named graph
    LexicalMemory {
        /// Text to tokenize and remember (repeatable; each text is one turn)
        #[arg(long = "learn")]
        learn: Vec<String>,
        /// Look up one word or classifier and print its soft type expectation
        #[arg(long)]
        query: Option<String>,
        /// Drop all remembered evidence before applying --learn
        #[arg(long)]
        forget: bool,
        /// Print the report/query as JSON
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
    /// Mine surface→canonical rewrite rules from paired question episodes and
    /// evaluate them on held-out predicates (offline; does not open the store)
    RewriteMine {
        /// JSONL or JSON episode corpus; omit for the built-in seed corpus
        #[arg(long)]
        episodes: Option<String>,
        /// Predicate to hold out (repeatable); defaults to every third sorted predicate
        #[arg(long = "holdout-predicate", value_name = "PREDICATE")]
        holdout_predicates: Vec<String>,
        /// Minimum support before a rule is proposed
        #[arg(long, default_value_t = 2)]
        min_support: usize,
        /// Persist the mined rule book to this path
        #[arg(long)]
        save: Option<String>,
        /// Exit non-zero when a held-out rewrite is wrong or coverage does not improve
        #[arg(long)]
        strict: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Inspect, adopt or retire a persisted surface→canonical rewrite book
    RewriteRules {
        /// Rule book path (created by `rewrite-mine --save`)
        #[arg(long, default_value = "rewrite_book.json")]
        book: String,
        /// Adopt a proposed rule id (repeatable; prefix allowed)
        #[arg(long = "adopt", value_name = "RULE_ID")]
        adopt: Vec<String>,
        /// Retire an active rule id (repeatable; prefix allowed)
        #[arg(long = "retire", value_name = "RULE_ID")]
        retire: Vec<String>,
        /// Print the book as JSON
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
    /// Evaluate constructive multi-hop recovery on the built-in missing-edge
    /// benchmark, comparing one-hop link prediction with bounded virtual facts
    EvalConstruction {
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
        /// Exit non-zero unless every target is recovered without a wrong fact
        #[arg(long)]
        strict: bool,
    },
    /// Evaluate answer-probed reference resolution (grounding certificates /
    /// repair recall / over-ask / truncation safety) on the built-in gold corpus
    /// with a regression ratchet
    EvalGrounding {
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
    /// Build an event-projection graph from text and optionally answer
    /// who/when/where/why/what-next/participants/did questions (read-only;
    /// does not open the knowledge store)
    Events {
        /// Text to analyse (mutually exclusive with --file)
        #[arg(long, conflicts_with = "file")]
        text: Option<String>,
        /// UTF-8 text file to analyse
        #[arg(long)]
        file: Option<String>,
        /// Event question; repeat by separating questions with newlines.
        /// Omit to print the graph and its evidence
        #[arg(long)]
        question: Option<String>,
        /// Maximum number of events to keep (0 = no limit)
        #[arg(long, default_value_t = 200)]
        limit: usize,
        /// Print the graph or answers as JSON
        #[arg(long)]
        json: bool,
        /// Also print the event graph as Prolog facts
        #[arg(long)]
        prolog: bool,
        /// Print only the event graph as N-Quads in a named graph
        /// (default graph untouched; conflicts with --question/--json)
        #[arg(long, conflicts_with_all = ["question", "json"])]
        nquads: bool,
        /// Named graph IRI for --nquads (default: derived from the source fingerprint)
        #[arg(long, requires = "nquads")]
        graph: Option<String>,
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
    /// Learn attribute extraction rules from seed facts x corpus; only
    /// explicitly adopted rules are persisted and reused by text imports
    BatchTeach {
        /// Seed facts file (TSV: subject<TAB>predicate<TAB>value[<TAB>entity_type], or JSONL;
        /// predicate must name a configured property or its canonical predicate)
        #[arg(long)]
        facts: Option<String>,
        /// Corpus file with natural-language sentences
        #[arg(long)]
        corpus: Option<String>,
        /// Explicit negative sentences (one per line); any match rejects a rule
        #[arg(long)]
        negatives: Option<String>,
        /// Minimum distinct supported facts before a rule may be adopted
        #[arg(long, default_value_t = batch_teaching::DEFAULT_MIN_SUPPORT)]
        min_support: usize,
        /// Minimum match purity in per mille
        #[arg(long, default_value_t = batch_teaching::DEFAULT_MIN_PURITY)]
        min_purity: u32,
        /// Maximum number of candidates to keep
        #[arg(long, default_value_t = batch_teaching::DEFAULT_MAX_CANDIDATES)]
        max_candidates: usize,
        /// Explicitly adopt a ready rule id (repeatable; prefix allowed, "all" = every ready rule)
        #[arg(long = "adopt", value_name = "RULE_ID")]
        adopt: Vec<String>,
        /// Retire an active rule id (repeatable; prefix allowed)
        #[arg(long = "retire", value_name = "RULE_ID")]
        retire: Vec<String>,
        /// Ledger path (defaults to <store>/batch_teaching.json)
        #[arg(long)]
        ledger: Option<String>,
        /// List active rules and pending candidates, then exit (does not open the store)
        #[arg(long)]
        list: bool,
        /// Print the result as JSON
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
    /// Search through the unified semantic port (dense + holographic) and
    /// print grouped answer values with an auditable receipt
    Search {
        /// Natural-language question or keyword query
        question: String,
        /// Retrieval action: lookup / enumerate / describe / verify
        #[arg(long)]
        mode: Option<String>,
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
        /// Maximum number of evidence rows to return
        #[arg(long, default_value_t = 8)]
        k: usize,
        /// Maximum number of candidates to score (default: unlimited)
        #[arg(long)]
        max_scanned: Option<usize>,
        /// Print answers, hits and the receipt as JSON
        #[arg(long)]
        json: bool,
        /// Also print the evidence-backed binding table with its coverage
        /// scope and knowledge revision
        #[arg(long)]
        table: bool,
    },
    /// Arbitrate query readings (exact plan / syntax rewrite / anchoring)
    /// with dense-retrieval evidence and print the three-way decision receipt
    RewriteRetrieval {
        /// Question to analyse
        question: String,
        /// Rewrite book path (overrides <store>/rewrite_book.json)
        #[arg(long)]
        book: Option<String>,
        /// Print the assessment as JSON
        #[arg(long)]
        json: bool,
    },
    /// Evaluate the rewrite-retrieval arbitration on a TSV corpus
    /// (`question<TAB>expected`, expected may use `|` alternatives)
    RewriteRetrievalEval {
        /// Corpus file path
        corpus: String,
        /// Rewrite book path (overrides <store>/rewrite_book.json)
        #[arg(long)]
        book: Option<String>,
        /// Exit non-zero when any accepted reading is wrong
        #[arg(long)]
        strict: bool,
        /// Print the report as JSON
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
    /// Inspect the layered fact snapshot: effective facts, audit records and
    /// functional-slot conflicts under one consistent read-only world
    Snapshot {
        /// Subject to inspect; omit --predicate to list all its facts
        #[arg(long)]
        subject: Option<String>,
        /// Predicate to inspect (requires --subject)
        #[arg(long, requires = "subject")]
        predicate: Option<String>,
        /// Show every layered record in the slot, not only effective values
        #[arg(long)]
        all: bool,
        /// Maximum rows printed in the whole-snapshot view
        #[arg(long, default_value_t = 40)]
        limit: usize,
        /// Print the snapshot/slot as JSON
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
        /// Also print the answer-time arbitration (independent source
        /// components, trusted winners, contested slots)
        #[arg(long)]
        arbitrate: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Import an RDF file into a named source namespace, list namespaces, or
    /// change a namespace status. Raw assertions stay in their own named
    /// graph; the default graph is untouched until `fuse --apply`
    Namespace {
        /// Namespace id (IRI-safe slug), e.g. "alice" or "batch-2026-10"
        #[arg(long, required_unless_present = "list")]
        source: Option<String>,
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
        #[arg(required_unless_present_any = ["list", "status"])]
        file_path: Option<String>,
        /// List namespaces, assertion counts and adjudications, then exit
        #[arg(long, conflicts_with = "file_path")]
        list: bool,
        /// Set namespace status: active / quarantined / retired
        #[arg(
            long,
            value_name = "STATUS",
            requires = "source",
            conflicts_with_all = ["list", "file_path"]
        )]
        status: Option<String>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Answer a question under an explicit context lens: the default graph, a
    /// fusion source namespace, a session/hypothesis/import kind, or every
    /// context. Reports support / conflict / unknown with provenance instead
    /// of silently flattening all contexts into one answer
    Lens {
        /// Question carrying a context cue, e.g. "据alice说，张三的师傅是谁"
        #[arg(required_unless_present_any = ["subject", "contexts", "facts"])]
        question: Option<String>,
        /// Explicit subject (with --facts, optional predicate filter)
        #[arg(long)]
        subject: Option<String>,
        /// Explicit predicate (requires --subject)
        #[arg(long, requires = "subject")]
        predicate: Option<String>,
        /// Force the lens: default / all / session / hypothesis / import /
        /// namespace:<id> / <namespace id>
        #[arg(long)]
        view: Option<String>,
        /// Treat this predicate as functional (repeatable)
        #[arg(long = "functional", value_name = "PREDICATE")]
        functional: Vec<String>,
        /// List known contexts and exit
        #[arg(long, conflicts_with = "facts")]
        contexts: bool,
        /// List facts in the selected lens instead of answering a slot query;
        /// --subject / --predicate become optional filters
        #[arg(long, conflicts_with_all = ["question", "contexts"])]
        facts: bool,
        /// Include session facts and current assumptions from a saved dialogue
        /// state file (see reasoning.dialogue_state_path)
        #[arg(long, value_name = "PATH")]
        dialogue_state: Option<String>,
        /// Print the result as JSON
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
    /// Inspect dialogue artifact URIs (session / turn / reply / claim /
    /// mention / evidence): list the index, or resolve a URI or selector
    DialogueArtifacts {
        /// URI or selector (上一轮 / 第2点 / 依据 / 实体：姚明); omit to list
        selector: Option<String>,
        /// Override the chronicle path (defaults to the configured dialogue
        /// state path or store directory)
        #[arg(long)]
        chronicle: Option<String>,
        /// Override the session identity sidecar (defaults to the chronicle
        /// path + .session.json)
        #[arg(long)]
        session: Option<String>,
    },
    /// Show the lexical transducer's surface → sense decisions, with an
    /// optional canonical form to echo back through the session ledger
    Transduce {
        /// Text to transduce; "-" or omission reads newline-delimited text from stdin
        text: Option<String>,
        /// Canonical form to echo back (uses recent user surface when reversible)
        #[arg(long)]
        echo: Option<String>,
        /// Restrict rules to one layer: entity / predicate / type
        #[arg(long)]
        kind: Option<String>,
        /// Print the transduction as JSON
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
    /// Replay session contracts (process receipts) offline; path may be a
    /// file or a directory (default examples/sessions); exits non-zero on any
    /// failure
    Session {
        /// Contract TOML file or directory; omit for examples/sessions
        path: Option<String>,
        /// Print the reports as a JSON array
        #[arg(long)]
        json: bool,
    },
    /// Mine, review, export or import OWL axioms proposed from the graph
    #[command(disable_help_flag = true)]
    Owl {
        /// Arguments forwarded to the OWL review command
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "ARGS"
        )]
        args: Vec<String>,
    },
    /// Run interactive mode (default)
    Interactive,
}

#[derive(Subcommand, Debug)]
enum OwlCommands {
    /// Mine candidate axioms from the default graph and record them in the
    /// review ledger; proposals never affect reasoning until reviewed
    Propose {
        /// Minimum evidence count for a candidate
        #[arg(long, default_value_t = 2)]
        min_support: usize,
        /// Minimum purity in per mille (0-1000)
        #[arg(long, default_value_t = 800)]
        min_purity: u32,
        /// Maximum number of candidates to keep per run
        #[arg(long, default_value_t = 32)]
        max_candidates: usize,
        /// Print the ledger and outcome as JSON
        #[arg(long)]
        json: bool,
    },
    /// Accept, reject, defer or revise one candidate axiom by id
    Review {
        /// Candidate id (owl-0001) or stable axiom key
        id: String,
        #[arg(value_enum)]
        action: OwlReviewAction,
        /// Reviewer name stored with the decision
        #[arg(long, default_value = "human")]
        reviewer: String,
        /// Human-readable reason stored with the decision
        #[arg(long)]
        note: Option<String>,
        /// Replacement lesson text for `revise`, e.g. "学生是一种人物"
        #[arg(long)]
        lesson: Option<String>,
    },
    /// Render the review ledger and the accepted axiom program
    Report {
        /// Print the ledger as JSON
        #[arg(long)]
        json: bool,
    },
    /// Export annotated axioms for other OWL tools or later import
    Export {
        /// Output format: turtle (annotated OWL) or json (full review ledger)
        #[arg(long, default_value = "turtle")]
        format: String,
        /// Output path; omit or "-" for stdout
        #[arg(long, value_name = "PATH")]
        output: Option<String>,
        /// Include proposed/deferred/rejected entries in the Turtle export
        #[arg(long)]
        include_proposed: bool,
    },
    /// Import an annotated Turtle file or review ledger JSON
    Import {
        /// Input file path
        path: String,
        /// Input format: auto, turtle or json
        #[arg(long, default_value = "auto")]
        format: String,
        /// Replace the existing ledger instead of merging decisions by axiom key
        #[arg(long)]
        replace: bool,
    },
}
