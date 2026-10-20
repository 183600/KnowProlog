//! 文法式关系抽取：把 Scryer Prolog 的 DCG 当"语言器官"使用。
//!
//! 声明式文法覆盖各类语序变换（被字句、把字句、并列、
//! 「A是B的P」与「A的P是B」）：
//!   Rust 侧：jieba 分词 + 词性标注（优先用外部配置词性）→ tok(词, 词性) 事实序列；
//!   Prolog 侧：DCG 规则描述句子构造 → 逐一枚举关系三元组。
//! 词库由谓词画像注册表（PPR）编译注入：verb_surface(词面, 规范谓词, keep|swap)，
//! 于是「改一次 TOML 声明 → 抽取也能认识新谓词」，且导入方向与查询口径一致。
//!
//! 纯符号：无神经网络、无训练；每条三元组都能追溯到文法规则与词库条目。

use crate::nlp::{shared_jieba, TaggedWord};
use crate::prolog_engine::escape_atom;
use parking_lot::Mutex;
use scryer_prolog::machine::parsed_results::{QueryResolution, Value};
use scryer_prolog::machine::Machine;
use std::collections::HashSet;
use std::fmt::Write as _;

/// DCG 文法（一次装载，常驻机器）。
///
/// 约定与 `chinese_reasoning_nlu` 保持一致：二元谓词 P(A, B)。
///   - SVO「A V B」        → V(A, B)
///   - 被字句「A 被 B V」  → V(B, A)（B 是施事）
///   - 把字句「A 把 B V」  → V(A, B)
///   - 所有格「A 的 P 是 B」→ P(A, B)（与 parse_proposition 同向）
///   - 逆所有格「A 是 B 的 P」→ P(B, A)
///   - V于O「A V 于 B」    → "V于"(A, B)
/// 实体并列（和/、/及）通过 member/2 展开为多条三元组。
const GRAMMAR: &str = r#"
:- use_module(library(dcgs)).
:- use_module(library(lists)).

% 词库：由 PPR 编译注入；缺省时词面自任谓词
:- dynamic(verb_surface/3).

entity_pos(nr). entity_pos(ns). entity_pos(nt). entity_pos(nz). entity_pos(n).
verb_pos(v). verb_pos(vn).

entity(tok(W, P)) --> [tok(W, P)], { entity_pos(P) }.
verb(tok(W, P)) --> [tok(W, P)], { verb_pos(P) }.

% 可选体标记「了」与常见状语，保证语序弹性
opt_le --> [tok('了', _)].
opt_le --> [].

opt_adv --> [tok('都', _)].
opt_adv --> [tok('也', _)].
opt_adv --> [tok('还', _)].
opt_adv --> [tok('就', _)].
opt_adv --> [].

map_surface(Surface, Canonical, Dir) :- verb_surface(Surface, Canonical, Dir), !.
map_surface(Surface, Surface, keep).

orient(keep, A, B, A, B).
orient(swap, A, B, B, A).

% 实体并列：A、B、C …（和/、/及），词而非 tok 包装
entity_list([W]) --> entity(tok(W, _)).
entity_list([W|Rest]) --> entity(tok(W, _)), [tok('和', _)], entity_list(Rest).
entity_list([W|Rest]) --> entity(tok(W, _)), [tok('、', _)], entity_list(Rest).
entity_list([W|Rest]) --> entity(tok(W, _)), [tok('及', _)], entity_list(Rest).

% 1) SVO：主语（可并列）+ 状语 + 动词 + 了 + 宾语（可并列）
clause(Pred, S, O) -->
    entity_list(L), opt_adv, verb(tok(V, _)), opt_le, entity_list(R),
    { map_surface(V, Pred, Dir), member(A, L), member(B, R), orient(Dir, A, B, S, O) }.

% 2) 被字句：受事 被 施事 动词（了）
clause(Pred, S, O) -->
    entity(tok(P, _)), [tok('被', _)], entity(tok(Ag, _)), verb(tok(V, _)), opt_le,
    { map_surface(V, Pred, Dir), orient(Dir, Ag, P, S, O) }.

% 3) 把字句：施事 把 受事 动词（了）
clause(Pred, S, O) -->
    entity(tok(Ag, _)), [tok('把', _)], entity(tok(P, _)), verb(tok(V, _)), opt_le,
    { map_surface(V, Pred, Dir), orient(Dir, Ag, P, S, O) }.

% 4) 所有格：A 的 P 是 B
clause(Pred, S2, O2) -->
    entity(tok(S, _)), [tok('的', _)], entity(tok(Surface, _)), [tok('是', _)], entity(tok(O, _)),
    { map_surface(Surface, Pred, Dir), orient(Dir, S, O, S2, O2) }.

% 5) 逆所有格：A 是 B 的 P
clause(Pred, S2, O2) -->
    entity(tok(O, _)), [tok('是', _)], entity(tok(S, _)), [tok('的', _)], entity(tok(Surface, _)),
    { map_surface(Surface, Pred, Dir), orient(Dir, S, O, S2, O2) }.

% 6) V于O：A V 于 B（如「供职于」「出生于」被分词切碎时的兜底）
clause(Pred, S2, O2) -->
    entity(tok(S, _)), verb(tok(V, _)), [tok('于', _)], entity(tok(O, _)),
    { atom_concat(V, '于', Surface), map_surface(Surface, Pred, Dir), orient(Dir, S, O, S2, O2) }.
"#;

/// 一条由文法抽取出的关系三元组。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GrammarTriple {
    pub predicate: String,
    pub subject: String,
    pub object: String,
}

struct Inner {
    machine: Machine,
    lexicon: Vec<(String, String, bool)>,
}

/// DCG 关系抽取器：线程安全（内部互斥），可跨导入轮次复用。
pub struct GrammarExtractor {
    inner: Mutex<Inner>,
}

impl GrammarExtractor {
    pub fn new() -> Self {
        let mut machine = Machine::new_lib();
        machine.consult_module_string("user", GRAMMAR.to_string());
        Self {
            inner: Mutex::new(Inner {
                machine,
                lexicon: Vec::new(),
            }),
        }
    }

    /// 同步 PPR 词库（词面 → 规范谓词, 是否逆向）；词库未变时不重编译。
    pub fn set_lexicon_if_changed(&self, pairs: &[(String, String, bool)]) -> bool {
        let mut inner = self.inner.lock();
        if inner.lexicon == pairs {
            return false;
        }
        let mut code = String::with_capacity(pairs.len() * 48 + 64);
        for (surface, canonical, inverted) in pairs {
            let direction = if *inverted { "swap" } else { "keep" };
            let _ = writeln!(
                code,
                "verb_surface('{}', '{}', {}).",
                escape_atom(surface),
                escape_atom(canonical),
                direction
            );
        }
        let _ = inner
            .machine
            .run_query("retractall(verb_surface(_, _, _)).".to_string());
        inner.machine.consult_module_string("user", code);
        inner.lexicon = pairs.to_vec();
        true
    }

    /// 当前词库条目数（用于自检/报告）。
    pub fn lexicon_size(&self) -> usize {
        self.inner.lock().lexicon.len()
    }

    /// 从中文文本抽取关系三元组：先用共享 jieba 标注词性（无配置场景）。
    /// 交互/导入主链路请用 `extract_from_tagged`，以便复用配置里的
    /// 人名表与自定义词性。
    pub fn extract(&self, text: &str) -> Vec<GrammarTriple> {
        let mut triples = Vec::new();
        let mut seen = HashSet::new();
        for sentence in split_sentences(text) {
            let tokens = tag_tokens(sentence);
            for triple in self.extract_tokens(tokens) {
                if seen.insert(triple.clone()) {
                    triples.push(triple);
                }
            }
        }
        triples
    }

    /// 从外部已标注词流抽取（按句切分，unknown 词性用 jieba 单词标注补全）。
    pub fn extract_from_tagged(&self, words: &[TaggedWord]) -> Vec<GrammarTriple> {
        let mut triples = Vec::new();
        let mut seen = HashSet::new();
        let mut sentence: Vec<(String, String)> = Vec::new();
        for word in words {
            if is_sentence_boundary(&word.text) {
                if !sentence.is_empty() {
                    for triple in self.extract_tokens(std::mem::take(&mut sentence)) {
                        if seen.insert(triple.clone()) {
                            triples.push(triple);
                        }
                    }
                }
                continue;
            }
            let text = word.text.trim();
            if text.is_empty() || is_punctuation(text) {
                continue;
            }
            sentence.push((text.to_string(), refine_pos(word)));
        }
        if !sentence.is_empty() {
            for triple in self.extract_tokens(sentence) {
                if seen.insert(triple.clone()) {
                    triples.push(triple);
                }
            }
        }
        triples
    }

    fn extract_tokens(&self, tokens: Vec<(String, String)>) -> Vec<GrammarTriple> {
        if tokens.len() < 3 {
            return Vec::new();
        }
        let token_list: Vec<String> = tokens
            .iter()
            .map(|(word, pos)| format!("tok('{}', '{}')", escape_atom(word), escape_atom(pos)))
            .collect();
        let query = format!("clause(P, S, O, [{}], []).", token_list.join(", "));

        let mut triples = Vec::new();
        let mut inner = self.inner.lock();
        match inner.machine.run_query(query) {
            Ok(QueryResolution::Matches(matches)) => {
                for query_match in matches {
                    let (Some(predicate), Some(subject), Some(object)) = (
                        query_match.bindings.get("P").and_then(value_text),
                        query_match.bindings.get("S").and_then(value_text),
                        query_match.bindings.get("O").and_then(value_text),
                    ) else {
                        continue;
                    };
                    if subject.is_empty()
                        || object.is_empty()
                        || subject == object
                        || predicate.is_empty()
                    {
                        continue;
                    }
                    triples.push(GrammarTriple {
                        predicate,
                        subject,
                        object,
                    });
                }
            }
            Ok(_) => {}
            Err(err) => eprintln!("文法抽取查询失败: {}", err),
        }
        triples
    }
}

impl Default for GrammarExtractor {
    fn default() -> Self {
        Self::new()
    }
}

fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::Atom(atom) => Some(atom.as_str().to_string()),
        Value::String(text) => Some(text.clone()),
        _ => None,
    }
}

/// 句末标点与换行：作为句子切分点。
fn is_sentence_boundary(text: &str) -> bool {
    matches!(text, "。" | "！" | "？" | "；" | "!" | "?" | ";" | "\n")
}

/// 其它标点（逗号等）：不参与文法匹配。
fn is_punctuation(text: &str) -> bool {
    text.chars().all(|ch| {
        matches!(
            ch,
            '，' | ','
                | '、'
                | '：'
                | ':'
                | '「'
                | '」'
                | '《'
                | '》'
                | '“'
                | '”'
                | '（'
                | '）'
                | '('
                | ')'
                | '—'
                | '…'
                | '·'
                | '．'
                | '.'
        )
    })
}

/// 句子切分：按常见句末标点与换行切开。
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '。' | '！' | '？' | '；' | '!' | '?' | ';' | '\n') {
            let sentence = text[start..index].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = index + ch.len_utf8();
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

/// jieba 分词 + 词性标注 → (词, 词性)，过滤标点与空白。
fn tag_tokens(text: &str) -> Vec<(String, String)> {
    shared_jieba()
        .tag(text, false)
        .into_iter()
        .filter(|tag| !tag.word.trim().is_empty() && !matches!(tag.tag, "w" | "x"))
        .map(|tag| (tag.word.to_string(), tag.tag.to_string()))
        .collect()
}

/// 词性补全：外部标注为 unknown 时，用 jieba 对单词本身再标注一次。
fn refine_pos(word: &TaggedWord) -> String {
    if word.pos != "unknown" {
        return word.pos.clone();
    }
    let tags = shared_jieba().tag(&word.text, false);
    if tags.len() == 1 {
        tags[0].tag.to_string()
    } else {
        "unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(spec: &[(&str, &str)]) -> Vec<TaggedWord> {
        spec.iter()
            .map(|(text, pos)| TaggedWord {
                text: text.to_string(),
                pos: pos.to_string(),
            })
            .collect()
    }

    fn predicates(triples: &[GrammarTriple]) -> Vec<(String, String, String)> {
        triples
            .iter()
            .map(|t| (t.predicate.clone(), t.subject.clone(), t.object.clone()))
            .collect()
    }

    #[test]
    fn extracts_svo() {
        let extractor = GrammarExtractor::new();
        let triples = extractor.extract_from_tagged(&words(&[
            ("张三", "nr"),
            ("批评", "v"),
            ("李四", "nr"),
            ("。", "w"),
        ]));
        assert!(
            predicates(&triples).contains(&("批评".into(), "张三".into(), "李四".into())),
            "SVO 应被抽取：{:?}",
            triples
        );
    }

    #[test]
    fn extracts_passive() {
        let extractor = GrammarExtractor::new();
        let triples = extractor.extract_from_tagged(&words(&[
            ("李四", "nr"),
            ("被", "p"),
            ("张三", "nr"),
            ("批评", "v"),
            ("了", "ul"),
        ]));
        assert!(
            predicates(&triples).contains(&("批评".into(), "张三".into(), "李四".into())),
            "被字句应换向为 V(施事, 受事)：{:?}",
            triples
        );
    }

    #[test]
    fn extracts_disposal() {
        let extractor = GrammarExtractor::new();
        let triples = extractor.extract_from_tagged(&words(&[
            ("张三", "nr"),
            ("把", "p"),
            ("李四", "nr"),
            ("批评", "v"),
            ("了", "ul"),
        ]));
        assert!(
            predicates(&triples).contains(&("批评".into(), "张三".into(), "李四".into())),
            "把字句应抽取 V(施事, 受事)：{:?}",
            triples
        );
    }

    #[test]
    fn extracts_possessive_with_registry_lexicon() {
        let extractor = GrammarExtractor::new();
        extractor.set_lexicon_if_changed(&[("老师".to_string(), "师傅".to_string(), false)]);
        let triples = extractor.extract_from_tagged(&words(&[
            ("张三", "nr"),
            ("的", "u"),
            ("老师", "n"),
            ("是", "v"),
            ("李四", "nr"),
        ]));
        assert!(
            predicates(&triples).contains(&("师傅".into(), "张三".into(), "李四".into())),
            "词库别名应归一为规范谓词：{:?}",
            triples
        );
    }

    #[test]
    fn extracts_inverse_possessive_with_swap() {
        let extractor = GrammarExtractor::new();
        extractor.set_lexicon_if_changed(&[("徒弟".to_string(), "师傅".to_string(), true)]);
        let triples = extractor.extract_from_tagged(&words(&[
            ("张三", "nr"),
            ("的", "u"),
            ("徒弟", "n"),
            ("是", "v"),
            ("李四", "nr"),
        ]));
        assert!(
            predicates(&triples).contains(&("师傅".into(), "李四".into(), "张三".into())),
            "逆向词面应换向：{:?}",
            triples
        );
    }

    #[test]
    fn extracts_reverse_possessive() {
        let extractor = GrammarExtractor::new();
        let triples = extractor.extract_from_tagged(&words(&[
            ("张三", "nr"),
            ("是", "v"),
            ("李四", "nr"),
            ("的", "u"),
            ("老师", "n"),
        ]));
        assert!(
            predicates(&triples).contains(&("老师".into(), "李四".into(), "张三".into())),
            "「A是B的P」应读作 P(B, A)：{:?}",
            triples
        );
    }

    #[test]
    fn extracts_coordination() {
        let extractor = GrammarExtractor::new();
        let triples = extractor.extract_from_tagged(&words(&[
            ("张三", "nr"),
            ("和", "c"),
            ("李四", "nr"),
            ("都", "d"),
            ("批评", "v"),
            ("王五", "nr"),
        ]));
        let extracted = predicates(&triples);
        assert!(
            extracted.contains(&("批评".into(), "张三".into(), "王五".into()))
                && extracted.contains(&("批评".into(), "李四".into(), "王五".into())),
            "并列主语应展开为多条：{:?}",
            triples
        );
    }

    #[test]
    fn extracts_verb_yu_surface() {
        let extractor = GrammarExtractor::new();
        let triples = extractor.extract_from_tagged(&words(&[
            ("张三", "nr"),
            ("供职", "v"),
            ("于", "p"),
            ("北京", "ns"),
        ]));
        assert!(
            triples
                .iter()
                .any(|t| t.predicate == "供职于" && t.subject == "张三" && t.object == "北京"),
            "「V于O」应拼回完整谓词：{:?}",
            triples
        );
    }

    #[test]
    fn lexicon_change_is_detected() {
        let extractor = GrammarExtractor::new();
        let pairs = vec![("老师".to_string(), "师傅".to_string(), false)];
        assert!(extractor.set_lexicon_if_changed(&pairs));
        assert!(!extractor.set_lexicon_if_changed(&pairs));
        assert_eq!(extractor.lexicon_size(), 1);
    }
}
