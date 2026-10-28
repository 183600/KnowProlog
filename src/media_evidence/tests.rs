use super::*;

use crate::context_image::ContextImage;

fn limits() -> MediaLimits {
    MediaLimits {
        max_assets: 8,
        max_asset_bytes: 64,
        max_total_bytes: 128,
        max_anchors_per_asset: 4,
        max_anchor_text_chars: 64,
    }
}

fn image(store: &mut MediaStore, bytes: &[u8]) -> AssetRecord {
    store
        .publish(Modality::Image, "image/png", "", bytes)
        .unwrap()
}

fn table(store: &mut MediaStore, bytes: &[u8]) -> AssetRecord {
    store
        .publish(Modality::Table, "text/csv", "", bytes)
        .unwrap()
}

fn cell_anchor(
    store: &mut MediaStore,
    asset: &AssetRecord,
    row: u32,
    column: u32,
    text: &str,
) -> Anchor {
    store
        .add_anchor(
            &asset.id,
            AnchorDraft::new(
                Locator::TableCell { row, column },
                text,
                "table_parser",
                0.9,
            ),
        )
        .unwrap()
}

#[test]
fn publish_is_content_addressed_and_idempotent() {
    let mut store = MediaStore::new(limits());
    let first = image(&mut store, b"same-bytes");
    let second = image(&mut store, b"same-bytes");
    assert_eq!(first.id, second.id);
    assert_eq!(first.sequence, second.sequence);
    assert_eq!(store.len(), 1);
    assert_eq!(store.total_bytes(), first.byte_len);
    assert_eq!(store.bytes(&first.id).unwrap(), b"same-bytes");
    assert_eq!(first.label, "图片 1");
    assert_eq!(store.assets_in_order().len(), 1);
}

#[test]
fn limits_are_typed_and_never_silently_truncate() {
    let mut tight = MediaStore::new(MediaLimits {
        max_assets: 1,
        max_asset_bytes: 4,
        max_total_bytes: 4,
        max_anchors_per_asset: 1,
        max_anchor_text_chars: 4,
    });
    assert_eq!(
        tight
            .publish(Modality::Image, "image/png", "", b"")
            .unwrap_err(),
        MediaError::EmptyAsset
    );
    assert_eq!(
        tight
            .publish(Modality::Image, "image/png", "", b"12345")
            .unwrap_err(),
        MediaError::AssetTooLarge {
            actual: 5,
            limit: 4
        }
    );
    tight
        .publish(Modality::Image, "image/png", "", b"1234")
        .unwrap();
    assert_eq!(
        tight
            .publish(Modality::Image, "image/png", "", b"12")
            .unwrap_err(),
        MediaError::AssetCapacityExceeded { limit: 1 }
    );
    assert_eq!(
        tight
            .publish(Modality::Image, "image/png", "", b"x")
            .unwrap_err()
            .code(),
        "asset_capacity_exceeded"
    );

    let mut total = MediaStore::new(MediaLimits {
        max_assets: 4,
        max_asset_bytes: 4,
        max_total_bytes: 5,
        max_anchors_per_asset: 1,
        max_anchor_text_chars: 4,
    });
    total
        .publish(Modality::Image, "image/png", "", b"abcd")
        .unwrap();
    assert_eq!(
        total
            .publish(Modality::Image, "image/png", "", b"12")
            .unwrap_err(),
        MediaError::TotalBytesExceeded {
            actual: 6,
            limit: 5
        }
    );
}

#[test]
fn anchors_validate_modality_text_confidence_and_capacity() {
    let mut store = MediaStore::new(limits());
    let asset = image(&mut store, b"anchor-image");
    assert_eq!(
        store
            .add_anchor(
                &asset.id,
                AnchorDraft::new(Locator::TableCell { row: 1, column: 1 }, "x", "ocr", 0.9)
            )
            .unwrap_err(),
        MediaError::LocatorModalityMismatch {
            modality: Modality::Image,
            locator: "table_cell"
        }
    );
    assert_eq!(
        store
            .add_anchor(
                &asset.id,
                AnchorDraft::new(
                    Locator::ImageRegion {
                        x: 900,
                        y: 0,
                        width: 200,
                        height: 100
                    },
                    "x",
                    "ocr",
                    0.9
                )
            )
            .unwrap_err()
            .code(),
        "invalid_locator"
    );
    assert_eq!(
        store
            .add_anchor(
                &asset.id,
                AnchorDraft::new(Locator::Whole, "x", "ocr", 1.5)
            )
            .unwrap_err(),
        MediaError::InvalidConfidence
    );
    assert_eq!(
        store
            .add_anchor(
                &asset.id,
                AnchorDraft::new(Locator::Whole, "x", "  ", 0.5)
            )
            .unwrap_err(),
        MediaError::EmptyExtractor
    );
    store
        .add_anchor(
            &asset.id,
            AnchorDraft::new(
                Locator::ImageRegion {
                    x: 0,
                    y: 0,
                    width: 300,
                    height: 1000,
                },
                "left caption",
                "ocr",
                0.8,
            )
            .with_entities(vec!["姚明".to_string()]),
        )
        .unwrap();
    assert_eq!(store.anchor(&asset.id, &store.anchors(&asset.id).unwrap()[0].id).unwrap().entities, vec!["姚明"]);
}

#[test]
fn selector_parses_chinese_deixis_without_hijacking_plain_questions() {
    let image_selector = MediaSelector::parse("这张图左边的人是谁").unwrap();
    assert_eq!(image_selector.modalities, vec![Modality::Image]);
    assert_eq!(image_selector.ordinal, SelectorOrdinal::Latest);
    assert_eq!(
        image_selector.locator,
        Some(LocatorHint::ImageSide(Side::Left))
    );

    let table_selector = MediaSelector::parse("表格第三行第二列是多少").unwrap();
    assert_eq!(table_selector.modalities, vec![Modality::Table]);
    assert_eq!(table_selector.ordinal, SelectorOrdinal::Any);
    assert_eq!(
        table_selector.locator,
        Some(LocatorHint::TableCell { row: 3, column: 2 })
    );

    let audio_selector = MediaSelector::parse("刚才那段录音最后一句说了什么").unwrap();
    assert_eq!(audio_selector.modalities, vec![Modality::Audio]);
    assert_eq!(audio_selector.ordinal, SelectorOrdinal::Latest);
    assert_eq!(audio_selector.locator, Some(LocatorHint::LastSentence));

    assert_eq!(
        MediaSelector::parse("第二张图").unwrap().ordinal,
        SelectorOrdinal::Index(2)
    );
    assert_eq!(
        MediaSelector::parse("图2里的文字").unwrap().ordinal,
        SelectorOrdinal::Index(2)
    );
    assert_eq!(
        MediaSelector::parse("第三行").unwrap().locator,
        Some(LocatorHint::TableRow { row: 3 })
    );
    assert_eq!(
        MediaSelector::parse("第二句话").unwrap().locator,
        Some(LocatorHint::Sentence(2))
    );
    assert_eq!(
        MediaSelector::parse("第六张表").unwrap().modalities,
        vec![Modality::Table]
    );
    assert_eq!(
        MediaSelector::parse("第三页").unwrap().modalities,
        vec![Modality::Document]
    );
    assert!(MediaSelector::parse("知识图谱里有什么").is_none());
    assert!(MediaSelector::parse("世界上最高的山").is_none());
    assert_eq!(
        MediaSelector::parse("地图上的北京在哪里").unwrap().modalities,
        vec![Modality::Image]
    );
    assert_eq!(
        MediaSelector::parse("这张照片里的人是谁").unwrap().modalities,
        vec![Modality::Image]
    );
    for plain in [
        "第三个人是谁",
        "第一次见面是什么时候",
        "最后的答案是什么",
        "上一个问题还没回答",
        "当前天气怎么样",
        "这个怎么用",
        "图书馆在哪里",
        "这个意图是什么",
        "切换视图",
        "第三段文字讲了什么",
        "给我一张纸",
        "第二份礼物是什么",
        "这份菜单多少钱",
    ] {
        assert!(
            MediaSelector::parse(plain).is_none(),
            "普通问题被媒体选择器误吞：{plain}"
        );
    }
}

#[test]
fn selection_prefers_focus_and_clarifies_ambiguity() {
    let mut store = MediaStore::new(limits());
    let first = image(&mut store, b"first-image");
    let second = image(&mut store, b"second-image");
    store.add_anchor(&first.id, AnchorDraft::new(Locator::Whole, "第一张", "manual", 1.0)).unwrap();
    store.add_anchor(&second.id, AnchorDraft::new(Locator::Whole, "第二张", "manual", 1.0)).unwrap();

    let selector = MediaSelector::parse("图片").unwrap();
    let selection = store.select(&selector, None);
    let MediaSelection::Ambiguous { candidates } = selection else {
        panic!("expected ambiguity, got {}", selection.describe_zh());
    };
    assert_eq!(candidates.len(), 2);
    let prompt = store
        .select(&selector, None)
        .clarification_prompt()
        .unwrap();
    assert!(prompt.contains("1. 图片 1"));
    assert!(prompt.contains("2. 图片 2"));

    let focus = AssetRef::from_record(&first);
    let selected = store.select(&selector, Some(&focus));
    assert_eq!(selected.asset().unwrap().id, first.id);

    let latest = store.select(&MediaSelector::parse("这张图").unwrap(), Some(&focus));
    assert_eq!(latest.asset().unwrap().id, second.id);

    let indexed = MediaSelector::parse("第二张图").unwrap();
    assert_eq!(
        store.select(&indexed, None).asset().unwrap().id,
        second.id
    );
    assert!(matches!(
        store.select(&MediaSelector::parse("第九张图").unwrap(), None),
        MediaSelection::Missing { .. }
    ));
}

#[test]
fn table_and_side_locators_select_exact_anchor() {
    let mut store = MediaStore::new(limits());
    let table_asset = table(&mut store, b"row1,row2,row3");
    let cell = cell_anchor(&mut store, &table_asset, 3, 2, "1981");
    let image_asset = image(&mut store, b"image-with-left-region");
    let left = store
        .add_anchor(
            &image_asset.id,
            AnchorDraft::new(
                Locator::ImageRegion {
                    x: 0,
                    y: 0,
                    width: 300,
                    height: 1000,
                },
                "左边的人：姚明",
                "ocr",
                0.77,
            ),
        )
        .unwrap();

    let table_selection = store.select(
        &MediaSelector::parse("表格第三行第二列").unwrap(),
        None,
    );
    assert_eq!(table_selection.anchors(), std::slice::from_ref(&cell));

    let side_selection = store.select(&MediaSelector::parse("这张图左边").unwrap(), None);
    assert_eq!(side_selection.anchors(), std::slice::from_ref(&left));
    assert_eq!(
        left.focus_text(Some(&LocatorHint::ImageSide(Side::Left))),
        "左边的人：姚明"
    );

    let without = store.select(&MediaSelector::parse("表格第九行第二列").unwrap(), None);
    assert!(matches!(without, MediaSelection::AssetWithoutAnchor { .. }));
    assert!(without.clarification_prompt().is_none());
}

#[test]
fn support_lattice_covers_five_states_and_normalizes_noise() {
    let mut store = MediaStore::new(limits());
    let asset = table(&mut store, b"table-values");
    let anchor_1980 = cell_anchor(&mut store, &asset, 1, 1, "1980年");
    let anchor_1981 = cell_anchor(&mut store, &asset, 2, 1, "1981年");

    let corroborated = assess_support(
        "公司",
        "成立年份",
        &["１９８０年".to_string()],
        &[MediaClaim::from_anchor("1980年", &asset, &anchor_1980)],
    );
    assert_eq!(corroborated.state, SupportState::Corroborated);
    assert!(corroborated.render_zh().contains("一致"));

    let media_only = assess_support(
        "公司",
        "成立年份",
        &[],
        &[MediaClaim::from_anchor("1980年", &asset, &anchor_1980)],
    );
    assert_eq!(media_only.state, SupportState::MediaOnly);
    assert!(media_only.render_zh().contains("仅依据你提供的媒体"));

    let text_only = assess_support(
        "公司",
        "成立年份",
        &["1980年".to_string()],
        &[],
    );
    assert_eq!(text_only.state, SupportState::TextOnly);

    let conflicted = assess_support(
        "公司",
        "成立年份",
        &["1980年".to_string()],
        &[MediaClaim::from_anchor("1981年", &asset, &anchor_1981)],
    );
    assert_eq!(conflicted.state, SupportState::Conflicted);
    assert!(conflicted.primary_value().is_none());
    assert_eq!(conflicted.citations().len(), 1);
    assert!(conflicted.render_zh().contains("不一致"));

    let missing = assess_support("公司", "成立年份", &[], &[]);
    assert_eq!(missing.state, SupportState::Missing);
    assert!(!missing.answerable());

    let graph_conflict = assess_support(
        "公司",
        "成立年份",
        &["1980年".to_string(), "1981年".to_string()],
        &[],
    );
    assert_eq!(graph_conflict.state, SupportState::Conflicted);
    assert!(
        graph_conflict.render_zh().contains("图谱自身"),
        "{}",
        graph_conflict.render_zh()
    );

    let single_media_twice = assess_support(
        "公司",
        "成立年份",
        &[],
        &[
            MediaClaim::from_anchor("1980年", &asset, &anchor_1980),
            MediaClaim::from_anchor("１９８０年", &asset, &anchor_1980),
        ],
    );
    assert_eq!(single_media_twice.state, SupportState::MediaOnly);
    assert_eq!(single_media_twice.values[0].media.len(), 2);
}

#[test]
fn session_snapshot_has_no_bytes_and_restore_drops_deleted_assets() {
    let mut store = MediaStore::new(limits());
    let first = image(&mut store, b"session-one");
    let second = image(&mut store, b"session-two");
    let mut session = SessionMedia::new(2);
    session.remember(&first).unwrap();
    session.remember(&second).unwrap();
    assert_eq!(session.focus().unwrap().asset_id, second.id);

    let snapshot = session.snapshot();
    let json = serde_json::to_string(&snapshot).unwrap();
    assert!(!json.contains("session-one"));
    assert!(!json.contains("session-two"));
    assert!(json.contains(&first.content_hash));

    let mut image = ContextImage::new();
    session.install_into(&mut image).unwrap();
    image.seal();
    assert_eq!(image.section(MEDIA_SECTION).is_some(), true);

    let (restored, dropped) = SessionMedia::restore_from(&image, &store).unwrap();
    assert!(dropped.is_empty());
    assert_eq!(restored.refs(), session.refs());

    store.remove(&second.id);
    let (restored, dropped) = SessionMedia::restore_from(&image, &store).unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(dropped.len(), 1);
    assert_eq!(dropped[0].reason, DropReason::Missing);
    assert!(dropped[0].render_zh().contains("不可用"));

    let mut empty = ContextImage::new();
    empty.set_section(MEDIA_SECTION, serde_json::json!({"version": 1, "assets": []}));
    SessionMedia::new(2).install_into(&mut empty).unwrap();
    assert!(empty.section(MEDIA_SECTION).is_none());
}

#[test]
fn stale_generation_after_republish_is_unavailable() {
    let mut store = MediaStore::new(limits());
    let original = image(&mut store, b"generation-bytes");
    let reference = AssetRef::from_record(&original);
    assert!(reference.resolve(&store).is_ok());

    store.remove(&original.id);
    assert_eq!(
        reference.resolve(&store).unwrap_err(),
        UnavailableReason::Missing
    );
    let republished = image(&mut store, b"generation-bytes");
    assert_eq!(republished.id, original.id);
    assert_ne!(republished.sequence, original.sequence);
    assert_eq!(
        reference.resolve(&store).unwrap_err(),
        UnavailableReason::Stale {
            expected_sequence: original.sequence,
            actual_sequence: republished.sequence
        }
    );
    assert_eq!(
        reference.resolve(&store).unwrap_err().code(),
        "stale"
    );
}

#[test]
fn zero_capacity_session_rejects_new_focus_without_panicking() {
    let mut store = MediaStore::new(limits());
    let asset = image(&mut store, b"zero-capacity");
    let mut session = SessionMedia::new(0);
    assert_eq!(
        session.remember(&asset).unwrap_err(),
        MediaError::SessionCapacityExceeded { limit: 0 }
    );
    assert!(session.focus().is_none());
    assert!(!session.forget(&asset.id));
}

#[test]
fn media_turn_grounds_answer_with_citation() {
    let mut store = MediaStore::new(limits());
    let table_asset = table(&mut store, b"turn-table");
    let cell = cell_anchor(&mut store, &table_asset, 3, 2, "1981");
    let mut session = SessionMedia::new(4);
    let selector = MediaSelector::parse("表格第三行第二列").unwrap();
    let turn = ground_media_turn(
        &selector,
        &store,
        &mut session,
        "公司",
        "成立年份",
        &["1981".to_string()],
    );
    assert_eq!(turn.state(), Some(SupportState::Corroborated));
    assert!(turn.response.contains("一致"), "{}", turn.response);
    assert!(turn.response.contains("表格第 3 行第 2 列"), "{}", turn.response);
    assert_eq!(turn.citations().len(), 1);
    assert_eq!(turn.citations()[0].anchor_id, cell.id);
    assert!(turn.focus_updated);
    assert_eq!(session.focus().unwrap().asset_id, table_asset.id);
}

#[test]
fn media_turn_clarifies_instead_of_guessing() {
    let mut store = MediaStore::new(limits());
    let first = image(&mut store, b"turn-first");
    let second = image(&mut store, b"turn-second");
    store
        .add_anchor(&first.id, AnchorDraft::new(Locator::Whole, "第一张图", "manual", 1.0))
        .unwrap();
    store
        .add_anchor(&second.id, AnchorDraft::new(Locator::Whole, "第二张图", "manual", 1.0))
        .unwrap();
    let mut session = SessionMedia::new(4);
    let selector = MediaSelector::parse("图片").unwrap();
    let turn = ground_media_turn(
        &selector,
        &store,
        &mut session,
        "人物",
        "身份",
        &[],
    );
    assert_eq!(turn.state(), None);
    assert!(turn.clarification.is_some());
    assert!(turn.response.contains("编号确认"), "{}", turn.response);
    assert!(!turn.focus_updated);
    assert!(session.is_empty());
}

#[test]
fn media_turn_uses_latest_asset_and_discloses_media_only() {
    let mut store = MediaStore::new(limits());
    let first = image(&mut store, b"turn-latest-first");
    let second = image(&mut store, b"turn-latest-second");
    store
        .add_anchor(&first.id, AnchorDraft::new(Locator::Whole, "第一张图的内容", "manual", 1.0))
        .unwrap();
    store
        .add_anchor(&second.id, AnchorDraft::new(Locator::Whole, "第二张图的内容", "manual", 1.0))
        .unwrap();
    let mut session = SessionMedia::new(4);
    session.remember(&first).unwrap();
    let selector = MediaSelector::parse("这张图").unwrap();
    let turn = ground_media_turn(
        &selector,
        &store,
        &mut session,
        "人物",
        "身份",
        &[],
    );
    assert_eq!(turn.state(), Some(SupportState::MediaOnly));
    assert!(turn.response.contains("仅依据你提供的媒体"), "{}", turn.response);
    assert!(turn.response.contains("第二张图的内容"), "{}", turn.response);
    assert!(turn.focus_updated);
    assert_eq!(session.focus().unwrap().asset_id, second.id);
}

#[test]
fn last_sentence_focus_and_persisted_root() {
    let temporary = tempfile::tempdir().unwrap();
    let mut store = MediaStore::with_root(temporary.path(), limits());
    let audio = store
        .publish(Modality::Audio, "audio/wav", "录音 1", b"audio-bytes")
        .unwrap();
    let anchor = store
        .add_anchor(
            &audio.id,
            AnchorDraft::new(
                Locator::AudioSpan {
                    start_ms: 0,
                    end_ms: 5_000,
                },
                "第一句。最后一句？",
                "asr",
                0.6,
            ),
        )
        .unwrap();
    assert_eq!(
        anchor.focus_text(Some(&LocatorHint::LastSentence)),
        "最后一句？"
    );
    assert_eq!(
        anchor.focus_text(Some(&LocatorHint::Sentence(1))),
        "第一句。"
    );
    let persisted = temporary
        .path()
        .join(format!("{}.bin", audio.content_hash));
    assert_eq!(std::fs::read(&persisted).unwrap(), b"audio-bytes");

    std::fs::write(&persisted, b"tampered").unwrap();
    let mut other = MediaStore::with_root(temporary.path(), limits());
    assert_eq!(
        other
            .publish(Modality::Audio, "audio/wav", "录音 1", b"audio-bytes")
            .unwrap_err()
            .code(),
        "persisted_hash_mismatch"
    );
}

#[test]
fn table_grid_parses_rectangular_csv_and_builds_candidates() {
    let grid = TableGrid::parse_csv(
        "公司，成立年份，总部\n甲公司, 1981 ,北京\n乙公司,1999,上海\n",
    )
    .unwrap();
    assert_eq!(grid.row_count(), 3);
    assert_eq!(grid.column_count(), 3);
    assert_eq!(grid.headers(), ["公司", "成立年份", "总部"]);
    assert_eq!(grid.cell(2, 2), Some("1981"));
    assert_eq!(grid.cell(0, 1), None);
    assert_eq!(grid.cell(4, 1), None);

    let candidate = grid.candidate(2, 2).unwrap();
    assert_eq!(candidate.subject, "甲公司");
    assert_eq!(candidate.predicate, "成立年份");
    assert_eq!(candidate.object, "1981");
    assert!(candidate.render_zh().contains("表格第 2 行第 2 列"));
    assert_eq!(grid.candidates().len(), 4);
}

#[test]
fn table_grid_rejects_ragged_empty_and_oversized_tables() {
    assert_eq!(
        TableGrid::from_rows(vec![
            vec!["公司".into(), "年份".into()],
            vec!["甲公司".into()],
        ])
        .unwrap_err()
        .code(),
        "invalid_table"
    );
    assert_eq!(
        TableGrid::from_rows(vec![
            vec!["".into(), "年份".into()],
            vec!["甲公司".into(), "1981".into()],
        ])
        .unwrap_err()
        .code(),
        "invalid_table"
    );
    let columns = MAX_TABLE_CELLS / 2 + 1;
    let wide = vec![vec!["表头".to_string(); columns]; 2];
    assert_eq!(
        TableGrid::from_rows(wide).unwrap_err().code(),
        "table_too_large"
    );
}

#[test]
fn table_ingest_feeds_cell_selection_and_support() {
    let mut store = MediaStore::new(limits());
    let asset = table(&mut store, b"company-grid");
    let grid = TableGrid::parse_csv(
        "公司,成立年份,总部\n甲公司,1981,北京\n乙公司,1999,上海\n",
    )
    .unwrap();
    let anchors = store.ingest_table(&asset.id, &grid).unwrap();
    assert_eq!(anchors.len(), 4);
    assert_eq!(anchors[0].entities, vec!["甲公司".to_string()]);

    let selector = MediaSelector::parse("表格第三行第二列").unwrap();
    let mut session = SessionMedia::new(4);
    let turn = ground_media_turn(
        &selector,
        &store,
        &mut session,
        "乙公司",
        "成立年份",
        &["1999".to_string()],
    );
    assert_eq!(turn.state(), Some(SupportState::Corroborated));
    assert!(turn.response.contains("1999"), "{}", turn.response);
    assert_eq!(turn.citations()[0].asset_id, asset.id);
}

#[test]
fn table_ingest_is_atomic_checked_and_modality_aware() {
    let mut custom = limits();
    custom.max_anchors_per_asset = 2;
    let mut store = MediaStore::new(custom);
    let asset = table(&mut store, b"capacity-grid");
    let grid = TableGrid::parse_csv(
        "公司,成立年份,总部\n甲公司,1981,北京\n乙公司,1999,上海\n",
    )
    .unwrap();
    assert_eq!(
        store.ingest_table(&asset.id, &grid).unwrap_err().code(),
        "anchor_capacity_exceeded"
    );
    assert!(store.anchors(&asset.id).unwrap().is_empty());

    let image = image(&mut store, b"not-a-table");
    assert_eq!(
        store.ingest_table(&image.id, &grid).unwrap_err().code(),
        "locator_modality_mismatch"
    );
}

#[test]
fn labels_and_ordinals_are_per_modality_without_shift_after_deletion() {
    let mut store = MediaStore::new(limits());
    let first_image = image(&mut store, b"ordinal-image-1");
    let grid = table(&mut store, b"ordinal-table-1");
    let second_image = image(&mut store, b"ordinal-image-2");
    assert_eq!(first_image.label, "图片 1");
    assert_eq!(grid.label, "表格 1");
    assert_eq!(second_image.label, "图片 2");
    assert_eq!(first_image.ordinal, 1);
    assert_eq!(grid.ordinal, 1);
    assert_eq!(second_image.ordinal, 2);

    let selector = MediaSelector::parse("图2").unwrap();
    let MediaSelection::Selected { asset, .. } = store.select(&selector, None) else {
        panic!("图2 应选中第二张图片");
    };
    assert_eq!(asset.id, second_image.id);

    let selector = MediaSelector::parse("第一张表格").unwrap();
    let MediaSelection::Selected { asset, .. } = store.select(&selector, None) else {
        panic!("第一张表格应选中表格 1");
    };
    assert_eq!(asset.id, grid.id);

    assert!(store.remove(&second_image.id));
    assert!(matches!(
        store.select(&MediaSelector::parse("图2").unwrap(), None),
        MediaSelection::Missing { .. }
    ));
    assert!(matches!(
        store.select(&MediaSelector::parse("第二张图").unwrap(), None),
        MediaSelection::Missing { .. }
    ));
}
