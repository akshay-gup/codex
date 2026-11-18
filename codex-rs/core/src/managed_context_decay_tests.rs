use super::*;
use pretty_assertions::assert_eq;

fn guide(path: &str, digest: &str) -> Guide {
    Guide { path: path.into(), digest: digest.into(), content: "same guide text".into() }
}

fn step(state: &mut State) -> Epoch {
    let epoch = state.begin(None).unwrap();
    state.acknowledge(epoch.id).unwrap();
    epoch
}

fn message(text: &str) -> ResponseItem {
    ContextualUserFragment::into(HookAdditionalContext::new(codex_hooks::HookContext::harness(text.to_string())))
}

#[test]
fn guidance_labels_escaped_source_without_changing_body() {
    let mut state = State::default();
    let epoch = step(&mut state);
    let additions = state.observe(epoch.id, vec![Guide {
        path: "/a&\"'<b>\t\n\r/AGENTS.md".into(),
        digest: "v1".into(),
        content: "  <literal>& \"quoted\"\nsecond line  \n".into(),
    }], NonZeroU64::new(2).unwrap()).unwrap();
    let mut expected = message(
        "<AGENTS_MD path=\"/a&amp;&quot;&#x27;&lt;b&gt;&#9;&#10;&#13;/AGENTS.md\">\n  <literal>& \"quoted\"\nsecond line\n</AGENTS_MD>",
    );
    expected.set_id(additions[0].id().cloned());
    assert_eq!(additions, vec![expected]);
}

#[test]
fn k_one_and_two_grant_k_actual_future_inferences() {
    for k in [1, 2] {
        let mut state = State::default();
        let first = step(&mut state);
        let additions = state.observe(first.id, vec![guide("a", "v1")], NonZeroU64::new(k).unwrap()).unwrap();
        assert!(!state.permitted(first.id));
        for ordinal in 2..=k + 1 {
            let epoch = state.begin(None).unwrap();
            assert_eq!(epoch.ordinal, ordinal);
            let mut request = additions.clone();
            state.filter(&mut request, epoch.ordinal);
            assert_eq!(request, additions);
            state.acknowledge(epoch.id).unwrap();
        }
        let epoch = state.begin(None).unwrap();
        let mut request = additions;
        state.filter(&mut request, epoch.ordinal);
        assert!(request.is_empty());
        state.acknowledge(epoch.id).unwrap();
        assert!(state.active.is_empty());
    }
}

#[test]
fn renewal_retains_position_and_compound_access_does_not_age() {
    let mut state = State::default();
    let first = step(&mut state);
    let k = NonZeroU64::new(1).unwrap();
    let mut history = vec![message("fixed root")];
    history.extend(state.observe(first.id, vec![guide("a", "v1"), guide("b", "v1")], k).unwrap());
    history.push(message("copied guide text"));
    let original = history.clone();
    for _ in 0..3 {
        assert!(state.observe(first.id, vec![guide("a", "v1")], k).unwrap().is_empty());
        assert_eq!(state.submitted, 1);
        assert!(!state.permitted(first.id));
    }
    let second = step(&mut state);
    assert!(state.observe(second.id, vec![guide("a", "v1")], k).unwrap().is_empty());
    assert!(state.permitted(second.id));
    state.filter(&mut history, 2);
    assert_eq!(history, original);
    let third = step(&mut state);
    state.filter(&mut history, third.ordinal);
    assert_eq!(history, vec![original[0].clone(), original[1].clone(), original[3].clone()]);
}

#[test]
fn identity_only_removal_revision_and_reentry() {
    let mut state = State::default();
    step(&mut state);
    let k = NonZeroU64::new(1).unwrap();
    let first_epoch = state.current.unwrap();
    let first = state.observe(first_epoch, vec![guide("a", "v1"), guide("b", "v1")], k).unwrap();
    let copy = message("<AGENTS_MD path=\"a\">\nsame guide text\n</AGENTS_MD>");
    let root = message("fixed root");
    let mut history = vec![root.clone(), first[0].clone(), copy.clone(), first[1].clone()];
    let epoch = step(&mut state);
    let revision = state.observe(epoch.id, vec![guide("a", "v2")], k).unwrap();
    assert!(!state.permitted(epoch.id));
    history.extend(revision.clone());
    state.filter(&mut history, epoch.ordinal);
    assert_eq!(history, vec![root.clone(), copy.clone(), first[1].clone(), revision[0].clone()]);
    step(&mut state);
    let expired = step(&mut state);
    state.filter(&mut history, expired.ordinal);
    assert_eq!(history, vec![root, copy]);
    let reentry = state.observe(expired.id, vec![guide("a", "v2")], k).unwrap();
    assert_ne!(reentry[0].id(), revision[0].id());
    assert!(!state.permitted(expired.id));
}

#[test]
fn failed_unacknowledged_request_does_not_age_and_transport_retry_counts_once() {
    let mut state = State::default();
    let failed = state.begin(None).unwrap();
    assert!(!state.permitted(failed.id));
    assert_eq!(state.begin(Some(failed.id)).unwrap().ordinal, 1);
    let next = state.begin(None).unwrap();
    assert_ne!(failed.id, next.id);
    assert_eq!(next.ordinal, 1);
    assert!(state.begin(Some(failed.id)).is_err());
    state.acknowledge(next.id).unwrap();
    state.acknowledge(next.id).unwrap();
    assert_eq!(state.begin(Some(next.id)).unwrap().ordinal, 1);
    assert_eq!(state.submitted, 1);
    assert_eq!(step(&mut state).ordinal, 2);
}

#[test]
fn expiry_preserves_assistant_copies_and_opaque_reasoning_even_with_reused_id() {
    let mut state = State::default();
    step(&mut state);
    let current = state.current.unwrap();
    let additions = state.observe(current, vec![guide("a", "v1")], NonZeroU64::new(1).unwrap()).unwrap();
    let mut assistant = additions[0].clone();
    if let ResponseItem::Message { role, .. } = &mut assistant {
        *role = "assistant".into();
    }
    let reasoning = ResponseItem::Reasoning {
        id: additions[0].id().cloned(),
        summary: Vec::new(),
        content: None,
        encrypted_content: Some("opaque reasoning remains unchanged".into()),
        internal_chat_message_metadata_passthrough: None,
    };
    let mut request = vec![additions[0].clone(), assistant.clone(), reasoning.clone()];
    state.filter(&mut request, 3);
    assert_eq!(request, vec![assistant, reasoning]);
}

#[test]
fn deferred_epoch_collects_other_scopes_but_never_reopens() {
    let mut state = State::default();
    let old = step(&mut state);
    let k = NonZeroU64::new(2).unwrap();
    state.observe(old.id, vec![guide("a", "v1")], k).unwrap();
    state.observe(old.id, vec![guide("b", "v1")], k).unwrap();
    state.observe(old.id, vec![guide("a", "v1")], k).unwrap();
    assert!(!state.permitted(old.id));
    let next = step(&mut state);
    assert!(state.permitted(next.id));
    assert!(!state.can_observe(old.id)); // late post fallback must be ignored
    assert_eq!(state.active.len(), 2);
}

#[test]
fn older_cell_continues_when_a_new_inference_does_not_change_guidance() {
    let mut state = State::default();
    let old = step(&mut state);
    let newer = step(&mut state);

    assert!(state.can_observe(old.id));
    assert!(state.permitted(old.id));
    assert!(state.permitted(newer.id));
    assert_eq!(old.guidance_revision, newer.guidance_revision);
}

#[test]
fn delivery_changes_revision_and_never_rebases_an_older_cell() {
    let mut state = State::default();
    let old = step(&mut state);
    let delivering = step(&mut state);
    let revision_before = state.guidance_revision;

    state.observe(
        delivering.id,
        vec![guide("nested", "v1")],
        NonZeroU64::new(2).unwrap(),
    ).unwrap();

    assert!(state.guidance_revision > revision_before);
    assert!(!state.can_observe(old.id));
    assert!(!state.permitted(old.id));
    assert!(!state.permitted(delivering.id));
    let reconsidered = step(&mut state);
    assert!(state.permitted(reconsidered.id));
    assert!(!state.permitted(delivering.id));
}

#[test]
fn older_cell_deferral_latches_but_collects_its_same_frontier() {
    let mut state = State::default();
    let old = step(&mut state);
    step(&mut state);
    let k = NonZeroU64::new(2).unwrap();

    state.observe(old.id, vec![guide("a", "v1")], k).unwrap();
    state.observe(old.id, vec![guide("b", "v1")], k).unwrap();
    assert!(!state.permitted(old.id));
    assert_eq!(state.active.len(), 2);

    step(&mut state);
    assert!(!state.can_observe(old.id));
    assert!(!state.permitted(old.id));
}

#[test]
fn acknowledged_expiry_stales_old_cells_but_raw_inference_age_does_not() {
    let mut state = State::default();
    let delivering = step(&mut state);
    state.observe(
        delivering.id,
        vec![guide("nested", "v1")],
        NonZeroU64::new(2).unwrap(),
    ).unwrap();
    let admitted = step(&mut state);
    assert!(state.permitted(admitted.id));

    let unchanged = step(&mut state);
    assert!(state.permitted(admitted.id));
    assert!(state.permitted(unchanged.id));

    let after_expiry = step(&mut state);
    assert!(!state.permitted(admitted.id));
    assert!(!state.permitted(unchanged.id));
    assert!(state.permitted(after_expiry.id));
}

#[test]
fn an_admitted_running_tool_is_not_retroactively_cancelled() {
    let mut state = State::default();
    let running = step(&mut state);
    let admitted_before_dispatch = state.permitted(running.id);
    let delivering = step(&mut state);
    state.observe(
        delivering.id,
        vec![guide("nested", "v1")],
        NonZeroU64::new(1).unwrap(),
    ).unwrap();

    assert!(admitted_before_dispatch);
    assert!(!state.permitted(running.id));
}

#[test]
fn interrupted_publication_is_excluded_without_consuming_activation() {
    let mut state = State::default();
    step(&mut state);
    let mut pending = state.clone();
    let current = pending.current.unwrap();
    let additions = pending.observe(current, vec![guide("a", "v1")], NonZeroU64::new(1).unwrap()).unwrap();
    state.owned.extend(additions.iter().filter_map(|item| item.id().cloned()));
    let mut request = additions;
    state.filter(&mut request, 2);
    assert!(request.is_empty());
    assert!(state.active.is_empty());
    let current = state.current.unwrap();
    assert_eq!(state.observe(current, vec![guide("a", "v1")], NonZeroU64::new(1).unwrap()).unwrap().len(), 1);
}

#[cfg(not(target_os = "windows"))]
#[tokio::test]
async fn concurrent_accesses_publish_once_and_stale_fallback_cannot_revive() {
    let (session, turn) = crate::session::tests::make_session_and_context().await;
    let session = Arc::new(session);
    let turn = Arc::new(turn);
    let first = StepContext::for_test(Arc::clone(&turn));
    let manager = ManagedContextDecay::new(NonZeroU64::new(1).unwrap(), vec![
        "python3".into(), "-c".into(),
        "import json,sys; p=json.load(sys.stdin); print(json.dumps({'schema':1,'guides':[{'path':p['path'],'digest':'v1','content':'guide'}],'observations':[]}))".into(),
    ]).unwrap();
    manager.snapshot(&session, &first).await.unwrap();
    manager.acknowledge(&first).await.unwrap();
    let (a, b) = tokio::join!(
        manager.access(&session, &first, Some(serde_json::json!({"path":"a"})), false),
        manager.access(&session, &first, Some(serde_json::json!({"path":"b"})), false),
    );
    assert!(matches!(a.unwrap(), PreToolUseHookResult::Blocked(message) if message == DEFER));
    assert!(matches!(b.unwrap(), PreToolUseHookResult::Blocked(message) if message == DEFER));
    assert_eq!(manager.state.lock().await.active.len(), 2);
    assert!(
        matches!(manager.access(&session, &first, None, false).await.unwrap(), PreToolUseHookResult::Blocked(message) if message == DEFER)
    );
    let second = StepContext::for_test(Arc::clone(&turn));
    let request = manager.snapshot(&session, &second).await.unwrap();
    manager.acknowledge(&second).await.unwrap();
    let ids = manager.state.lock().await.owned.clone();
    assert_eq!(request.iter().filter(|item| item.id().is_some_and(|id| ids.contains(id))).count(), 2);
    // The original cell's post result arrives after the next snapshot. Ignore
    // it even before a newer request has completed, without invoking resolver.
    assert!(
        matches!(manager.access(&session, &first, Some(serde_json::json!({"path":"late"})), true).await.unwrap(), PreToolUseHookResult::Blocked(message) if message == DEFER)
    );
    assert!(!manager.state.lock().await.active.contains_key("late"));
    // A current-epoch fallback is activated and becomes available next step.
    assert!(
        matches!(manager.access(&session, &second, Some(serde_json::json!({"path":"fallback"})), true).await.unwrap(), PreToolUseHookResult::Blocked(message) if message == DEFER)
    );
    let third = StepContext::for_test(turn);
    manager.snapshot(&session, &third).await.unwrap();
    manager.acknowledge(&third).await.unwrap();
    let state = manager.state.lock().await;
    assert_eq!(state.active.keys().cloned().collect::<Vec<_>>(), vec!["fallback".to_string()]);
}

#[test]
fn advisory_reports_expire_after_one_inference_without_staling_programs() {
    let mut state = State::default();
    let first = step(&mut state);
    let guide = state
        .observe(
            first.id,
            vec![guide("scope", "v1")],
            NonZeroU64::new(3).unwrap(),
        )
        .unwrap();
    let second = step(&mut state);
    let revision = state.guidance_revision;
    let report = || Guide {
        path: "notes.md".into(),
        digest: "bad".into(),
        content: "WRITE_REPORT".into(),
    };
    let additions = state.observe_reports(vec![report()]);
    assert_eq!(state.guidance_revision, revision);
    assert!(state.permitted(second.id));
    assert!(state.observe_reports(vec![report()]).is_empty());
    let copy = message("WRITE_REPORT");
    let mut history = vec![guide[0].clone(), additions[0].clone(), copy.clone()];
    state.filter(&mut history, 3);
    assert_eq!(history.len(), 3);
    let third = step(&mut state);
    let fourth = step(&mut state);
    state.filter(&mut history, fourth.ordinal);
    assert_eq!(history, vec![guide[0].clone(), copy]);
    assert_eq!(state.guidance_revision, revision);
    assert!(state.permitted(third.id));
    assert!(state.permitted(fourth.id));
}

#[cfg(not(target_os = "windows"))]
#[tokio::test]
async fn hard_denial_keeps_its_message_and_leaves_guidance_and_admission_unchanged() {
    let (session, turn) = crate::session::tests::make_session_and_context().await;
    let session = Arc::new(session);
    let step = StepContext::for_test(Arc::new(turn));
    let manager = ManagedContextDecay::new(NonZeroU64::new(2).unwrap(), vec![
        "python3".into(), "-c".into(),
        "import json,sys; json.load(sys.stdin); print(json.dumps({'schema':1,'guides':[{'path':'ignored','digest':'v1','content':'guide'}],'observations':[],'deny':'ROOT_FIXED'}))".into(),
    ]).unwrap();
    manager.snapshot(&session, &step).await.unwrap();
    manager.acknowledge(&step).await.unwrap();
    let result = manager
        .access(
            &session,
            &step,
            Some(serde_json::json!({})),
            /*post*/ false,
        )
        .await
        .unwrap();
    assert!(matches!(result, PreToolUseHookResult::Blocked(message) if message == "ROOT_FIXED"));
    let state = manager.state.lock().await;
    assert!(state.active.is_empty());
    assert_eq!(state.guidance_revision, 0);
    assert!(state.permitted(*step.managed_context_epoch.get().unwrap()));
}

#[test]
fn report_after_a_pending_snapshot_survives_until_the_next_eligible_inference() {
    let mut state = State::default();
    step(&mut state);
    let pending = state.begin(None).unwrap();
    let report = state.observe_reports(vec![Guide {
        path: "notes.md".into(),
        digest: "bad".into(),
        content: "late report".into(),
    }]);
    state.acknowledge(pending.id).unwrap();
    let next = state.begin(None).unwrap();
    let mut input = report.clone();
    state.filter(&mut input, next.ordinal);
    assert_eq!(input, report);
    state.acknowledge(next.id).unwrap();
    let expired = state.begin(None).unwrap();
    state.filter(&mut input, expired.ordinal);
    assert!(input.is_empty());
}

#[test]
fn failed_pending_request_does_not_give_a_late_report_an_extra_acknowledged_step() {
    let mut state = State::default();
    step(&mut state);
    state.begin(None).unwrap();
    let report = state.observe_reports(vec![Guide {
        path: "notes.md".into(),
        digest: "bad".into(),
        content: "late report".into(),
    }]);
    let replacement = state.begin(None).unwrap();
    let mut input = report.clone();
    state.filter(&mut input, replacement.ordinal);
    assert_eq!(input, report);
    state.acknowledge(replacement.id).unwrap();
    let next = state.begin(None).unwrap();
    state.filter(&mut input, next.ordinal);
    assert!(input.is_empty());
}
