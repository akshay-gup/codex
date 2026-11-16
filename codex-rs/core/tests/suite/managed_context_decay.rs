use std::fs;
use std::num::NonZeroU64;
use std::sync::Arc;

use anyhow::Result;
use codex_core::managed_context_decay::ManagedContextDecay;
use codex_features::Feature;
use codex_protocol::models::PermissionProfile;
use core_test_support::responses::*;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use tempfile::TempDir;
use test_case::test_case;

#[cfg(not(target_os = "windows"))]
#[test_case(false; "native_patch")]
#[test_case(true; "code_mode_nested_patch")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn write_report_is_advisory_for_one_inference_and_denial_does_not_latch(
    code_mode: bool,
) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let fixture = TempDir::new()?;
    let marker = fixture.path().join("executions");
    let resolver = fixture.path().join("resolve.py");
    fs::write(
        &resolver,
        r#"
import json, sys
p = json.load(sys.stdin)
command = p.get('tool_input', {}).get('command', '')
denied = p.get('hook_event_name') == 'PreToolUse' and 'ROOT_BLOCK' in command
reports = []
if p.get('hook_event_name') == 'PostToolUse' and p.get('tool_name') == 'apply_patch':
    reports = [{'path': '/notes.md', 'digest': 'v1', 'content': 'WRITE_REPORT'}]
print(json.dumps({'schema': 1, 'guides': [], 'observations': [], 'reports': reports, 'deny': 'ROOT_FIXED' if denied else ''}))
"#,
    )?;
    let denied_patch = "*** Begin Patch\n*** Add File: AGENTS.md\n+ROOT_BLOCK\n*** End Patch\n";
    let safe_patch =
        "*** Begin Patch\n*** Add File: nested/AGENTS.md\n+malformed prose\n*** End Patch\n";
    let command = format!("printf x >> {}", marker.display());
    let mut first = vec![ev_response_created("r1")];
    if code_mode {
        let code = format!(
            "try {{ text(await tools.apply_patch({denied_patch:?})); }} catch(e) {{ text(String(e)); }}\ntext(await tools.apply_patch({safe_patch:?}));\ntext(await tools.exec_command({{cmd:{command:?}}}));"
        );
        first.push(ev_custom_tool_call("edits", "exec", &code));
    } else {
        first.extend([
            ev_apply_patch_custom_tool_call("denied", denied_patch),
            ev_apply_patch_custom_tool_call("safe", safe_patch),
            ev_function_call(
                "after",
                "exec_command",
                &json!({"cmd": command}).to_string(),
            ),
        ]);
    }
    first.push(ev_completed("r1"));
    let next_command = format!("printf y >> {}", marker.display());
    let next_call = if code_mode {
        ev_custom_tool_call(
            "next",
            "exec",
            &format!("text(await tools.exec_command({{cmd:{next_command:?}}}));"),
        )
    } else {
        ev_function_call(
            "next",
            "exec_command",
            &json!({"cmd": next_command}).to_string(),
        )
    };
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(first),
            sse(vec![
                ev_response_created("r2"),
                next_call,
                ev_completed("r2"),
            ]),
            sse(vec![
                ev_response_created("r3"),
                ev_assistant_message("done", "done"),
                ev_completed("r3"),
            ]),
        ],
    )
    .await;
    let resolver_command = vec!["python3".into(), resolver.to_string_lossy().into_owned()];
    let mut builder = test_codex()
        .with_model("test-gpt-5.1-codex")
        .with_config(move |config| {
            if code_mode {
                let _ = config.features.enable(Feature::CodeMode);
            }
            config.developer_instructions = Some("FIXED_ROOT".into());
            config.project_doc_max_bytes = 0;
            config.managed_context_decay = Some(Arc::new(
                ManagedContextDecay::new(NonZeroU64::new(5).unwrap(), resolver_command).unwrap(),
            ));
        });
    let test = builder.build_with_auto_env(&server).await?;
    fs::write(test.workspace_path("AGENTS.md"), "fixed root file\n")?;
    test.submit_turn_with_permission_profile("Begin.", PermissionProfile::Disabled)
        .await?;
    assert_eq!(fs::read_to_string(marker)?, "xy");
    assert_eq!(
        fs::read_to_string(test.workspace_path("AGENTS.md"))?,
        "fixed root file\n"
    );
    assert_eq!(
        fs::read_to_string(test.workspace_path("nested/AGENTS.md"))?,
        "malformed prose\n"
    );
    let requests = responses.requests();
    assert_eq!(requests.len(), 3);
    let denial = if code_mode {
        requests[1].custom_tool_call_output("edits").to_string()
    } else {
        requests[1].custom_tool_call_output("denied").to_string()
    };
    assert!(denial.contains("ROOT_FIXED"));
    for (index, request) in requests.iter().enumerate() {
        let developers = request.message_input_texts("developer");
        assert_eq!(
            developers.iter().any(|text| text.contains("WRITE_REPORT")),
            index == 1
        );
        assert!(developers.iter().any(|text| text.contains("FIXED_ROOT")));
    }
    Ok(())
}

// Exercise native registry admission and actual request serialization. The
// fixture resolver deliberately returns the same text as a later tool output.
#[test_case(1; "k_one")]
#[test_case(2; "k_two")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decay_filters_only_managed_input_and_reentry_defers_again(lifetime: u64) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let fixture = TempDir::new()?;
    let scope = fixture.path().join("scope");
    fs::create_dir(&scope)?;
    let marker = scope.join("executions");
    let resolver = fixture.path().join("resolve.py");
    fs::write(&resolver, format!(r#"
import json, sys
p = json.load(sys.stdin)
guides = []
if p.get('tool_input', {{}}).get('workdir') == {scope:?}:
    guides = [{{'path': {path:?}, 'digest': 'v1', 'content': 'SCOPED_GUIDE'}}]
print(json.dumps({{'schema': 1, 'guides': guides, 'observations': []}}))
"#, scope = scope.to_string_lossy(), path = scope.join("AGENTS.md").to_string_lossy()))?;
    let command = format!("printf x >> {}; printf SCOPED_GUIDE", marker.display());
    let args = json!({"cmd": command, "workdir": scope, "max_output_tokens": 256}).to_string();
    let mut sequence = vec![
        sse(vec![ev_response_created("r1"), ev_function_call("enter", "exec_command", &args), ev_completed("r1")]),
        sse(vec![ev_response_created("r2"), ev_function_call("retry", "exec_command", &args), ev_completed("r2")]),
    ];
    for index in 0..lifetime {
        let id = format!("elsewhere-{index}");
        sequence.push(sse(vec![ev_response_created(&id), ev_assistant_message(&format!("copy-{index}"), "SCOPED_GUIDE"), ev_function_call(&id, "exec_command", &json!({"cmd": "printf root"}).to_string()), ev_completed(&id)]));
    }
    sequence.extend([
        sse(vec![ev_response_created("r4"), ev_function_call("reenter", "exec_command", &args), ev_completed("r4")]),
        sse(vec![ev_response_created("r5"), ev_function_call("retry2", "exec_command", &args), ev_completed("r5")]),
        sse(vec![ev_response_created("r6"), ev_assistant_message("done", "done"), ev_completed("r6")]),
    ]);
    let responses = mount_sse_sequence(&server, sequence).await;
    let resolver_command = vec!["python3".into(), resolver.to_string_lossy().into_owned()];
    let test = test_codex().with_model("test-gpt-5.1-codex").with_config(move |config| {
        config.developer_instructions = Some("FIXED_ROOT".into());
        config.managed_context_decay = Some(Arc::new(ManagedContextDecay::new(
            NonZeroU64::new(lifetime).unwrap(), resolver_command,
        ).unwrap()));
    }).build(&server).await?;
    test.submit_turn_with_permission_profile("Begin.", PermissionProfile::Disabled).await?;
    let requests = responses.requests();
    let labelled_guide = format!("<AGENTS_MD path=\"{}\">\nSCOPED_GUIDE\n</AGENTS_MD>", scope.join("AGENTS.md").display());
    assert_eq!(requests.len(), lifetime as usize + 5);
    assert_eq!(fs::read_to_string(marker)?, "xx");
    let expired = lifetime as usize + 2;
    for (index, request) in requests.iter().enumerate() {
        let present = index != 0 && index != expired;
        let developers = request.message_input_texts("developer");
        assert_eq!(developers.iter().any(|text| text.contains("SCOPED_GUIDE")), present);
        assert_eq!(developers.iter().any(|text| text == &labelled_guide), present);
        assert!(developers.iter().any(|text| text.contains("FIXED_ROOT")));
    }
    let activation = requests[1].input().into_iter().enumerate().find(|(_, item)| {
        item["role"] == "developer" && item.to_string().contains("SCOPED_GUIDE")
    }).expect("managed activation in sampled request");
    for request in &requests[2..expired] {
        assert_eq!(request.input()[activation.0], activation.1);
        assert_eq!(request.message_input_texts("developer").iter().filter(|text| text.contains("SCOPED_GUIDE")).count(), 1);
    }
    assert!(requests[1].function_call_output_text("enter").unwrap().contains("tool was not executed"));
    assert!(requests[expired + 1].function_call_output_text("reenter").unwrap().contains("tool was not executed"));
    assert!(requests[expired].function_call_output_text("retry").unwrap().contains("SCOPED_GUIDE"));
    assert!(requests[expired].inputs_of_type("message").iter().any(|item| item["role"] == "assistant" && item.to_string().contains("SCOPED_GUIDE")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decay_code_mode_caught_retry_cannot_bypass_same_inference_gate() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let fixture = TempDir::new()?;
    let marker = fixture.path().join("executions");
    let resolver = fixture.path().join("resolve.py");
    fs::write(&resolver, r#"
import json, sys
p = json.load(sys.stdin)
guides = [{'path': '/scope/AGENTS.md', 'digest': 'v1', 'content': 'SCOPED_GUIDE'}] if p.get('tool_name') == 'Bash' else []
print(json.dumps({'schema': 1, 'guides': guides, 'observations': []}))
"#)?;
    let command = serde_json::to_string(&format!("printf x >> {}", marker.display()))?;
    let first_code = format!("for (let i=0; i<2; i++) {{ try {{ text(await tools.exec_command({{cmd:{command}}})); }} catch(e) {{ text(String(e)); }} }}");
    let retry_code = format!("text(await tools.exec_command({{cmd:{command}}}));");
    let responses = mount_sse_sequence(&server, vec![
        sse(vec![ev_response_created("r1"), ev_custom_tool_call("enter", "exec", &first_code), ev_completed("r1")]),
        sse(vec![ev_response_created("r2"), ev_custom_tool_call("retry", "exec", &retry_code), ev_completed("r2")]),
        sse(vec![ev_response_created("r3"), ev_assistant_message("done", "done"), ev_completed("r3")]),
    ]).await;
    let resolver_command = vec!["python3".into(), resolver.to_string_lossy().into_owned()];
    let test = test_codex().with_model("test-gpt-5.1-codex").with_config(move |config| {
        let _ = config.features.enable(Feature::CodeMode);
        config.managed_context_decay = Some(Arc::new(ManagedContextDecay::new(
            NonZeroU64::new(1).unwrap(), resolver_command,
        ).unwrap()));
    }).build(&server).await?;
    test.submit_turn_with_permission_profile("Begin.", PermissionProfile::Disabled).await?;
    assert_eq!(fs::read_to_string(marker)?, "x");
    let requests = responses.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests[1].custom_tool_call_output("enter").to_string().contains("tool was not executed"));
    Ok(())
}

#[test_case(false; "unchanged_guidance_continues")]
#[test_case(true; "changed_guidance_blocks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decay_yielded_code_mode_cell_uses_its_original_guidance_revision(changed: bool) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let fixture = TempDir::new()?;
    let marker = fixture.path().join("old-cell-executed");
    let release = fixture.path().join("release");
    let resolver = fixture.path().join("resolve.py");
    fs::write(&resolver, r#"
import json, sys
p = json.load(sys.stdin)
guides = [{'path': '/scope/AGENTS.md', 'digest': 'v1', 'content': 'SCOPED_GUIDE'}] if p.get('tool_input', {}).get('command') == 'printf activate' else []
print(json.dumps({'schema': 1, 'guides': guides, 'observations': []}))
"#)?;
    let command = serde_json::to_string(&format!("printf x >> {}", marker.display()))?;
    let wait_command = serde_json::to_string(&format!("while [ ! -f {} ]; do sleep 0.05; done", release.display()))?;
    let first = format!(r#"
const pending = tools.exec_command({{cmd:{wait_command}, yield_time_ms:30000}});
text("waiting"); yield_control(); text(await pending);
try {{ text(await tools.exec_command({{cmd:{command}}})); }} catch(e) {{ text(String(e)); }}
"#);
    let second = if changed {
        r#"try { text(await tools.exec_command({cmd:"printf activate"})); } catch(e) { text(String(e)); }
text("changed guidance");"#
    } else {
        r#"text("unchanged guidance");"#
    };
    let responses = mount_sse_sequence(&server, vec![
        sse(vec![ev_response_created("r1"), ev_custom_tool_call("old", "exec", &first), ev_completed("r1")]),
        sse(vec![ev_response_created("r2"), ev_custom_tool_call("release", "exec", second), ev_completed("r2")]),
        sse(vec![ev_response_created("r3"), ev_function_call("join", "wait", &json!({"cell_id":"1", "yield_time_ms":10000}).to_string()), ev_completed("r3")]),
        sse(vec![ev_response_created("r4"), ev_assistant_message("done", "done"), ev_completed("r4")]),
    ]).await;
    let resolver_command = vec!["python3".into(), resolver.to_string_lossy().into_owned()];
    let test = test_codex().with_model("test-gpt-5.1-codex").with_config(move |config| {
        let _ = config.features.enable(Feature::CodeMode);
        config.managed_context_decay = Some(Arc::new(ManagedContextDecay::new(
            NonZeroU64::new(2).unwrap(), resolver_command,
        ).unwrap()));
    }).build(&server).await?;
    let release_after_new_inference = async {
        tokio::time::timeout(std::time::Duration::from_secs(20), async {
            while responses.requests().len() < 3 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }).await?;
        fs::write(&release, "ready")?;
        anyhow::Ok(())
    };
    tokio::try_join!(test.submit_turn_with_permission_profile("Begin.", PermissionProfile::Disabled), release_after_new_inference)?;
    let requests = responses.requests();
    assert_eq!(requests.len(), 4);
    assert!(requests[1].custom_tool_call_output("old").to_string().contains("Script running with cell ID 1"));
    let joined = requests[3].function_call_output("join").to_string();
    assert!(!joined.contains("release not received"));
    if changed {
        assert!(!marker.exists());
        assert!(joined.contains("tool was not executed"), "{joined}");
    } else {
        assert!(marker.exists(), "{joined}");
        assert_eq!(fs::read_to_string(marker)?, "x");
        assert!(!joined.contains("tool was not executed"));
    }
    Ok(())
}
