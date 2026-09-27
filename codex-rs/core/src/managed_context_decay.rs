//! Opt-in Metalanguage directory guidance. Audit history is append-only; only
//! owned message identities are excluded from model sampling input.

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::num::NonZeroU64;
use std::process::Stdio;
use std::sync::Arc;

use codex_protocol::models::ResponseItem;
use codex_protocol::ResponseItemId;
use serde::Deserialize;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::context::ContextualUserFragment;
use crate::context::HookAdditionalContext;
use crate::session::session::Session;
use crate::session::step_context::StepContext;

pub(crate) const DEFER: &str = "Local context activated; tool was not executed.";

/// One instance per live managed rollout, never shared between sessions.
#[derive(Debug)]
pub struct ManagedContextDecay {
    lifetime: NonZeroU64,
    resolver: Vec<String>,
    state: Mutex<State>,
    audit: Option<std::sync::Mutex<File>>,
}

// Config clones share this runtime owner; equal settings do not make two live
// owners interchangeable.
impl PartialEq for ManagedContextDecay {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

#[derive(Debug, Default, Clone)]
struct State {
    submitted: u64,
    sequence: u64,
    guidance_revision: u64,
    current: Option<u64>,
    epochs: BTreeMap<u64, Epoch>,
    active: BTreeMap<String, ActiveGuide>,
    owned: HashSet<ResponseItemId>,
}

#[derive(Debug, Clone, Copy)]
struct Epoch {
    id: u64,
    ordinal: u64,
    guidance_revision: u64,
    acknowledged: bool,
    blocked_at_sequence: Option<u64>,
}

#[derive(Debug, Clone)]
struct ActiveGuide {
    digest: String,
    id: ResponseItemId,
    expires_after: u64,
}

#[derive(Debug, Deserialize)]
struct Resolution {
    schema: u32,
    guides: Vec<Guide>,
    observations: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct Guide {
    path: String,
    digest: String,
    content: String,
}

impl State {
    fn begin(&mut self, existing: Option<u64>) -> Result<Epoch, String> {
        if let Some(id) = existing {
            return (self.current == Some(id))
                .then(|| self.epochs.get(&id).copied())
                .flatten()
                .ok_or_else(|| "decay request belongs to a superseded inference".into());
        }
        self.sequence = self.sequence.checked_add(1).ok_or("decay epoch overflow")?;
        let epoch = Epoch {
            id: self.sequence,
            ordinal: self.submitted.checked_add(1).ok_or("decay step overflow")?,
            guidance_revision: self.guidance_revision,
            acknowledged: false,
            blocked_at_sequence: None,
        };
        self.current = Some(epoch.id);
        self.epochs.insert(epoch.id, epoch);
        Ok(epoch)
    }

    fn acknowledge(&mut self, id: u64) -> Result<(), String> {
        if self.current != Some(id) {
            return Err("decay acknowledgement belongs to a superseded inference".into());
        }
        let epoch = self.epochs.get(&id).copied()
            .ok_or("decay acknowledgement has no inference")?;
        if !epoch.acknowledged {
            self.submitted = epoch.ordinal;
            let mut expired = false;
            self.active.retain(|path, guide| {
                let retained = guide.expires_after >= epoch.ordinal;
                if !retained {
                    expired = true;
                    tracing::info!(target: "metalanguage::agents_decay", action = "expired", path, step = epoch.ordinal);
                }
                retained
            });
            if expired {
                self.guidance_revision = self.guidance_revision.checked_add(1)
                    .ok_or("decay guidance revision overflow")?;
            }
            let current = self.epochs.get_mut(&id)
                .ok_or("decay acknowledgement has no inference")?;
            current.acknowledged = true;
            // The response was sampled from the post-expiry request view.
            current.guidance_revision = self.guidance_revision;
        }
        Ok(())
    }

    fn can_observe(&self, id: u64) -> bool {
        self.epochs.get(&id).is_some_and(|epoch| {
            epoch.acknowledged
                && (epoch.guidance_revision == self.guidance_revision
                    || epoch.blocked_at_sequence == Some(self.sequence))
        })
    }

    fn permitted(&self, id: u64) -> bool {
        self.epochs.get(&id).is_some_and(|epoch| {
            epoch.acknowledged
                && epoch.blocked_at_sequence.is_none()
                && epoch.guidance_revision == self.guidance_revision
        })
    }

    fn filter(&self, input: &mut Vec<ResponseItem>, ordinal: u64) {
        let retained: HashSet<_> = self.active.values()
            .filter(|guide| guide.expires_after >= ordinal)
            .map(|guide| &guide.id).collect();
        input.retain(|item| {
            !matches!(item, ResponseItem::Message { role, .. } if role == "developer")
                || item.id().is_none_or(|id| !self.owned.contains(id) || retained.contains(id))
        });
    }

    fn observe(
        &mut self,
        epoch_id: u64,
        guides: Vec<Guide>,
        lifetime: NonZeroU64,
    ) -> Result<Vec<ResponseItem>, String> {
        if !self.can_observe(epoch_id) {
            return Err("decay access belongs to stale guidance".into());
        }
        let deadline = self.submitted.checked_add(lifetime.get())
            .ok_or("decay expiry overflow")?;
        let mut additions = Vec::new();
        for guide in guides {
            if let Some(active) = self.active.get_mut(&guide.path)
                && active.digest == guide.digest
            {
                active.expires_after = deadline;
                tracing::info!(target: "metalanguage::agents_decay", action = "renewed", path = guide.path, step = self.submitted, expires_after = deadline);
                continue;
            }
            let revised = self.active.contains_key(&guide.path);
            // Identity is generated here, never recovered by matching the body.
            let id = ResponseItemId::new("msg");
            let path_attribute = guide.path.replace('&', "&amp;")
                .replace('<', "&lt;").replace('>', "&gt;")
                .replace('"', "&quot;").replace('\'', "&#x27;")
                .replace('\t', "&#9;").replace('\n', "&#10;").replace('\r', "&#13;");
            let mut item = ContextualUserFragment::into(HookAdditionalContext::new(
                format!("<AGENTS_MD path=\"{path_attribute}\">\n{}\n</AGENTS_MD>", guide.content.trim_end()),
            ));
            item.set_id(Some(id.clone()));
            self.owned.insert(id.clone());
            tracing::info!(target: "metalanguage::agents_decay", action = if revised { "revised" } else { "added" }, path = guide.path, digest = guide.digest, message_id = id.as_str(), step = self.submitted, expires_after = deadline);
            self.active.insert(guide.path, ActiveGuide { digest: guide.digest, id, expires_after: deadline });
            additions.push(item);
        }
        if !additions.is_empty() {
            self.guidance_revision = self.guidance_revision.checked_add(1)
                .ok_or("decay guidance revision overflow")?;
            let sequence = self.sequence;
            let epoch = self.epochs.get_mut(&epoch_id)
                .ok_or("decay access has no inference")?;
            epoch.blocked_at_sequence = Some(sequence);
        }
        Ok(additions)
    }
}

impl ManagedContextDecay {
    pub fn new(lifetime: NonZeroU64, resolver: Vec<String>) -> Result<Self, String> {
        if resolver.first().is_none_or(|program| program.is_empty()) {
            return Err("directory AGENTS decay requires a resolver command".into());
        }
        Ok(Self { lifetime, resolver, state: Mutex::new(State::default()), audit: None })
    }

    /// Host-owned audit destination, outside the rollout's writable roots.
    pub fn with_audit_file(mut self, file: File) -> Self {
        self.audit = Some(std::sync::Mutex::new(file));
        self
    }

    fn audit(&self, value: Value) -> Result<(), String> {
        if let Some(file) = &self.audit {
            let mut file = file.lock().map_err(|_| "decay audit lock poisoned")?;
            serde_json::to_writer(&mut *file, &value).map_err(|error| error.to_string())?;
            file.write_all(b"\n").and_then(|()| file.flush()).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Capture the guidance revision used by this request. Merely starting a
    /// newer inference does not stale older yielded programs; delivery or an
    /// acknowledged expiry advances the revision instead.
    pub(crate) async fn snapshot(&self, sess: &Session, step: &StepContext) -> Result<Vec<ResponseItem>, String> {
        let mut state = self.state.lock().await;
        let epoch = state.begin(step.managed_context_epoch.get().copied())?;
        let _ = step.managed_context_epoch.set(epoch.id);
        let mut input = sess.clone_history().await.for_prompt(&step.settings.model_info.input_modalities);
        let before = input.len();
        let original_ids: HashSet<_> = input.iter().filter_map(|item| item.id().cloned()).collect();
        state.filter(&mut input, epoch.ordinal);
        let retained_ids: HashSet<_> = input.iter().filter_map(|item| item.id().cloned()).collect();
        let mut removed: Vec<_> = original_ids.difference(&retained_ids).collect();
        removed.sort();
        self.audit(serde_json::json!({"event": "request_snapshot", "epoch": epoch.id,
            "step": epoch.ordinal, "guidance_revision": epoch.guidance_revision,
            "excluded_managed_ids": removed,
            "retained_managed_ids": state.active.values().filter(|guide| retained_ids.contains(&guide.id)).map(|guide| &guide.id).collect::<Vec<_>>() }))?;
        tracing::info!(target: "metalanguage::agents_decay", action = "request", step = epoch.ordinal, epoch = epoch.id, excluded_blocks = before - input.len(), active_blocks = state.active.len());
        Ok(input)
    }

    pub(crate) async fn acknowledge(&self, step: &StepContext) -> Result<(), String> {
        let id = *step.managed_context_epoch.get().ok_or("decay request has no epoch")?;
        let mut state = self.state.lock().await;
        let retry = state.epochs.get(&id).is_some_and(|epoch| epoch.acknowledged);
        state.acknowledge(id)?;
        let guidance_revision = state.epochs.get(&id)
            .map(|epoch| epoch.guidance_revision)
            .ok_or("decay acknowledgement has no inference")?;
        self.audit(serde_json::json!({"event": "response_created", "epoch": id,
            "step": state.submitted, "guidance_revision": guidance_revision, "retry": retry}))
    }

    /// Pre calls from a program latched in the current admission sequence keep
    /// resolving so a compound/parallel A+B frontier can accumulate without
    /// permitting a same-program retry. Older unblocked programs remain admissible only
    /// while their captured guidance revision is still current. Late post
    /// results from a stale or latched program cannot revive context.
    pub(crate) async fn access(
        &self,
        sess: &Arc<Session>,
        step: &StepContext,
        payload: Option<Value>,
        post: bool,
    ) -> Result<bool, String> {
        let mut state = self.state.lock().await;
        let Some(id) = step.managed_context_epoch.get().copied() else { return Ok(false) };
        if !state.can_observe(id) {
            self.audit(serde_json::json!({"event": "stale_access", "epoch": id,
                "origin_guidance_revision": state.epochs.get(&id).map(|epoch| epoch.guidance_revision),
                "current_guidance_revision": state.guidance_revision, "post": post}))?;
            return Ok(false);
        }
        if let Some(payload) = payload {
            let resolution = self.resolve(payload).await?;
            let mut next = state.clone();
            let additions = next.observe(id, resolution.guides, self.lifetime)?;
            self.audit(serde_json::json!({"event": "access_prepared", "epoch": id,
                "post": post, "observations": resolution.observations,
                "guidance_revision": next.guidance_revision,
                "added_managed_ids": additions.iter().filter_map(|item| item.id()).collect::<Vec<_>>(),
                "deferred": !next.permitted(id),
                "active": next.active.iter().map(|(path, guide)| serde_json::json!({
                    "path": path, "digest": guide.digest, "id": guide.id, "expires_after": guide.expires_after
                })).collect::<Vec<_>>() }))?;
            // Register ownership before the cancellable history write. If it
            // is interrupted, any partially recorded additions stay excluded;
            // activation/deadlines commit only after publication completes.
            state.owned.extend(additions.iter().filter_map(|item| item.id().cloned()));
            // No partial publication: snapshot and other accesses use this same
            // lock. Audit history keeps the original items after expiry.
            sess.record_conversation_items(&step.turn, &step.settings.model_info, &additions).await;
            *state = next;
            self.audit(serde_json::json!({"event": "access_committed", "epoch": id, "post": post}))?;
            tracing::info!(target: "metalanguage::agents_decay", action = if post { "post_access" } else { "pre_access" }, epoch = id, observations = %serde_json::json!(resolution.observations), deferred = !state.permitted(id));
        }
        Ok(state.permitted(id))
    }

    async fn resolve(&self, payload: Value) -> Result<Resolution, String> {
        let mut child = Command::new(&self.resolver[0]).args(&self.resolver[1..])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .kill_on_drop(true).spawn().map_err(|error| format!("start directory AGENTS resolver: {error}"))?;
        let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
        let mut stdin = child.stdin.take().ok_or("directory AGENTS resolver has no stdin")?;
        stdin.write_all(&bytes).await.map_err(|error| format!("write directory AGENTS resolver input: {error}"))?;
        drop(stdin);
        let output = child.wait_with_output().await.map_err(|error| format!("wait for directory AGENTS resolver: {error}"))?;
        if !output.status.success() {
            return Err(format!("directory AGENTS resolver failed ({})", output.status));
        }
        let result: Resolution = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("invalid directory AGENTS resolver response: {error}"))?;
        if result.schema != 1 {
            return Err("unsupported directory AGENTS resolver schema".into());
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "managed_context_decay_tests.rs"]
mod tests;
