# Plan v2: Native macOS App

**Issue**: #94 · **Spec**: docs/specs/native-macos-app.md · **Branch**: native-macos-app · **Base**: main
**Mode**: Standard, vertically sliced. Rationale: SwiftUI shell + safety-sensitive subsystems (update, supervision) that need design-first, not TDD-first.

Revised after external review (see .context/attachments). Four required artifacts are now first-class: (A) service ownership/shutdown protocol, (B) feature-parity matrix, (C) secure update + release-publishing design, (D) early packaged-app tracer bullet + CI. Packaging moves to the front; the giant commit is split into vertical slices; the update design drops the hand-rolled swap.

## Grounding facts verified against the code

- `recover_stuck_meetings()` (backend/services/recovery.py) runs at service startup → orphaned `PROCESSING` meetings become `ERROR "Transcription interrupted by app restart"`. Reconciliation exists; **no backend change needed** for job recovery.
- `GET /api/meetings/{id}` → `MeetingDetail.metadata.job_id` (schemas.py:166,188). Reopen can resume polling from the tracked id even though `MeetingSummary` omits `job_id`.
- Service isolation (service spec BR-12/BR-19–22): bundled service uses its **own** Application Support data dir + ephemeral port; it never shares state with a legacy Terminal install. So the "shared data dir collision" is prevented by the service, not the app.
- Release today: `.github/workflows/release.yml` (ubuntu, semantic-release only) + `.releaserc.js` (`@semantic-release/github`, **no assets**). No macOS artifact is produced anywhere → must be added (finding #2).
- Overview tab computes trajectory/interruptions/latency **client-side** from raw `AudioAnalysis` fields (overview-viewer.js:18,19,20) — analytical derivations, not formatting (finding #5).
- Service exposes **no shutdown endpoint** (routers/service.py) → shutdown is signal-based (finding #10).

## Artifact A — Service ownership & lifecycle protocol

- **Ownership = the child we spawned.** The app ALWAYS launches its own service `Process` (never attaches to a pre-existing one) and holds the handle. It terminates only that child (BR-5). No CLI-service adoption, so no ambiguous ownership.
- **Discovery precedence:** the spawned child's **stdout `ready` handshake is authoritative** for port. `service.json` is used only if the handshake line never arrives, and is accepted **only if its `pid` matches the child we launched** (rejects stale files / PID reuse). A `service.json` written by a *different* pid is ignored.
- **Instance nonce (small backend change, in scope):** the app generates a per-launch nonce, passes it via env `MT_SERVICE_NONCE`; `service_main.py` echoes it in the handshake JSON and writes it into `service.json`. The app accepts port/health only when the nonce matches. Belt-and-suspenders against stale records. (One ~5-line edit to service_main.py + a test.) **The nonce is launch-coordination set by the spawning parent — not configuration — so it is exempt in spirit from service BR-21's "ignore ambient env" rule (it does not affect data dir, model, or behavior). When absent (dev `run.py` path, any non-app client) the service starts normally and echoes an empty nonce; app-side matching applies only when the app actually set one.**
- **Readiness:** ready when `GET /api/health` on the discovered port **connects successfully** (health returns `status:"ready"` unconditionally once HTTP is up; the real not-ready window is pre-HTTP, covered by the handshake).
- **Crash detection:** a termination handler on the child flips app state to `.failed`; UI offers Restart (BR-4, EC-3). In-flight job is lost — surfaced honestly; `recover_stuck_meetings()` cleans persisted state on the next start.
- **Clean shutdown protocol (BR-5, EC-5, finding #10):** on quit → `SIGTERM` to child → bounded wait (10 s) → `SIGKILL`; if the child already exited during the confirm dialog, no-op. Documented guarantee: partial audio/metadata of an interrupted job are reconciled by `recover_stuck_meetings()`, not left as live PROCESSING.
- **Guarded `service.json` cleanup (finding #5):** `service.json` is a single shared path, so on shutdown AND in the crash handler the app **re-reads it and deletes only if both `pid` and `nonce` still match the terminating child** — never deletes a record a restarted/other process may have overwritten in the interim.
- **Single instance (EC-9, finding #9):** `LSMultipleInstancesProhibited=true` + `applicationDidBecomeActive`/`reopen` handler that unhides/creates the main window (including the no-visible-window case). Verified by the tracer-bullet smoke test.

## Artifact B — Feature-parity matrix (vs web UI)

Full matrix lives in `docs/macos-app.md`. Each web interaction classified:
`API-data` (render as-is) · `present` (format in Swift, no logic) · `port` (deterministic client aggregation reproduced in Swift, with contract tests vs captured fixtures) · `excluded` (product sign-off).

| Web capability | API | Classification |
|---|---|---|
| Meeting list + status badges | `GET /meetings` | API-data |
| Upload (file/type/langs/speakers/preprocess/audio-analysis/context) + client validation | `POST /meetings` | API-data + `present` (extension/size checks mirror server) |
| Transcript tab, synced audio | `GET /meetings/{id}`, `/audio` | API-data + AVFoundation |
| Playback speed / seek / segment-click seek | client | `present` |
| Per-segment insights (emotion label, prosody bars, interaction ↺, word/tone-mismatch badge) | `AudioAnalysis` fields + `segment.text` | `present`/`port`: labels/bars are `present`; the mismatch **badge** is a 4-line display predicate (`primary_emotion ∈ {frustrated,uncertain}` ∧ segment text contains an agreement phrase) `port`ed to Swift with a fixture test. The **business** mismatch content in the analysis prompt stays server-side (`analysis_context.py`), preserving the thin-client boundary. Resolves OQ-5 for the badge. |
| Per-segment detected-language badge | `TranscriptSegment.language` | API-data |
| Degraded audio-analysis states (status/reason `failed`/`unavailable`, `emotion_unavailable`, `prosody_unavailable`, `dominant_speaker_limitation` banner) | `AudioAnalysis` fields | `present` — insight + Overview render degraded/unavailable, not just the happy path |
| **Overview**: energy trajectory, interruption summary, response latency | raw `emotions`/`interactions`/`segment_interactions` | **`port`** — reproduce the 3 deterministic aggregations in Swift + fixture contract tests (see Decision 3) |
| Plain Text | transcript | `present` |
| Analysis prompt | `GET /meetings/{id}/analysis-prompt` (assembled) | API-data |
| Speaker rename — meeting-wide | `PATCH /meetings/{id}` | API-data |
| Speaker rename — single segment | `PATCH /meetings/{id}/segments/speaker` | API-data |
| Title/type/context edit | `PATCH /meetings/{id}` | API-data |
| Delete meeting | `DELETE /meetings/{id}` | API-data (destructive confirm) |
| Cancel while PROCESSING | `POST /meetings/{id}/cancel` | API-data |
| Meeting Retry on ERROR | `POST /meetings/{id}/retry` | API-data |
| Recent speaker names, theme | UserDefaults | `present` (BR-11) |

## Artifact C — Secure update + release publishing

- **Adopt Sparkle** (SPM), not a hand-rolled swap. Removes the entire custom download/verify/quarantine/rollback surface (finding #1). **No programmatic quarantine clearing.**
  - **EdDSA-signed update enclosures** (NOT a signed feed — finding #5): each `<enclosure>` carries an `edSignature`; the app verifies it before install via `SUPublicEDKey`. We do **not** enable `SURequireSignedFeed`/feed signing (a separate, newer capability with its own config); the trust model is signed-archive-only, the standard Sparkle model. `SUFeedURL` → the appcast asset on GitHub Releases; Sparkle handles signature verification, atomic install, rollback-on-failure, and the not-in-/Applications case.
  - Only version metadata + asset URLs are fetched (BR-21). Update never touches user data (outside the bundle, BR-24, EC-10).
  - "Don't update mid-transcription": gate `SPUUpdaterDelegate` to defer when a job is active.
- **Release publishing (finding #2):** the macOS build runs **as a second job in `release.yml` (same run), not a tag-triggered workflow** — the `GITHUB_TOKEN` recursion guard (CLAUDE.md) means a semantic-release tag/release will NOT trigger any `on: push tags`/`on: release`/`workflow_run` workflow. Implementation: the existing `release` job (ubuntu) exposes an output `new_release_published` + `new_release_version` (semantic-release exec/env); a new `build-macos` job with `needs: release`, `if: needs.release.outputs.new_release_published == 'true'`, `runs-on: macos-14`, factored into a reusable `release-macos.yml` (`on: workflow_call`, version passed in). Steps: retrieve/build the PyInstaller service bundle → assemble `.app` → inject `CFBundleShortVersionString`/`CFBundleVersion` from the version input → `codesign --sign -` (ad-hoc) → zip → Sparkle `sign_update` (EdDSA) → generate/update `appcast.xml` → `gh release upload <version>` (zip + appcast) onto the release the same run created. No new PAT/secret needed for the trigger.
- **Signing & keys — verified distribution design (finding #4):**
  - **Pinned Sparkle:** `2.9.6` — verified via the GitHub API as the newest stable release (published 2026-08-17); it contains the symlink-validation security fix and the root-process privilege-escalation fix that landed across 2.9.2→2.9.6, so it supersedes the 2.9.2/2.9.5 partial fixes. Pinned exactly in Package.swift `.exact("2.9.6")`, not a range. **Dependency-update policy:** re-check the newest stable Sparkle at each app release and during any security advisory; never call a version "latest" without checking it that day; bump the exact pin deliberately with the reason recorded here.
  - **Ad-hoc + library validation:** the `.app` is ad-hoc signed (`codesign --sign -`). Sparkle warns ad-hoc signing can block framework/XPC loading when **library validation** is enabled. Mitigation: the app's entitlements set `com.apple.security.cs.disable-library-validation` (and hardened runtime is **off** for the ad-hoc/self-signed build). Documented as a self-signing consequence.
  - **Framework embedding/signing order (custom packaging):** `build_app.sh` copies `Sparkle.framework` into `Contents/Frameworks/` **preserving symlinks (`cp -RP`) and executable permissions**, then signs **inner→outer**: XPC services (`org.sparkle-project.*` in the framework) → `Autoupdate`/`Updater.app` → `Sparkle.framework` → the app. EdDSA appcast signature is the trust anchor, independent of codesign.
  - **Stable feed URL:** `SUFeedURL` = a fixed `https://github.com/nimblehq/audio-transcriber/releases/latest/download/appcast.xml` (the "latest" asset alias is stable across versions).
  - **Appcast preservation across releases (finding #4 — `generate_appcast` operates over an archive *directory*, not just an XML):** we do **not** re-run `generate_appcast` over historical archives (they aren't retained locally). Instead: **generate the new `<item>` with Sparkle's own tooling** (`generate_appcast` run over a directory containing *only* the new update archive, or `sign_update` feeding a Sparkle-emitted item template) so the entry carries correct namespace, `enclosure length`, `sparkle:version`/`shortVersionString`, MIME type, `edSignature`, and `sparkle:minimumSystemVersion` — **not** hand-written XML (finding #3 this round). Then splice the retained prior `<item>` entries **verbatim** (their signatures were computed at their release; enclosure URLs point at still-live prior releases). **Deltas disabled** (historical archives absent); cap retained entries at the last N (e.g. 10). **Validate before publish, not just with an XML parser:** every retained enclosure URL is reachable, and the **merged feed passes a real Sparkle check** — specifically the packaged **v1→v2 install test runs against the merged feed** so we know Sparkle actually accepts the new item.
    - **Pre-publication test transport (finding #2 this round — draft-release assets are NOT reachable via the normal unauthenticated release URL, so Sparkle can't fetch them from the draft):** the v1→v2 test serves the merged appcast + v2 archive from a **temporary localhost HTTP fixture** the test build accepts. The test build differs from the shipping build **only in `SUFeedURL`** — same signing keys, same packaging, same EdDSA verification — so the install path is representative. After the release is published, a lightweight post-publish step re-verifies the **real** enclosure URL is reachable. Any retained enclosure 404 or a failed install test fails the job (draft stays).
  - **Key materialization:** EdDSA **private key = GitHub Actions secret** (`SPARKLE_ED_PRIVATE_KEY`) imported for `sign_update`/`generate_appcast`; **public key = `SUPublicEDKey`** in Info.plist.
- **Draft-until-complete release (finding #7, corrected option name):** `@semantic-release/github` is configured **`draftRelease: true`** (verified — the option is `draftRelease`, not `draft`; a wrong key would be silently ignored and publish immediately). The release stays a **draft** until `build-macos` uploads the zip + appcast, then flips it published (`gh release edit <tag> --draft=false`). Workflow assertions (integration-level): (1) release is draft before upload; (2) zip + appcast exist and are downloadable; (3) only then publish; (4) a failed macOS job leaves it draft.
- **Recovery mechanism (finding #2 this round — a *fresh* workflow run won't work: semantic-release already pushed the tag, so the next run reports no new release, `new_release_published=false`, and `build-macos` stays gated off).** Two supported paths: **(a) Normal** — GitHub "**Re-run failed jobs**" on the original run, preserving the original `release` job outputs (version/tag), so `build-macos` re-executes against the existing draft. **(b) Manual** — `workflow_dispatch` on the reusable `release-macos.yml` with an explicit `tag`/`version` input; it **resolves the draft by tag, verifies the draft's target commit matches that tag's commit, and refuses to modify an unrelated or already-published release.** Neither path creates a duplicate tag/release.
- **Fault handling:** Sparkle's built-in staged validation + rollback; if the appcast/asset is unreachable the current version keeps working (BR-22).

## Artifact D — Two packaging tracer bullets, both before product UI

**D1 — Plumbing tracer bullet (Commit 1, runs anywhere).** Minimal SwiftUI window, `scripts/build_app.sh` assembling `Contents/{MacOS,Resources,Frameworks,Info.plist}` + entitlements + ad-hoc `codesign`, launching a **stub service** (tiny script printing the handshake + serving `/api/health`), handshake parse, health-poll, close/reopen window, quit + verify child termination, install under `/Applications`. **Proves ONLY (renamed per finding #6):** child-process plumbing, stdout-handshake parsing, resource lookup, AppKit lifecycle, single-instance. It does **NOT** prove: a self-contained service without system Python, PyInstaller resource resolution, native-dependency loading, real app size/startup latency, or download-quarantine Gatekeeper behavior (a CI/ad-hoc app carries no browser quarantine attribute).

**D0 — Real-service packaging milestone (Commit 1b, before product UI; maintainer/arm64).** Per docs/packaging.md the real bundle is built on an Apple-Silicon machine (multi-GB torch/whisperx/pyannote via `MeetingTranscriber.spec`). This milestone: build the **actual** PyInstaller service, embed it in the `.app`, launch it, and run **health + provisioning smoke** (`/api/health`, `POST /provisioning/token` empty, `POST /provisioning/models` Whisper-only to completion, one clean-machine transcription) — retiring the finding-#1 risk that the hardest artifact is hand-waved. Outputs a documented clean-machine checklist. **This cannot run in the dev CLT env or cheaply in GitHub CI** (asset size + build time) — see Decision 1 for where it runs and how the service artifact is sourced/authenticated/version-pinned.

## Test strategy (tiered — finding #8)

1. **Unit (custom harness, runs here via `swift run …KitTests`):** Codable decoding vs real fixtures, APIError mapping, semver, preferences, speaker color, Overview aggregations vs captured JS-output fixtures, quit policy, ownership/nonce matching, discovery precedence.
2. **Integration (executable, runs here):** supervisor against a **real stub child process** (handshake parse, malformed/mixed stdout, stale `service.json` rejection, SIGTERM→SIGKILL escalation, port-file-vs-handshake race) and `APIClient` against a **local HTTP stub** (200/404/409/503, multipart).
3. **Packaged-app smoke (macOS CI):** tracer-bullet checklist on `macos-14`.
4. **Manual acceptance checklist** (docs): UI/audio/Gatekeeper/first-run.
5. **Update fault-injection:** Sparkle appcast with wrong version / unreachable feed / deferred-during-job (delegate unit tests), **plus a real v1→v2 packaged update test** — `build_app.sh` produces a v1 and a v2 `.app` + signed appcast and Sparkle installs v2 over v1 (run in the D0 milestone / release-smoke env, not the dev CLT env). This is what proves the ad-hoc-signed Sparkle install actually works end to end (finding #4).

**Coverage note:** the repo's 80% "Test & Coverage" gate is **Python-only** (existing CI). Swift Kit confidence is expressed via **suite completeness** (the enumerated unit + integration cases above), not a numeric gate; no Swift job is added to the Python coverage gate. (llvm-cov can be wired later if a number is wanted.)

## Files

Backend (small, justified): `service_main.py` (+nonce echo). Frontend/web client untouched.
`macos/`: `Package.swift` (Kit lib, App exe, KitTests exe, IntegrationTests exe; Sparkle dep) · `Sources/MeetingTranscriberKit/{Models,API,Service,Update(Sparkle glue),Preferences,Presentation,Overview}` · `Sources/MeetingTranscriberApp/{App,AppDelegate,AppState,Audio,Views/*}` · test sources · `Resources/{Info.plist,entitlements}` · `scripts/{build_app.sh,make_appcast.sh,stub_service.py}`.
CI: edit `.github/workflows/release.yml` (add `build-macos` job, `needs: release`, gated on `new_release_published`) + new reusable `.github/workflows/release-macos.yml` (`on: workflow_call`). Docs: `docs/macos-app.md` (parity matrix, ownership/shutdown, Gatekeeper, update, distribution, manual checklist).

## Sequence (finding #11 — two milestones, then numbered vertical slices)

**Milestone D1** — plumbing tracer bullet + stub service + build script (ad-hoc sign, Frameworks/ layout) — runnable `.app`; claims scoped to plumbing/lifecycle only.
**Milestone D0** — real-service packaging (Decision B): embed the actual PyInstaller service, launch, health + provisioning + one clean-machine transcription smoke; **emit + record `service-manifest.json`; measure the archive and settle the >2 GB path**; documented checklist. Gate before product-UI slices. (Maintainer arm64; not in the dev CLT env.)

1. Codable models + decoding fixtures/tests.
2. APIClient + error mapping + HTTP-stub integration tests.
3. Service supervisor/discovery/ownership/guarded-cleanup/shutdown + nonce backend edit + stub-child integration tests (Artifact A).
4. Bootstrap/error/setup-wizard views. **Invalid-token contract (finding #2):** UI never claims "token rejected" (backend doesn't validate). On `download_state == failed` show the generic `download_error` + two actions: **Retry** (re-POST `/provisioning/models`) and **Continue without diarization** (POST **empty** token → `required_repos()` drops pyannote → re-POST for a Whisper-only provision). Empty token at first entry → same path. Contract tests cover empty · non-empty-but-bad · offline · retry · continue-disabled against captured backend fixtures. EC-4/6/7, BR-13. No backend change.
5. Meeting browser + delete/confirm.
6. Upload + job lifecycle (poll, auto-navigate, cancel, meeting-retry).
7. Transcript + AVFoundation audio (speed/seek) + speaker editing (both scopes) + title/type/context edit.
8. Overview + per-segment insights + mismatch badge (Artifact B `port` aggregations + fixtures) + degraded states + language badge.
9. Analysis tab (assembled prompt).
10. Sparkle 2.9.6 update integration + defer-during-job + `release.yml` `build-macos` job + reusable `release-macos.yml` + `draftRelease` flow + appcast preservation + EdDSA enclosure signing + manifest verification (Artifact C).
11. `docs/macos-app.md` (all artifacts) + manual checklist. Update plan deviations. Open PR.

## Decisions (approved by user)

1. **Updater = Sparkle.** ✅
2. **Add `release-macos.yml`** (macOS CI builds/signs/attaches `.app` + appcast). ✅
3. **Overview parity = port the 3 aggregations into Swift** with fixture contract tests. ✅
4. **Ownership nonce** via ~5-line `service_main.py` edit. ✅

Resolved this round (in-plan, no user fork needed): #2 invalid-token UI contract, #3/OQ-5 mismatch badge → port, #4 Sparkle specifics (pin/embed/keys/library-validation), #5 guarded `service.json` cleanup, #6 renamed D1 tracer-bullet claims, #7 draft-until-complete release.

## Decision 1 (DECIDED = B) — service artifact origin & CI-build strategy (finding #1)

The `.app` must embed the **real** self-contained service (Torch/WhisperX/PyAnnote/ffmpeg, arm64, multi-GB). `MeetingTranscriber.spec` + `docs/packaging.md` exist, but the packaging **toolchain viability at CI scale is untested** and GitHub release assets cap at **2 GB/file**.

- **Option A — CI builds the service every app release** on `macos-14`: retires the risk most directly (reviewer's ask), but each release runs a multi-GB PyInstaller build (long, cache-heavy) and the frozen artifact may approach/exceed the 2 GB asset limit → may need splitting or external hosting. Built from the exact app commit "for free."
- **Option B (recommended) — maintainer builds the service bundle once per service version** on an arm64 machine (as docs/packaging.md already prescribes), uploads it as a **pinned, sha256-checksummed** release asset; the app-release CI **downloads that pinned artifact**, verifies the checksum, and embeds it. Fast app releases, no torch-in-CI, respects the existing packaging story. Compatibility is **not** asserted by version+checksum alone (that proves only artifact identity) — it is enforced by the manifest checks below (source commit SHA, `service_version` tag, embedded-manifest match, post-extraction content hash, arch). The D0 milestone does the clean-machine smoke.

Either way, **D0 (real-service embed + health/provisioning/transcription smoke) happens before product-UI slices.** Pick A or B (I recommend **B**).

**DECIDED: Option B.** ✅ Maintainer builds the service bundle once per service version on arm64; uploads it as a pinned, checksummed release asset; app-release CI downloads + verifies + embeds it.

**D0 must produce and record a service manifest** (`service-manifest.json`, checked into the app repo as the pinned expectation, and emitted alongside the service artifact — finding #3): compressed size, installed size, **exact service source commit SHA**, architecture (`arm64`), **`service_version`** (= the exact service release tag this app build pins — a build identifier, not an independent API-compatibility version; renamed from "API contract version" since the app pins one exact service build and there is no separate API-versioning policy yet), SHA-256 of the outer archive, PyInstaller + Python toolchain versions, and the smoke-test result. "Pin + checksum" only proves artifact identity, **not** same-commit or schema compatibility — the recorded **source commit SHA + `service_version`** are what tie the embedded service to a known-compatible build.

**`release-macos.yml` manifest verification (finding #4 this round) — CI rejects the artifact unless ALL hold:**
- outer downloaded-archive **SHA-256** matches the pinned manifest;
- **service source commit SHA** equals the app's pinned expected commit (or an explicitly listed approved-compatible commit);
- **service release tag** matches the pinned expected tag;
- the **manifest embedded inside the service artifact** matches the app-repo copy (no drift between what shipped and what's pinned);
- **hash of the embedded service after extraction** matches the manifest's installed-content hash (a valid outer archive must not contain an unexpected payload);
- the service **executable and the vendored ffmpeg/ffprobe are `arm64`** (`file` / `lipo -info`).

**>2 GB fallback — measured on the FINAL app enclosure, not just the service (finding #1 this round):** Sparkle needs a **single downloadable enclosure**; splitting the intermediate service archive does nothing for the final app update ZIP. So D0 measures the **final packaged application update ZIP** (app + embedded service, compressed). Decision tree:
- **Final ZIP ≤ 2 GB** → single GitHub Release enclosure (primary path). Split-archive may still be used only for the *intermediate* service artifact transfer, reassembled before embedding.
- **Final ZIP > 2 GB** → external enclosure hosting, but **gated (finding #1 this round — BR-21 conflict).** The native spec's BR-21 allowlists **only the public GitHub Releases API and asset URLs**; moving the enclosure to R2/S3 keeps the privacy objective (the host sees only an artifact request, no user data) but breaks that explicit allowlist. So: **the GitHub-only (≤2 GB) path is the approved implementation.** External hosting proceeds **only** if D0 finds the final ZIP exceeds 2 GB **and** shrinking cannot fix it, **and** requires **product approval + a documented BR-21 deviation** first. The chosen host must provide HTTPS, **stable immutable per-version URLs**, reasonable availability, and no unnecessary client-identifying telemetry. The appcast still lives on GitHub Releases; Sparkle's EdDSA verification is unchanged by transport. **Preferred first lever: shrink the bundle** (strip unused torch/CUDA components, test-only deps) to stay on the GitHub-only path.
- Multipart Sparkle enclosures are **not** part of the design (Sparkle doesn't support them).

**Compatibility anchor:** the pinned `service_version` (release tag) + **source commit SHA** recorded in `service-manifest.json`; CI asserts the embedded bundle's recorded version/commit equals the app's expected values (no backend change — uses the manifest, not a live endpoint).

## Deviations from spec

Resolves OQ-1 (macOS 13), OQ-2 (handshake+nonce, `service.json` fallback), OQ-3 (Sparkle + GitHub appcast), OQ-4 (dock), OQ-7 (spawn-own-child supervision + SIGTERM/KILL). **OQ-5 resolved for the per-segment mismatch badge** (ported as a display predicate; business content stays server-side). OQ-8 (parallel web + native) assumed → parity matrix guards divergence.

## Deviations from plan

_Populated after implementation._
