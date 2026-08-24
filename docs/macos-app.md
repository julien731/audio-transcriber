# Native macOS App

A thin native SwiftUI front end over the bundled local transcription service
(`docs/specs/local-transcription-service.md`). It embeds and supervises that
service and drives the whole workflow through the service HTTP API — no
transcription or workflow logic lives in the app (BR-6).

Spec: `docs/specs/native-macos-app.md` · Plan: `docs/plans/94-native-macos-app.md`.

**Name:** the product is **Blah** (`Blah.app`, bundle id `co.nimblehq.Blah`).
Internal SwiftPM targets (`MeetingTranscriberKit`/`MeetingTranscriberApp`) and the
embedded, separately-versioned **service** component keep the `MeetingTranscriber`
name — they're implementation details users never see, and the service owns its
own Application Support data dir.

## Layout

```
macos/
  Package.swift                     # Kit lib + App exe + two executable test targets; Sparkle 2.9.6 (pinned)
  Sources/MeetingTranscriberKit/    # testable, UI-independent logic
  Sources/MeetingTranscriberApp/    # SwiftUI shell (assembled into a .app by scripts/build_app.sh)
  Sources/MeetingTranscriberKitTests/          # unit suite  (swift run MeetingTranscriberKitTests)
  Sources/MeetingTranscriberIntegrationTests/  # integration (swift run MeetingTranscriberIntegrationTests)
  scripts/                          # build_app.sh, make_appcast.sh, fetch_service.sh, verify_update.sh, stub_service.py
  Resources/                        # Info.plist, entitlements
```

## Build & test (developer)

The dev toolchain here is Command Line Tools only (no XCTest/Testing module), so
tests are executables:

```bash
cd macos
swift run MeetingTranscriberKitTests            # unit
swift run MeetingTranscriberIntegrationTests    # integration (spawns real child processes)
bash scripts/build_app.sh debug                 # assemble build/Blah.app (ad-hoc signed)
open build/Blah.app
```

The debug build embeds a **stub** service (`scripts/stub_service.py`) that
simulates health + provisioning so the launch and setup flows are walkable
without the multi-GB ML stack. The real self-contained service is embedded at
Milestone D0 (below).

## App icon

Provide a 1024×1024 PNG, then:
```bash
bash macos/scripts/make_icon.sh path/to/icon-1024.png   # -> macos/Resources/AppIcon.icns
```
`build_app.sh` embeds `AppIcon.icns` automatically (absent → macOS default icon).
`Info.plist` already declares `CFBundleIconFile = AppIcon`.

## Feature-parity matrix (vs the web UI)

Each interaction is classified `API-data` (render as-is), `present` (format in
Swift, no logic), `port` (deterministic client aggregation reproduced in Swift
with fixture contract tests), or `excluded` (product sign-off).

| Web capability | API | Classification |
|---|---|---|
| Meeting list + status badges | `GET /meetings` | API-data |
| Upload (file/type/langs/speakers/preprocess/audio-analysis/context) + client validation | `POST /meetings` | API-data + `present` |
| Transcript tab, synced audio | `GET /meetings/{id}`, `/audio` | API-data + AVFoundation |
| Playback speed / seek / segment-click seek | client | `present` |
| Per-segment emotion label, prosody bars, interaction marker | `AudioAnalysis` | `present` |
| Word/tone-mismatch badge | `AudioAnalysis` + `segment.text` | `port` (4-line predicate; business content stays server-side) |
| Per-segment detected-language badge | `TranscriptSegment.language` | API-data |
| Degraded analysis states (failed/unavailable, dominant-speaker banner) | `AudioAnalysis` | `present` |
| Overview: energy trajectory, interruptions, response latency | raw emotion/interaction fields | `port` (fixture-tested) |
| Plain Text | transcript | `present` (`PlainTextRenderer`) |
| Analysis prompt | `GET /meetings/{id}/analysis-prompt` | API-data (assembled server-side) |
| Speaker rename — meeting-wide / single-segment | `PATCH /meetings/{id}`, `.../segments/speaker` | API-data |
| Title/type/context edit | `PATCH /meetings/{id}` | API-data |
| Delete meeting | `DELETE /meetings/{id}` | API-data (destructive confirm) |
| Cancel while processing / retry on error | `POST .../cancel`, `.../retry` | API-data |
| Recent speaker names, theme | UserDefaults | `present` (BR-11) |

## Service ownership & lifecycle

- The app **always spawns its own** service child and holds the `Process` handle;
  it never attaches to a pre-existing service. It terminates only that child.
- Discovery: the child's stdout `{"event":"ready","port",nonce}` handshake is
  authoritative; `service.json` is a fallback accepted only if its `pid` **and**
  the per-launch `nonce` (env `MT_SERVICE_NONCE`) match — rejecting stale files
  and PID reuse.
- Readiness = `GET /api/health` becomes reachable.
- Shutdown = `SIGTERM` → 10 s → `SIGKILL`; `service.json` is deleted only if it
  still identifies our child. Interrupted jobs are reconciled server-side by
  `recover_stuck_meetings()`.
- Closing the window keeps the app + service running (background transcription,
  BR-15); reopening restores state (BR-17). Quitting while a transcription is
  active prompts for confirmation (BR-16, EC-5).

## First-run setup

The setup wizard collects an optional HuggingFace token and shows model-download
progress. The backend does **not** validate the token, so the UI never claims
"token rejected": a failed download shows the service's own message plus **Retry**
and **Continue without diarization** (which clears the token and re-provisions
Whisper-only). Setup shows only until provisioning completes (BR-14).

## Gatekeeper: one-time bypass (self-signed build, BR-23, EC-1)

The app is self-signed (not notarized), so a downloaded copy is blocked on first
launch: *"Blah" Not Opened — Apple could not verify … is free of malware* with
only **Move to Trash** / **Done**. It is **not** permanently blocked.

On **macOS 15 (Sequoia) and later** (incl. macOS 26), Apple removed the old
Control-click → Open bypass; approve it in Settings instead:

1. Click **Done** (not "Move to Trash").
2. Open **System Settings → Privacy & Security**.
3. Under **Security**, find *""Blah" was blocked to protect your Mac"* and click
   **Open Anyway**; authenticate (Touch ID / password).
4. Confirm **Open** on the final prompt.

macOS remembers the choice; subsequent launches are a normal double-click. (On
macOS 13–14 the older **right-click → Open → Open** flow still works.)

> The real fix for this friction is **Apple notarization** (a paid Developer
> account), deferred by BR-23 in favor of self-signing. Until then, each user does
> the one-time Open-Anyway step above. `xattr -dr com.apple.quarantine Blah.app`
> also works but is a blunter, less transparent bypass.

## Auto-update (Sparkle) & release publishing

- The app uses **Sparkle 2.9.6** (pinned). The appcast lives at the stable
  `releases/latest/download/appcast.xml`; each enclosure is **EdDSA-signed** and
  verified via `SUPublicEDKey` before install. Updates are deferred while a
  transcription is active. An update failure leaves the current version working
  (BR-22). Only the public GitHub Releases API + asset URLs are contacted (BR-21).
- Publishing runs in the **same** `release.yml` run (a tag-triggered workflow
  would not fire under the `GITHUB_TOKEN` recursion guard): the `build-macos` job
  (macos-15, gated on a release having published) fetches the pinned service
  artifact, assembles + signs the `.app`, produces the signed enclosure, updates
  the appcast, uploads to the release, then flips the **draft** to published.
- User data lives in Application Support, outside the bundle, so updates never
  touch it (BR-24, EC-10).

### Required CI secrets

- `SPARKLE_ED_PRIVATE_KEY` / `SPARKLE_ED_PUBLIC_KEY` — the EdDSA key pair
  (`generate_keys` from Sparkle). Public half is committed as `SUPublicEDKey` in
  `Info.plist` and set as the public secret; CI also injects it at build time.

### Generating the Sparkle signing key (one-time)

The private key signs every release's appcast and cannot be rotated without
breaking auto-update for already-installed apps (the public key is baked into
each shipped `Info.plist`). Generate once, back it up durably, and guard it.

```bash
cd macos
BIN=.build/artifacts/sparkle/Sparkle/bin      # populated by `swift build`

# 1. Generate (private key → login Keychain; prints the public key).
"$BIN/generate_keys"                          # re-print later with: "$BIN/generate_keys" -p

# 2. Export the private key for the CI secret + a durable backup.
"$BIN/generate_keys" -x /tmp/sparkle_private_key.txt

# 3. Set the two GitHub Actions secrets.
gh secret set SPARKLE_ED_PRIVATE_KEY --repo <owner>/<repo> < /tmp/sparkle_private_key.txt
gh secret set SPARKLE_ED_PUBLIC_KEY  --repo <owner>/<repo> --body '<public-key>'

# 4. Back up the private key to a secrets manager, then securely delete the file.
rm -P /tmp/sparkle_private_key.txt

# 5. Commit the public key into Info.plist (safe — it is public):
/usr/libexec/PlistBuddy -c "Set :SUPublicEDKey <public-key>" Resources/Info.plist
```

**Losing the private key = you can no longer ship auto-updates to existing
installs.** Keep it in a secrets manager, not only the Keychain.

## Milestone D0 — real-service packaging (maintainer, arm64 + full Xcode)

Runs before shipping; not reproducible in the CLT-only dev env or cheaply in CI.

1. Build the real self-contained service: `./scripts/vendor_ffmpeg.sh && pyinstaller MeetingTranscriber.spec` (see `docs/packaging.md`).
2. Package + pin it with **`macos/scripts/package_service.sh`** (see "Cutting a
   release" below) — produces the release zip (with an embedded manifest) and the
   committed `macos/service-manifest.json` (sha256, `service_version`, source
   commit, arch, sizes).
3. Embed the real service under `Contents/Resources/service/`, assemble the
   `.app`, and smoke-test: `/api/health`, `POST /provisioning/token` (empty),
   `POST /provisioning/models` (Whisper-only) to completion, and one clean-machine
   transcription. (Validated in this session on a short speech clip → `ready`.)
4. **Measure the final app zip** (`release-macos.yml` does this automatically and
   fails > 2 GB). ≤ 2 GB → single GitHub enclosure. > 2 GB → shrink the bundle
   first; only if that fails, external enclosure hosting with product approval +
   a documented BR-21 deviation.
5. Generate the Sparkle EdDSA keys and set the secrets (see "Generating the
   Sparkle signing key" above).
6. **Pre-publish gate** (`scripts/verify_update.sh`, run by `release-macos.yml`):
   headless checks that the appcast parses with a signed enclosure, the enclosure
   length matches the zip, the `.app` passes `codesign --verify --deep --strict`,
   and `SUPublicEDKey` is a real key — fails the release before publishing on any
   miss. The full GUI v1→v2 Sparkle install remains a **manual** check on the
   published release (download v(N-1), let it update to v(N)).

## Cutting a release (runbook)

The app version is computed by `semantic-release` from merged-PR labels (see
`CLAUDE.md`); the service is pinned independently via `service-manifest.json`.

### One-time setup
1. Generate the Sparkle keys + set `SPARKLE_ED_PRIVATE_KEY` / `SPARKLE_ED_PUBLIC_KEY`
   secrets, and commit `SUPublicEDKey` in `Info.plist` (see above).

### Publish/pin the service artifact (Decision B) — whenever the service changes
On an arm64 machine with the venv + `pyinstaller` (see `docs/packaging.md`):
```bash
./scripts/vendor_ffmpeg.sh && pyinstaller MeetingTranscriber.spec   # build dist/MeetingTranscriber
VERSION=service-vX.Y.Z bash macos/scripts/package_service.sh        # -> zip + manifests
gh release create service-vX.Y.Z macos/build/MeetingTranscriberService-service-vX.Y.Z.zip \
  --repo <owner>/<repo> --title "Bundled service service-vX.Y.Z" \
  --notes "Self-contained arm64 service."
git add macos/service-manifest.json && git commit -m "chore: pin service service-vX.Y.Z"
```
`macos/service-manifest.json` is the CI's pinned expectation. If it is absent or
does not match the published asset, `release-macos.yml` fails (by design).

### Ship an app release
2. Open a PR to `main` with a release-triggering label (`feature`/`bug`/`breaking`;
   see `CLAUDE.md`). Merge (squash).
3. `release.yml` runs `semantic-release` → creates a **draft** release, then the
   `build-macos` job (same run) fetches+verifies the pinned service, assembles +
   signs the `.app`, signs the enclosure, updates `appcast.xml`, runs the
   pre-publish gate, uploads the zip + appcast, and flips the draft to published.
4. **Post-publish (manual):** on a clean Mac, download the `.app`, confirm the
   one-time Settings → Open Anyway (Gatekeeper, macOS 15+), and — from the prior release —
   confirm Sparkle auto-updates to the new version.

### Recovery
A failed `build-macos` job leaves the release a **draft** (never a broken
published version). Re-run the failed job (preserves the version), or re-invoke
`release-macos.yml` via `workflow_dispatch` with the tag.

## Manual acceptance checklist

- [ ] First launch: Settings → Privacy & Security → Open Anyway past Gatekeeper (EC-1).
- [ ] Setup wizard: enter a token → models download → main UI; and: skip / bad
      token → generic failure → Continue without diarization completes (BR-13).
- [ ] Upload a recording → progress → transcript appears automatically (BR-9).
- [ ] Transcript: audio plays, active segment highlights, clicking anywhere on a
      segment (or its timecode) seeks to that segment's exact start, change
      playback speed.
- [ ] Rename a speaker (single segment and meeting-wide); recent names appear.
- [ ] Overview: trajectory + interruptions + latency render; opted-out and
      unavailable states render.
- [ ] Analysis: generate a prompt, copy it; unnamed-speaker warning shows.
- [ ] Delete a meeting (confirm dialog).
- [ ] Close the window during a transcription → it keeps running; reopen restores
      the in-progress job (BR-15/BR-17).
- [ ] Quit during a transcription → confirmation dialog (BR-16).
- [ ] Offline: existing meetings browse/read; new transcription blocked with the
      service's message (EC-7).
- [ ] Check for Updates… (against a test appcast).
