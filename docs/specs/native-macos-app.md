# Spec: Native macOS App

## Overview

### Problem Statement

Meeting Transcriber is used day-to-day by non-technical members of the Nimble sales team, but running it means opening Terminal and running a server, then using the tool through a web browser. The team wants a real Mac app: something they install by dragging one file to Applications, launch by double-clicking, and use in a native window that feels like macOS — not a browser tab. This spec covers building that native SwiftUI application. The app is a thin native front end over the bundled local service specified in `local-transcription-service.md`: it embeds and supervises that service, and drives the entire workflow through the service's HTTP API. No transcription logic lives in the app itself.

### Goals

- Give the team a native macOS app installed by dragging one file to Applications and launched by double-clicking — no Terminal.
- Deliver a native SwiftUI interface with feature parity to the current web UI (meeting list, upload, transcript viewer, overview, audio insights, speaker editing, analysis prompts).
- Embed and supervise the bundled local service so the user never manages a server.
- Keep the app thin: all workflow logic comes from the service's HTTP API (`local-transcription-service.md`).
- Keep the app current with a self-update mechanism, and make first-launch security friction a documented one-time step.
- Preserve the fully-local, privacy-first behavior end to end.

### Scope

This spec covers:

- A native SwiftUI application for Apple Silicon, distributed as a single `.app` bundle.
- Embedding the bundled local service inside the app and supervising its lifecycle (start, readiness, shutdown).
- Native views with feature parity to the current web UI.
- A native first-run setup experience that drives the service's provisioning (HuggingFace token entry, model-download progress).
- Native audio playback, job-progress polling, and local UI preferences.
- Window, dock/menu-bar presence, and quit behavior (including background transcription).
- Auto-update, self-signing, and documented Gatekeeper bypass.
- Distribution to the team.

Out of scope:

- The transcription pipeline, bundling of Python/native binaries, model/token provisioning, data storage, isolation, and API completeness — all owned by `local-transcription-service.md`.
- Running transcription against a remote server — a future spec. This app talks only to its embedded local service.
- Apple notarization and Mac App Store distribution (self-signing is used instead).
- Windows or Linux apps.
- Authentication (local-only, unauthenticated, consistent with current behavior).

## User Stories

### End User (non-technical teammate)

- As an **End User**, I can install the app by dragging one file to Applications, so that I never touch Terminal.
- As an **End User**, I can double-click the app and get a native window, so that it feels like a normal Mac app.
- As an **End User**, I can complete a short native setup on first launch (enter my HuggingFace token, watch models download), so that I can start transcribing without technical configuration.
- As an **End User**, I can upload a recording, watch transcription progress, and read the transcript with synced audio playback, so that I can review meetings.
- As an **End User**, I can view the overview, audio insights, edit speaker names, and generate an analysis prompt to paste into an LLM, so that I keep the full current workflow.
- As an **End User**, I can keep my previously-transcribed meetings after switching from the old Terminal version, so that I lose no history.
- As an **End User**, I can close the window while a transcription is running and have it finish, so that I do not babysit long jobs.
- As an **End User**, I can use the app offline to browse and read existing meetings, so that connectivity does not block my own data.
- As an **End User**, I can have the app update itself, so that I get improvements without reinstalling.
- As an **End User**, I can follow clear one-time instructions to open the app past macOS's security warning, so that self-signing does not leave me stuck.

### Maintainer (app publisher)

- As a **Maintainer**, I can build a signed `.app` bundle that embeds the local service and publish it to a shared location, so that the team can download it.
- As a **Maintainer**, I can publish a new version that installed apps discover and install automatically, so that I do not walk each user through updating.

## Business Rules

### App Packaging & Service Supervision

| # | Rule | Rationale |
|---|------|-----------|
| BR-1 | The app is distributed as a single native macOS `.app` bundle targeting Apple Silicon, embedding the bundled local service (`local-transcription-service.md`). | One double-clickable artifact keeps install trivial; the service travels inside it. |
| BR-2 | On launch, the app starts the embedded service and waits for its readiness signal before presenting the main UI. | The UI must not issue requests before the service is up (service BR-5). |
| BR-3 | The app discovers the service's ephemeral local port at runtime rather than assuming a fixed port. | The service binds an ephemeral port to avoid collisions; the app must learn it (service BR-4). |
| BR-4 | The app supervises the service process: if the service exits unexpectedly, the app surfaces a clear error and can restart it. | A crashed backend must not leave a silently broken window. |
| BR-5 | When the app quits, it shuts the embedded service down cleanly. | No orphaned background server processes. |
| BR-6 | The app contains no transcription, diarization, or workflow business logic of its own; it drives everything through the service's HTTP API. | Keeps the app thin and the API the single source of truth (service BR-16). |

### User Interface

| # | Rule | Rationale |
|---|------|-----------|
| BR-7 | The app provides native SwiftUI views with feature parity to the current web UI: meeting list, upload, transcript viewer, overview, audio insights, speaker editor, and analysis-prompt generation. | Users must not lose any current capability in the move to native. |
| BR-8 | The app plays meeting audio natively (AVFoundation), streaming from the service's audio endpoint, with playback synced to transcript segments as today. | Native playback replaces the HTML `<audio>` element while preserving synced review. |
| BR-9 | The app polls the service's job endpoint to show transcription progress and transitions to the transcript when complete. | Reproduces the current progress-and-auto-navigate behavior natively. |
| BR-10 | The app requests the fully-assembled analysis prompt from the service rather than assembling it locally. | Prompt assembly is server-side (service BR-17); the app must not reimplement it. |
| BR-11 | Local UI preferences (theme, recently-used speaker names) are stored in native app storage (e.g., `UserDefaults`). | Replaces the web client's `localStorage` for presentation-only state. |

### First-Run Setup (native)

| # | Rule | Rationale |
|---|------|-----------|
| BR-12 | On first launch, the app presents a native guided setup that collects the HuggingFace token and shows model-download progress, driving the service's provisioning. | Non-technical users need setup handled in-app; the service performs provisioning (service BR-8, BR-9). |
| BR-13 | If the token is empty or rejected, the app explains diarization requires a valid token and lets the user proceed with diarization disabled or re-enter it. | Presents the service's graceful-degradation behavior (service BR-10). |
| BR-14 | The app shows setup only until provisioning is complete, using the service's provisioning-complete state. | Avoids re-prompting on later launches (service BR-11). |

### Window, Background & Quit Behavior

| # | Rule | Rationale |
|---|------|-----------|
| BR-15 | Closing the window does not stop an in-progress transcription; the app keeps running (dock/menu presence) until the job completes or the user quits. | Transcriptions run 1–2 hours; users should not keep a window open. |
| BR-16 | Explicitly quitting while a transcription is active prompts the user to confirm, warning the running job will be lost. | Job state is in-memory and lost on service shutdown; the user should choose knowingly. |
| BR-17 | Reopening the window while the app is still running returns the user to current state, including any in-progress job. | Continuity for the background-run model. |
| BR-18 | The app functions offline for browsing and reading existing meetings, reflecting the service's offline behavior. | Users' data is local; connectivity should not block access (service BR-23, BR-24). |

### Distribution, Update & Signing

| # | Rule | Rationale |
|---|------|-----------|
| BR-19 | The app is distributed as a shared file (shared drive or download link) and checks a published source for newer versions. | Chosen distribution model: shared file plus auto-update; minimal hosting. |
| BR-20 | When a newer version is available, the app notifies the user and can download and install it, applying the update on next relaunch. | Keeps the team current without manual reinstalls. |
| BR-21 | Update checks and downloads transmit only version metadata and the app binary — never user meeting data. | Preserves the privacy guarantee even while networked for updates. |
| BR-22 | An update failure (offline, download error) leaves the current version fully functional. | Users must never be left with a broken install. |
| BR-23 | The app is code-signed (self-signed for now) and ships with documented one-time Gatekeeper bypass instructions. | Self-signing chosen over paid notarization; users need a clear path past the warning. |
| BR-24 | An app update never disturbs user data, which lives outside the app bundle in the service's Application Support location. | Data survives updates because it is not inside the bundle (service BR-12). |

## Data Requirements

### App-Local Preferences

Stored in native app storage; presentation-only, not authoritative workflow state.

| Field | Required | Notes |
|-------|----------|-------|
| `theme` | No | Light/dark preference. Replaces the web client's localStorage theme. |
| `recent_speaker_names` | No | Recently-used speaker names for quick reassignment. Replaces the web client's localStorage list. |
| `service_port` | No | Last-known service port for the current run; transient, re-discovered each launch (BR-3). |

### Authoritative Data (owned by the service)

All meetings, transcripts, configuration, and models are owned and stored by the local service (see `local-transcription-service.md`). The app holds no authoritative copy; it reads and writes them exclusively through the service API.

### Entity Relationships

| Entity | Relationship | Entity |
|--------|-------------|--------|
| Native App | embeds and supervises | Local Service |
| Native App | reads/writes via API | Meetings (owned by the service) |
| App-Local Preferences | belongs to | Native App |

## Edge Cases

| # | Scenario | Expected Behavior |
|---|----------|--------------------|
| EC-1 | First launch shows macOS's "unidentified developer" warning (self-signed build). | The app is not permanently blocked; documented one-time right-click → Open lets the user proceed (BR-23). |
| EC-2 | The embedded service fails to start or never becomes ready. | The app shows a clear error state (not a blank window) and offers to retry, rather than hanging (BR-2, BR-4). |
| EC-3 | The service process crashes while the app is open (e.g., mid-transcription). | The app surfaces the failure, can restart the service, and reflects that the in-progress job was lost (BR-4, BR-16). |
| EC-4 | User closes the window during a 90-minute transcription. | Transcription continues in the background; reopening shows the in-progress job (BR-15, BR-17). |
| EC-5 | User selects Quit while a transcription is active. | The app warns the job will be lost and asks for confirmation before shutting the service down (BR-16, BR-5). |
| EC-6 | User provides an invalid/empty HuggingFace token during setup. | The app explains diarization needs a valid token and offers to continue disabled or re-enter it (BR-13). |
| EC-7 | Machine is offline on first launch before models are downloaded. | The app lets the user browse existing meetings but blocks new transcription with a clear message (BR-18). |
| EC-8 | Auto-update check fails (offline or source unreachable). | The current version keeps working; the failure is a non-blocking notice (BR-22). |
| EC-9 | User launches a second copy of the app while one is running. | The app focuses the existing instance rather than starting a second embedded service against the same data. |
| EC-10 | The app is updated while a prior version's data lives in Application Support. | Data is untouched because it lives outside the bundle (BR-24). |

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| OQ-1 | Minimum supported macOS version for the SwiftUI app? | Product / Engineering | Open |
| OQ-2 | How does the app discover the service's port and readiness — stdout handshake, a written port file, or a fixed-port fallback? | Engineering | Open — spans this spec and `local-transcription-service.md` (its OQ-2) |
| OQ-3 | Where is the auto-update feed/binary hosted, and what mechanism performs the update (e.g., Sparkle) given a self-signed build? | Engineering / Product | Open |
| OQ-4 | Should the app present a menu-bar presence, a dock icon, or both when running in the background after the window is closed? | Product / Design | Open |
| OQ-5 | Is a paid Apple Developer account acceptable later to remove Gatekeeper friction entirely (notarization)? | Product | Open — deferred; self-signing chosen for now |
| OQ-6 | What is the exact feature-parity checklist against the current web UI (e.g., every tab in the transcript viewer, keyboard behaviors, speaker-color scheme)? | Product / Design | Open |
| OQ-7 | How is the embedded service supervised on macOS (child process lifecycle, crash detection, clean shutdown on app termination)? | Engineering | Open |
| OQ-8 | Does the transition ship the web client and native app in parallel for a period, or does the native app replace the web UI outright once ready? | Product | Open |
