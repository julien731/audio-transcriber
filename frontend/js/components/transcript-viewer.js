let currentAudio = null;
let pollInterval = null;
let autoScroll = true;
let userScrolledAway = false;
let scrollTimeout = null;
let backToTopHandler = null;

const ICON_PLAY = '<svg viewBox="0 0 12 12" fill="currentColor" stroke="none"><path d="M3 2 L10 6 L3 10 Z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 12 12" fill="currentColor" stroke="none"><rect x="3" y="2" width="2.5" height="8" rx="0.5"/><rect x="6.5" y="2" width="2.5" height="8" rx="0.5"/></svg>';

function renderTranscriptView(container, meetingId) {
    container.innerHTML = '<div class="loading">Loading meeting...</div>';
    loadMeetingView(container, meetingId);
}

async function loadMeetingView(container, meetingId) {
    try {
        const meeting = await API.getMeeting(meetingId);
        const meta = meeting.metadata;
        const transcript = meeting.transcript;
        const audioAnalysis = meeting.audio_analysis || null;

        const isProcessing = meta.status === 'processing';
        const isError = meta.status === 'error';

        container.innerHTML = `
            <div class="meeting-view">
                <div class="page-header">
                    <button class="btn-back" onclick="App.navigate('/')" aria-label="Back to meetings"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2 L4 7 L9 12"/></svg></button>
                    <h1 id="meeting-title">${escapeHtml(meta.title)}</h1>
                    <div class="header-actions">
                        <span class="meta-pill meta-pill-${meta.type}">${escapeHtml(meta.type)}</span>
                        ${!isProcessing ? `<button class="btn btn-outline" onclick="handleRerunTranscription('${meetingId}')">Re-run</button>` : ''}
                        <button class="btn btn-outline btn-outline-danger" onclick="handleDeleteMeeting('${meetingId}')">Delete</button>
                    </div>
                </div>

                ${isProcessing ? `
                    <div id="progress-section" class="progress-section"></div>
                    <div class="cancel-section">
                        <button class="btn btn-text btn-cancel" onclick="handleCancelTranscription('${meetingId}')">Cancel transcription</button>
                    </div>
                ` : ''}
                ${isError ? `
                    <div class="error-state">
                        ${meta.error ? escapeHtml(meta.error) : 'Transcription failed.'}
                        <button class="btn btn-text" onclick="retryTranscription('${meetingId}')">Retry</button>
                    </div>
                ` : ''}

                ${!isProcessing && !isError ? `
                    <div class="form-group">
                        <label for="meeting-context">Context</label>
                        <textarea id="meeting-context" rows="3" placeholder="Add context about this meeting for better analysis...">${escapeHtml(meta.context || '')}</textarea>
                    </div>

                    <div class="audio-player" id="audio-player">
                        <audio id="audio-element" preload="metadata"></audio>
                        <div class="player-controls">
                            <button class="btn btn-icon" id="skip-back" title="Back 15s" aria-label="Back 15 seconds"><svg viewBox="0 0 12 12" fill="currentColor" stroke="none" aria-hidden="true"><path d="M10 2 L5 6 L10 10 Z M5 2 L0 6 L5 10 Z"/></svg></button>
                            <button class="btn btn-icon btn-play" id="play-pause" title="Play/Pause" aria-label="Play"><svg viewBox="0 0 12 12" fill="currentColor" stroke="none" aria-hidden="true"><path d="M3 2 L10 6 L3 10 Z"/></svg></button>
                            <button class="btn btn-icon" id="skip-forward" title="Forward 15s" aria-label="Forward 15 seconds"><svg viewBox="0 0 12 12" fill="currentColor" stroke="none" aria-hidden="true"><path d="M2 2 L7 6 L2 10 Z M7 2 L12 6 L7 10 Z"/></svg></button>
                        </div>
                        <div class="player-seek">
                            <span id="current-time">0:00</span>
                            <input type="range" id="seek-bar" min="0" max="100" value="0" step="0.1">
                            <span id="total-time">${formatDuration(meta.duration_seconds)}</span>
                        </div>
                        <div class="player-speed">
                            <select id="speed-select">
                                <option value="0.5">0.5x</option>
                                <option value="1" selected>1x</option>
                                <option value="1.25">1.25x</option>
                                <option value="1.5">1.5x</option>
                                <option value="2">2x</option>
                            </select>
                        </div>
                    </div>

                    <div class="tabs">
                        <button class="tab tab-active" data-tab="transcript" onclick="switchTab('transcript', '${meetingId}', '${meta.type}')">Transcript</button>
                        <button class="tab" data-tab="overview" onclick="switchTab('overview', '${meetingId}', '${meta.type}')">Overview</button>
                        <button class="tab" data-tab="plaintext" onclick="switchTab('plaintext', '${meetingId}', '${meta.type}')">Plain Text</button>
                        <button class="tab" data-tab="analysis" onclick="switchTab('analysis', '${meetingId}', '${meta.type}')">Analysis</button>
                        <label class="auto-scroll-toggle">
                            <input type="checkbox" id="auto-scroll-check" ${autoScroll ? 'checked' : ''} onchange="autoScroll = this.checked">
                            Auto-scroll
                        </label>
                    </div>

                    <div id="transcript-tab" class="tab-content tab-content-active"></div>
                    <div id="overview-tab" class="tab-content" hidden></div>
                    <div id="plaintext-tab" class="tab-content" hidden></div>
                    <div id="analysis-tab" class="tab-content" hidden></div>

                    <aside id="speakers-sidebar" class="speakers-sidebar" aria-label="Speakers"></aside>
                    <button class="speakers-fab" id="speakers-fab" title="Show speakers" aria-label="Show speakers" aria-expanded="false" aria-controls="speakers-sidebar">
                        <span class="speakers-fab-icon" aria-hidden="true">👤</span>
                        <span class="speakers-fab-badge" id="speakers-fab-badge" hidden></span>
                    </button>

                    <button class="back-to-top" id="back-to-top" title="Back to top" aria-label="Back to top">↑</button>
                ` : ''}
            </div>
        `;

        if (isProcessing && meta.job_id) {
            startPolling(meetingId, meta.job_id);
        }

        if (!isProcessing && !isError && transcript) {
            setupAudioPlayer(meetingId);
            setupContextEditor(meetingId);
            renderSegments(document.getElementById('transcript-tab'), transcript, meta, meetingId, audioAnalysis);
            setupScrollDetection();
            setupBackToTop();
            setupSpeakersSidebar();
            window._meetingAudioAnalysis = audioAnalysis;
            window._meetingMeta = meta;
            window._meetingTranscript = transcript;
        }
    } catch (err) {
        container.innerHTML = `<div class="error-state">Failed to load meeting: ${escapeHtml(err.message)}</div>`;
    }
}

function setupAudioPlayer(meetingId) {
    const audio = document.getElementById('audio-element');
    if (!audio) return;

    audio.src = API.audioUrl(meetingId);
    currentAudio = audio;

    const playPause = document.getElementById('play-pause');
    const seekBar = document.getElementById('seek-bar');
    const currentTimeEl = document.getElementById('current-time');
    const speedSelect = document.getElementById('speed-select');

    playPause.addEventListener('click', () => {
        if (audio.paused) {
            audio.play();
            playPause.innerHTML = ICON_PAUSE;
            playPause.setAttribute('aria-label', 'Pause');
        } else {
            audio.pause();
            playPause.innerHTML = ICON_PLAY;
            playPause.setAttribute('aria-label', 'Play');
        }
    });

    document.getElementById('skip-back').addEventListener('click', () => {
        audio.currentTime = Math.max(0, audio.currentTime - 15);
    });

    document.getElementById('skip-forward').addEventListener('click', () => {
        audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 15);
    });

    audio.addEventListener('timeupdate', () => {
        if (audio.duration) {
            seekBar.value = (audio.currentTime / audio.duration) * 100;
            currentTimeEl.textContent = formatDuration(audio.currentTime);
            highlightCurrentSegment(audio.currentTime);
        }
    });

    audio.addEventListener('ended', () => {
        playPause.innerHTML = ICON_PLAY;
        playPause.setAttribute('aria-label', 'Play');
    });

    seekBar.addEventListener('input', () => {
        if (audio.duration) {
            audio.currentTime = (seekBar.value / 100) * audio.duration;
        }
    });

    speedSelect.addEventListener('change', () => {
        audio.playbackRate = parseFloat(speedSelect.value);
    });
}

function setupContextEditor(meetingId) {
    const textarea = document.getElementById('meeting-context');
    if (!textarea) return;

    let savedValue = textarea.value.trim();

    textarea.addEventListener('blur', async () => {
        const newValue = textarea.value.trim();
        if (newValue === savedValue) return;
        try {
            await API.updateMeeting(meetingId, { context: newValue });
            savedValue = newValue;
            showToast('Context saved');
        } catch (err) {
            showToast('Failed to save context', 'error');
        }
    });
}

function renderSegments(container, transcript, meta, meetingId, audioAnalysis) {
    const speakers = { ...meta.speakers };
    const speakerIds = [];
    const firstSegmentIndex = {};
    transcript.segments.forEach((seg, i) => {
        if (!(seg.speaker in firstSegmentIndex)) {
            firstSegmentIndex[seg.speaker] = i;
            speakerIds.push(seg.speaker);
        }
    });
    const speakerColorMap = {};
    speakerIds.forEach((id, i) => { speakerColorMap[id] = getSpeakerColor(i); });

    // Store speakers data globally so onclick handlers can reference it without inline JSON
    window._speakerEditorState = { speakers, meetingId, speakerIds, speakerColorMap, firstSegmentIndex };

    const showInsights = AudioInsights.hasCompletedAnalysis(meta, audioAnalysis);
    const insightsBySegment = showInsights ? AudioInsights.indexBySegment(audioAnalysis) : {};

    // Meeting's primary detected language; legacy/single-language segments that
    // carry no per-segment language fall back to it for the badge (EC-8).
    const meetingLanguage = transcript.language;

    let prevSpeaker = null;
    container.innerHTML = `
        <div class="segments" id="segments-container">
            ${transcript.segments.map((seg, i) => {
                const speakerName = speakers[seg.speaker] || seg.speaker;
                const color = speakerColorMap[seg.speaker];
                const insights = showInsights ? renderSegmentInsights(seg, insightsBySegment[seg.id]) : '';
                const languageBadge = renderLanguageBadge(seg, meetingLanguage);
                const isContinuation = seg.speaker === prevSpeaker;
                prevSpeaker = seg.speaker;
                return `
                    <div class="segment${isContinuation ? ' segment-cont' : ''}" id="seg-${i}" data-start="${seg.start}" data-end="${seg.end}" data-segment-id="${escapeHtml(seg.id)}">
                        <div class="segment-header">
                            <span class="segment-time">${formatTimestamp(seg.start)}</span>
                            <button type="button" class="speaker-label" style="background-color: ${color}" data-speaker="${seg.speaker}"
                                onclick="handleSpeakerClick(this, '${seg.speaker}', '${seg.id}')">
                                ${escapeHtml(speakerName)}<svg class="speaker-label-chevron" width="8" height="6" viewBox="0 0 8 6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M1 1.5 L4 4.5 L7 1.5"/></svg>
                            </button>
                            ${languageBadge}
                            ${insights}
                            <button class="btn btn-icon btn-segment-play" onclick="playFromSegment(${seg.start})" title="Play from here">▶</button>
                        </div>
                        <div class="segment-text">${escapeHtml(seg.text)}</div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

// Renders a presentational badge naming the segment's detected language.
// Uses the segment's own stored language when present, otherwise falls back to
// the meeting's primary detected language (EC-8). The badge is UI-only and is
// never read by the copy/export path (renderPlainTextTab), so output is
// unchanged (OQ-5). Omitted entirely when no language can be resolved.
function renderLanguageBadge(seg, meetingLanguage) {
    const code = seg.language || meetingLanguage;
    const name = formatLanguageName(code);
    if (!name) return '';
    const title = seg.language
        ? `Detected language: ${name}`
        : `Meeting language: ${name}`;
    return `<span class="language-badge" title="${escapeHtml(title)}">${escapeHtml(name)}</span>`;
}

function renderSegmentInsights(seg, insights) {
    if (!insights) return '';
    const parts = [];
    const { emotion, prosody, interaction } = insights;

    if (emotion) {
        const label = AudioInsights.EMOTION_LABELS[emotion.primary_emotion] || emotion.primary_emotion;
        const muted = emotion.low_confidence ? ' emotion-badge-muted' : '';
        const tooltip = `Tone: ${label} (confidence ${(emotion.confidence * 100).toFixed(0)}%)`;
        const hedge = emotion.low_confidence ? '?' : '';
        parts.push(
            `<span class="emotion-badge emotion-${emotion.primary_emotion}${muted}" title="${escapeHtml(tooltip)}">${escapeHtml(label)}${hedge}</span>`
        );
    }

    if (prosody) {
        parts.push(
            `<span class="prosody-indicator" title="${escapeHtml(AudioInsights.formatProsodyTooltip(prosody))}" aria-label="Prosody summary"><i></i><i></i><i></i></span>`
        );
    }

    if (emotion && AudioInsights.isWordToneMismatch(emotion, seg.text)) {
        const tooltip = `Words signal agreement but tone reads ${emotion.primary_emotion} (${(emotion.confidence * 100).toFixed(0)}% confidence). Click to seek.`;
        parts.push(
            `<button type="button" class="mismatch-marker" title="${escapeHtml(tooltip)}" onclick="playFromSegment(${seg.start})" aria-label="Word-tone mismatch — seek to ${formatTimestamp(seg.start)}">⚠ tone</button>`
        );
    }

    if (interaction && AudioInsights.hasInteractionMarker(interaction)) {
        parts.push(
            `<span class="interaction-marker" title="${escapeHtml(AudioInsights.interactionTooltip(interaction))}" aria-label="Interaction event">↺</span>`
        );
    }

    return parts.length ? `<span class="segment-insights">${parts.join('')}</span>` : '';
}

function handleSpeakerClick(element, speakerId, segmentId) {
    const state = window._speakerEditorState;
    const speakerName = state.speakers[speakerId] || speakerId;
    openSpeakerEditor(
        element.closest('.segment'),
        speakerId,
        speakerName,
        state.meetingId,
        { ...state.speakers },
        () => App.navigate('/meetings/' + state.meetingId),
        segmentId
    );
}

function renderSpeakersSidebar() {
    const sidebar = document.getElementById('speakers-sidebar');
    const fabBadge = document.getElementById('speakers-fab-badge');
    const state = window._speakerEditorState;
    if (!sidebar || !state || !state.speakerIds) return;

    const { speakers, speakerIds, speakerColorMap } = state;
    const total = speakerIds.length;
    const unnamedCount = speakerIds.filter(id => isUnidentifiedSpeaker(speakers[id] || id)).length;

    const headerText = unnamedCount === 0
        ? `${total} speaker${total === 1 ? '' : 's'}, all named`
        : `${unnamedCount} of ${total} unnamed`;

    sidebar.innerHTML = `
        <div class="speakers-sidebar-header">
            <span class="speakers-sidebar-title">Speakers</span>
            <button class="speakers-sidebar-close" id="speakers-sidebar-close" title="Close" aria-label="Close speakers panel">×</button>
        </div>
        <div class="speakers-sidebar-summary">${escapeHtml(headerText)}</div>
        <ul class="speakers-list">
            ${speakerIds.map(id => {
                const name = speakers[id] || id;
                const unnamed = isUnidentifiedSpeaker(name);
                const color = speakerColorMap[id];
                return `
                    <li class="speaker-row${unnamed ? ' speaker-row-unnamed' : ''}"
                        data-speaker-id="${escapeHtml(id)}"
                        onclick="handleSpeakerRowClick('${escapeHtml(id)}')"
                        title="Jump to first segment and edit">
                        <span class="speaker-row-dot" style="background-color: ${color}"></span>
                        <span class="speaker-row-name">${escapeHtml(unnamed ? 'Unnamed speaker' : name)}</span>
                        ${unnamed ? '<span class="speaker-row-flag" aria-hidden="true">?</span>' : ''}
                    </li>
                `;
            }).join('')}
        </ul>
    `;

    if (fabBadge) {
        if (unnamedCount > 0) {
            fabBadge.textContent = unnamedCount;
            fabBadge.hidden = false;
        } else {
            fabBadge.hidden = true;
        }
    }

    const closeBtn = document.getElementById('speakers-sidebar-close');
    if (closeBtn) closeBtn.addEventListener('click', () => closeSpeakersSidebar());
}

function setupSpeakersSidebar() {
    renderSpeakersSidebar();

    const fab = document.getElementById('speakers-fab');
    if (fab) {
        fab.addEventListener('click', () => {
            const sidebar = document.getElementById('speakers-sidebar');
            if (!sidebar) return;
            const opened = sidebar.classList.toggle('speakers-sidebar-open');
            fab.setAttribute('aria-expanded', opened ? 'true' : 'false');
        });
    }
}

function closeSpeakersSidebar() {
    const sidebar = document.getElementById('speakers-sidebar');
    const fab = document.getElementById('speakers-fab');
    if (sidebar) sidebar.classList.remove('speakers-sidebar-open');
    if (fab) fab.setAttribute('aria-expanded', 'false');
}

function handleSpeakerRowClick(speakerId) {
    const state = window._speakerEditorState;
    if (!state) return;
    const segIndex = state.firstSegmentIndex[speakerId];
    if (segIndex == null) return;

    const segmentEl = document.getElementById(`seg-${segIndex}`);
    if (!segmentEl) return;

    closeSpeakersSidebar();
    segmentEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    userScrolledAway = true;

    const speakerLabel = segmentEl.querySelector('.speaker-label');
    const segId = segmentEl.dataset.segmentId;
    if (speakerLabel && segId) {
        handleSpeakerClick(speakerLabel, speakerId, segId);
    }
}

function setupScrollDetection() {
    const segmentsContainer = document.getElementById('segments-container');
    if (!segmentsContainer) return;

    segmentsContainer.addEventListener('wheel', () => {
        userScrolledAway = true;
        clearTimeout(scrollTimeout);
    });

    segmentsContainer.addEventListener('touchmove', () => {
        userScrolledAway = true;
        clearTimeout(scrollTimeout);
    });

    const checkbox = document.getElementById('auto-scroll-check');
    if (checkbox) {
        checkbox.addEventListener('change', () => {
            autoScroll = checkbox.checked;
            if (autoScroll) userScrolledAway = false;
        });
    }
}

function setupBackToTop() {
    const btn = document.getElementById('back-to-top');
    if (!btn) return;

    backToTopHandler = () => {
        btn.classList.toggle('back-to-top-visible', window.scrollY > 300);
    };
    window.addEventListener('scroll', backToTopHandler, { passive: true });
    backToTopHandler();

    btn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

function playFromSegment(startTime) {
    if (!currentAudio) return;
    userScrolledAway = false;
    currentAudio.currentTime = startTime;
    currentAudio.play();
    const playPause = document.getElementById('play-pause');
    if (playPause) {
        playPause.innerHTML = ICON_PAUSE;
        playPause.setAttribute('aria-label', 'Pause');
    }
}

function highlightCurrentSegment(currentTime) {
    const segments = document.querySelectorAll('.segment');
    let activeSegment = null;

    segments.forEach(seg => {
        const start = parseFloat(seg.dataset.start);
        const end = parseFloat(seg.dataset.end);
        if (currentTime >= start && currentTime < end) {
            seg.classList.add('segment-active');
            activeSegment = seg;
        } else {
            seg.classList.remove('segment-active');
        }
    });

    if (activeSegment && autoScroll && !userScrolledAway) {
        activeSegment.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function renderPlainTextTab(container) {
    const state = window._speakerEditorState;
    if (!state) return;

    const segments = document.querySelectorAll('#segments-container .segment');
    const lines = [];

    segments.forEach(seg => {
        const speakerId = seg.querySelector('.speaker-label').dataset.speaker;
        const speakerName = state.speakers[speakerId] || speakerId;
        const time = seg.querySelector('.segment-time').textContent;
        const text = seg.querySelector('.segment-text').textContent;
        lines.push(`[${time}] ${speakerName}: ${text}`);
    });

    const plainText = lines.join('\n');

    container.innerHTML = `
        ${renderUnnamedSpeakersWarning()}
        <div class="plaintext-view">
            <div class="plaintext-actions">
                <button class="btn btn-primary" onclick="copyPlainText()">Copy to clipboard</button>
            </div>
            <pre class="plaintext-content">${escapeHtml(plainText)}</pre>
        </div>
    `;
}

function copyPlainText() {
    const content = document.querySelector('.plaintext-content');
    if (!content) return;
    copyToClipboard(content.textContent);
}

function switchTab(tabName, meetingId, meetingType) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('tab-active'));
    document.querySelectorAll('.tab-content').forEach(t => { t.hidden = true; t.classList.remove('tab-content-active'); });

    document.querySelector(`[data-tab="${tabName}"]`).classList.add('tab-active');
    const tabEl = document.getElementById(`${tabName}-tab`);
    tabEl.hidden = false;
    tabEl.classList.add('tab-content-active');

    const sidebar = document.getElementById('speakers-sidebar');
    const fab = document.getElementById('speakers-fab');
    const onTranscript = tabName === 'transcript';
    if (sidebar) sidebar.hidden = !onTranscript;
    if (fab) fab.hidden = !onTranscript;
    if (!onTranscript) closeSpeakersSidebar();

    if (tabName === 'plaintext') {
        renderPlainTextTab(tabEl);
    }

    if (tabName === 'overview') {
        renderOverviewTab(tabEl, {
            metadata: window._meetingMeta,
            audio_analysis: window._meetingAudioAnalysis,
            transcript: window._meetingTranscript,
        });
    }

    if (tabName === 'analysis' && !tabEl.dataset.loaded) {
        tabEl.dataset.loaded = 'true';
        renderAnalysisTab(tabEl, meetingId, meetingType);
    }
}

function startPolling(meetingId, jobId) {
    updateProgress(meetingId, jobId);
    pollInterval = setInterval(() => updateProgress(meetingId, jobId), 3000);
}

async function updateProgress(meetingId, jobId) {
    try {
        const job = await API.getJob(jobId);
        const section = document.getElementById('progress-section');
        if (!section) return;

        if (job.status === 'completed') {
            clearInterval(pollInterval);
            sendNotification('Transcription complete', {
                body: 'Your meeting has been transcribed and is ready to view.',
                url: `/meetings/${meetingId}`,
            });
            App.navigate(`/meetings/${meetingId}`);
            return;
        }

        if (job.status === 'failed') {
            clearInterval(pollInterval);
            sendNotification('Transcription failed', {
                body: job.error || 'An error occurred during transcription.',
                url: `/meetings/${meetingId}`,
            });
            section.innerHTML = `
                <div class="error-state">
                    Transcription failed: ${escapeHtml(job.error || 'Unknown error')}
                </div>
            `;
            return;
        }

        const stageLabels = {
            uploading: 'Uploading...',
            preprocessing: 'Preprocessing audio...',
            transcribing: 'Transcribing audio...',
            downloading_align_model: 'Downloading alignment model...',
            aligning: 'Aligning timestamps...',
            diarizing: 'Identifying speakers...',
            saving: 'Saving results...',
        };

        section.innerHTML = `
            <div class="progress-info">
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: ${job.progress}%"></div>
                </div>
                <span class="progress-label">${stageLabels[job.stage] || 'Processing...'} ${job.progress}%</span>
            </div>
        `;
    } catch {
        // Silently continue polling
    }
}

async function retryTranscription(meetingId) {
    try {
        await API.retryTranscription(meetingId);
        showToast('Retrying transcription...');
        App.navigate(`/meetings/${meetingId}`);
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function handleRerunTranscription(meetingId) {
    if (!confirm('Re-run transcription? The existing transcript and analysis will be overwritten.')) return;
    await retryTranscription(meetingId);
}

async function handleCancelTranscription(meetingId) {
    if (!confirm('Are you sure? This will stop the current transcription.')) return;
    try {
        await API.cancelTranscription(meetingId);
        showToast('Transcription cancelled');
        App.navigate(`/meetings/${meetingId}`);
    } catch (err) {
        showToast('Failed to cancel transcription', 'error');
    }
}

async function handleDeleteMeeting(meetingId) {
    if (!confirm('Delete this meeting and all its files?')) return;
    try {
        await API.deleteMeeting(meetingId);
        showToast('Meeting deleted');
        App.navigate('/');
    } catch (err) {
        showToast('Failed to delete meeting', 'error');
    }
}

function cleanupTranscriptView() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    userScrolledAway = false;
    clearTimeout(scrollTimeout);
    if (backToTopHandler) {
        window.removeEventListener('scroll', backToTopHandler);
        backToTopHandler = null;
    }
    closeSpeakerPopover();
}
