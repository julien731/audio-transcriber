function renderAnalysisTab(container, meetingId, meetingType) {
    if (meetingId) container.dataset.meetingId = meetingId;
    container.innerHTML = `
        ${renderUnnamedSpeakersWarning()}
        <div class="analysis-generator">
            <p>Select a template to generate a prompt you can paste into any LLM for analysis.</p>
            <div class="form-group">
                <label for="analysis-type">Template</label>
                <select id="analysis-type">
                    <option value="interview" ${meetingType === 'interview' ? 'selected' : ''}>Interview</option>
                    <option value="sales" ${meetingType === 'sales' ? 'selected' : ''}>Sales</option>
                    <option value="client" ${meetingType === 'client' ? 'selected' : ''}>Client</option>
                    <option value="other" ${meetingType === 'other' ? 'selected' : ''}>Other</option>
                    <option value="prototype">Prototype Scope</option>
                </select>
            </div>
            <button class="btn btn-primary" id="generate-prompt-btn" onclick="handleGeneratePrompt()">
                Generate Prompt
            </button>
        </div>
    `;
}

async function handleGeneratePrompt() {
    const btn = document.getElementById('generate-prompt-btn');
    const type = document.getElementById('analysis-type').value;
    const tabContainer = document.getElementById('analysis-tab');
    const meetingId = tabContainer ? tabContainer.dataset.meetingId : '';
    btn.disabled = true;
    btn.textContent = 'Generating...';

    try {
        // Prompt assembly (template selection, audio/meeting context and
        // transcript substitution) lives server-side so any client stays thin
        // (BR-16, BR-17). We only pass the live meeting-context textarea value.
        const { prompt } = await API.getAnalysisPrompt(meetingId, type, getMeetingContext());
        renderPromptContent(tabContainer, prompt);
    } catch (err) {
        showToast(err.message, 'error');
        btn.disabled = false;
        btn.textContent = 'Generate Prompt';
    }
}

function getMeetingContext() {
    const textarea = document.getElementById('meeting-context');
    return textarea ? textarea.value.trim() : '';
}

function renderPromptContent(container, prompt) {
    container.innerHTML = `
        ${renderUnnamedSpeakersWarning()}
        <div class="analysis-content">
            <div class="analysis-actions">
                <button class="btn btn-primary" onclick="copyPrompt()">Copy to clipboard</button>
                <button class="btn btn-text" onclick="renderAnalysisTab(this.closest('.tab-content'), '', document.getElementById('analysis-type')?.value || 'other')">Back</button>
            </div>
            <pre class="plaintext-content">${escapeHtml(prompt)}</pre>
        </div>
    `;
    container.dataset.rawPrompt = prompt;
}

function copyPrompt() {
    const container = document.getElementById('analysis-tab');
    const raw = container.dataset.rawPrompt || '';
    copyToClipboard(raw);
}
