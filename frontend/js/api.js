const API = {
    async listMeetings() {
        const res = await fetch('/api/meetings');
        if (!res.ok) throw new Error('Failed to fetch meetings');
        return res.json();
    },

    async getMeeting(id) {
        const res = await fetch(`/api/meetings/${id}`);
        if (!res.ok) throw new Error('Failed to fetch meeting');
        return res.json();
    },

    async createMeeting(file, title, meetingType, expectedLanguages = [], numSpeakers, preprocessAudio = true, context = '', audioAnalysisEnabled = false) {
        const form = new FormData();
        form.append('file', file);
        form.append('title', title || '');
        form.append('meeting_type', meetingType || 'other');
        // Zero, one, or many expected languages. None selected → backend auto-detects
        // a single language; two or more → per-chunk multilingual transcription.
        (expectedLanguages || []).forEach(code => form.append('expected_languages', code));
        form.append('num_speakers', numSpeakers || 'auto');
        form.append('preprocess_audio', preprocessAudio ? 'true' : 'false');
        form.append('audio_analysis_enabled', audioAnalysisEnabled ? 'true' : 'false');
        form.append('context', context || '');
        const res = await fetch('/api/meetings', { method: 'POST', body: form });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        return res.json();
    },

    async updateMeeting(id, data) {
        const res = await fetch(`/api/meetings/${id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error('Failed to update meeting');
        return res.json();
    },

    async cancelTranscription(id) {
        const res = await fetch(`/api/meetings/${id}/cancel`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to cancel transcription');
        return res.json();
    },

    async retryTranscription(id) {
        const res = await fetch(`/api/meetings/${id}/retry`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to retry transcription');
        return res.json();
    },

    async deleteMeeting(id) {
        const res = await fetch(`/api/meetings/${id}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to delete meeting');
        return res.json();
    },

    async getJob(jobId) {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error('Failed to fetch job status');
        return res.json();
    },

    async getTemplate(templateType) {
        const res = await fetch(`/api/templates/${templateType}`);
        if (!res.ok) throw new Error('Failed to fetch template');
        return res.json();
    },

    async getAnalysisContext(meetingId) {
        const res = await fetch(`/api/meetings/${meetingId}/analysis-context`);
        if (!res.ok) throw new Error('Failed to fetch analysis context');
        return res.json();
    },

    async getAnalysisPrompt(meetingId, templateType, meetingContext) {
        const params = new URLSearchParams({ template_type: templateType });
        if (meetingContext != null) params.set('meeting_context', meetingContext);
        const res = await fetch(`/api/meetings/${meetingId}/analysis-prompt?${params.toString()}`);
        if (!res.ok) {
            let detail = 'Failed to generate prompt';
            try {
                const body = await res.json();
                if (body.detail) detail = body.detail;
            } catch (e) { /* keep default */ }
            throw new Error(detail);
        }
        return res.json();
    },

    async updateSegmentSpeaker(meetingId, segmentId, speakerName) {
        const res = await fetch(`/api/meetings/${meetingId}/segments/speaker`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ segment_id: segmentId, speaker_name: speakerName }),
        });
        if (!res.ok) throw new Error('Failed to update segment speaker');
        return res.json();
    },

    audioUrl(meetingId) {
        return `/api/meetings/${meetingId}/audio`;
    },
};
