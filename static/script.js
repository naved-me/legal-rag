document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const messagesWrapper = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('send-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const apiKeyInput = document.getElementById('api-key-input');
    const sessionListEl = document.getElementById('session-list');

    // Store the session ID to maintain conversational memory
    let currentSessionId = null;

    // API key (optional) persisted locally; sent as Authorization: Bearer <key>
    const savedKey = localStorage.getItem('apiKey') || '';
    if (apiKeyInput) apiKeyInput.value = savedKey;
    if (apiKeyInput) {
        apiKeyInput.addEventListener('change', () => {
            const key = apiKeyInput.value.trim();
            if (key) localStorage.setItem('apiKey', key);
            else localStorage.removeItem('apiKey');
        });
    }

    function getHeaders() {
        const headers = { 'Content-Type': 'application/json' };
        const apiKey = (apiKeyInput ? apiKeyInput.value.trim() : '') || localStorage.getItem('apiKey') || '';
        if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
        return headers;
    }

    // Escape untrusted text before inserting into the DOM (prevents XSS).
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

    // Scroll to the bottom of the chat
    function scrollToBottom() {
        messagesWrapper.scrollTop = messagesWrapper.scrollHeight;
    }

    // Add a message to the chat UI
    function addMessage(text, isUser, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;

        let avatar = isUser ? 'ME' : 'AI';

        let contentHTML = `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;

        // If there are sources, add them as tags below the text
        if (sources.length > 0) {
            contentHTML += `<div class="sources"><strong>Sources:</strong><br>`;
            // Extract unique pages
            const pages = [...new Set(sources.map(s => s.page))].filter(p => p !== 0);
            pages.forEach(p => {
                contentHTML += `<span class="source-tag">Page ${escapeHtml(p)}</span>`;
            });
            contentHTML += `</div>`;
        }

        messageDiv.innerHTML = `
            <div class="avatar">${avatar}</div>
            <div class="bubble">
                ${contentHTML}
            </div>
        `;

        messagesWrapper.appendChild(messageDiv);
        scrollToBottom();
    }

    // Show a loading typing indicator
    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message ai-message';
        loadingDiv.id = 'loading-indicator';
        loadingDiv.innerHTML = `
            <div class="avatar">AI</div>
            <div class="bubble">
                <div class="typing-indicator">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        messagesWrapper.appendChild(loadingDiv);
        scrollToBottom();
    }

    // Remove the loading typing indicator
    function removeLoading() {
        const loadingDiv = document.getElementById('loading-indicator');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    function renderSessions(list, activeId) {
        sessionListEl.innerHTML = '';
        if (!list || list.length === 0) {
            sessionListEl.innerHTML = '<div class="session-empty">No chats yet</div>';
            return;
        }
        list.forEach(s => {
            const item = document.createElement('div');
            item.className = 'session-item' + (s.session_id === activeId ? ' active' : '');
            item.dataset.id = s.session_id;

            const title = document.createElement('span');
            title.className = 'session-title';
            title.textContent = s.title || 'New chat';

            const del = document.createElement('button');
            del.className = 'session-delete';
            del.title = 'Delete chat';
            del.textContent = '✕';
            del.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteSession(s.session_id);
            });

            item.addEventListener('click', () => openSession(s.session_id));
            item.appendChild(title);
            item.appendChild(del);
            sessionListEl.appendChild(item);
        });
    }

    async function loadSessions() {
        try {
            const resp = await fetch('/sessions', { headers: getHeaders() });
            if (!resp.ok) throw new Error('failed');
            renderSessions(await resp.json(), currentSessionId);
        } catch (err) {
            console.error('Failed to load sessions:', err);
        }
    }

    function clearChat(welcomeText) {
        messagesWrapper.innerHTML = `
            <div class="message ai-message welcome-message">
                <div class="avatar">AI</div>
                <div class="bubble">
                    <p>${escapeHtml(welcomeText)}</p>
                </div>
            </div>
        `;
    }

    // Create a fresh thread WITHOUT clearing existing ones (ChatGPT-style).
    async function newChat() {
        try {
            const resp = await fetch('/sessions', { method: 'POST', headers: getHeaders() });
            if (!resp.ok) throw new Error('failed');
            const data = await resp.json();
            currentSessionId = data.session_id;
            clearChat('New chat started. How can I help you?');
            loadSessions();
        } catch (err) {
            console.error('Failed to create session:', err);
        }
    }

    // Switch to an existing thread and re-render its history.
    async function openSession(sessionId) {
        try {
            const resp = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, { headers: getHeaders() });
            if (!resp.ok) throw new Error('failed');
            const data = await resp.json();
            currentSessionId = sessionId;
            messagesWrapper.innerHTML = '';
            if (!data.messages || data.messages.length === 0) {
                clearChat('New chat started. How can I help you?');
            } else {
                data.messages.forEach(m => addMessage(m.content, m.role === 'human'));
            }
            loadSessions();
        } catch (err) {
            console.error('Failed to open session:', err);
        }
    }

    async function deleteSession(sessionId) {
        try {
            const resp = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                method: 'DELETE',
                headers: getHeaders(),
            });
            if (!resp.ok && resp.status !== 404) throw new Error('failed');
            if (sessionId === currentSessionId) {
                currentSessionId = null;
                clearChat('Chat deleted. How can I help you?');
            }
            loadSessions();
        } catch (err) {
            console.error('Failed to delete session:', err);
        }
    }

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const question = userInput.value.trim();
        if (!question) return;

        // 1. Display user message
        addMessage(question, true);

        // 2. Clear input & disable button
        userInput.value = '';
        sendBtn.disabled = true;

        // 3. Show loading indicator
        showLoading();

        try {
            // 4. Call our FastAPI backend
            const payload = { question: question };
            if (currentSessionId) {
                payload.session_id = currentSessionId;
            }

            const response = await fetch('/ask', {
                method: 'POST',
                headers: getHeaders(),
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }

            const data = await response.json();

            // Save the session ID so the server remembers us next time
            currentSessionId = data.session_id;

            // 5. Remove loading and display AI answer
            removeLoading();
            addMessage(data.answer, false, data.sources);

            // 6. Refresh the sidebar (new session / updated title & order)
            loadSessions();

        } catch (error) {
            console.error('Error:', error);
            removeLoading();
            addMessage('Sorry, I encountered an error connecting to the server. Is the FastAPI backend running?', false);
        } finally {
            // Re-enable input
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    newChatBtn.addEventListener('click', newChat);

    // Load the sidebar on startup
    loadSessions();
});