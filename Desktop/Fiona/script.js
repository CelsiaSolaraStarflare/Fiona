// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const attachBtn = document.getElementById('attachBtn');
const charCount = document.getElementById('charCount');
const typingIndicator = document.getElementById('typingIndicator');
const chatHistory = document.getElementById('chatHistory');
const historyList = document.getElementById('historyList');
const createChatBtn = document.getElementById('createChatBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');


// State
let isTyping = false;
let messageQueue = [];
let chatSessions = [];
let currentSession = {
    id: 'current',
    title: 'Current Session',
    messages: [],
    lastMessage: 'Hello! I\'m your AI assistant...',
    timestamp: new Date()
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    autoResizeTextarea();
    updateCharCount();
    
    // Add some initial messages for demo
    setTimeout(() => {
        addBotMessage("I'm here to help you with any questions or tasks you might have. Feel free to ask me anything!");
    }, 1000);
});

// Event Listeners
function initializeEventListeners() {
    // Send message on button click
    sendBtn.addEventListener('click', sendMessage);
    
    // Send message on Enter (but allow Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        autoResizeTextarea();
        updateCharCount();
        updateSendButton();
    });
    
    // Focus management
    messageInput.addEventListener('focus', () => {
        messageInput.parentElement.style.transform = 'translateY(-2px)';
    });
    
    messageInput.addEventListener('blur', () => {
        if (!messageInput.value.trim()) {
            messageInput.parentElement.style.transform = 'translateY(0)';
        }
    });
    
    // Attachment button
    attachBtn.addEventListener('click', () => {
        showAttachmentMenu();
    });
    
    // Create new chat button
    createChatBtn.addEventListener('click', () => {
        createNewSession();
    });
    
    // Clear history button
    clearHistoryBtn.addEventListener('click', () => {
        clearChatHistory();
    });
    
    // History item clicks
    historyList.addEventListener('click', (e) => {
        const historyItem = e.target.closest('.history-item');
        if (historyItem) {
            const sessionId = historyItem.dataset.session;
            loadChatSession(sessionId);
        }
    });
    

}

// Message Functions
function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isTyping) return;
    
    // Add user message
    addUserMessage(message);
    
    // Clear input
    messageInput.value = '';
    autoResizeTextarea();
    updateCharCount();
    updateSendButton();
    
    // Simulate bot response
    simulateBotResponse(message);
}

function addUserMessage(text) {
    const messageElement = createMessageElement(text, 'user');
    chatMessages.appendChild(messageElement);
    
    // Animate in
    requestAnimationFrame(() => {
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translateY(0)';
    });
    
    // Add to current session
    currentSession.messages.push({ type: 'user', text, timestamp: new Date() });
    currentSession.lastMessage = text;
    currentSession.timestamp = new Date();
    updateHistoryItem();
    
    scrollToBottom();
}

function addBotMessage(text) {
    const messageElement = createMessageElement(text, 'bot');
    chatMessages.appendChild(messageElement);
    
    // Animate in
    requestAnimationFrame(() => {
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translateY(0)';
    });
    
    // Add to current session
    currentSession.messages.push({ type: 'bot', text, timestamp: new Date() });
    currentSession.lastMessage = text;
    currentSession.timestamp = new Date();
    updateHistoryItem();
    
    scrollToBottom();
}

function createMessageElement(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';
    messageDiv.style.transition = 'all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `<p>${escapeHtml(text)}</p>`;
    
    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = getCurrentTime();
    
    content.appendChild(bubble);
    content.appendChild(time);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    return messageDiv;
}

// Bot Response Simulation
function simulateBotResponse(userMessage) {
    isTyping = true;
    showTypingIndicator();
    
    // Simulate typing delay
    const typingDelay = Math.min(userMessage.length * 50, 2000);
    
    setTimeout(() => {
        hideTypingIndicator();
        isTyping = false;
        
        // Generate response based on user input
        const response = generateBotResponse(userMessage);
        addBotMessage(response);
    }, typingDelay);
}

function generateBotResponse(userMessage) {
    const lowerMessage = userMessage.toLowerCase();
    
    // Simple response logic
    if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
        return "Hello! How can I assist you today?";
    } else if (lowerMessage.includes('help')) {
        return "I'm here to help! I can answer questions, provide information, or just chat with you. What would you like to know?";
    } else if (lowerMessage.includes('weather')) {
        return "I'd be happy to help with weather information! However, I don't have access to real-time weather data. You might want to check a weather app or website for current conditions.";
    } else if (lowerMessage.includes('time')) {
        return `The current time is ${new Date().toLocaleTimeString()}.`;
    } else if (lowerMessage.includes('thank')) {
        return "You're welcome! I'm glad I could help. Is there anything else you'd like to know?";
    } else if (lowerMessage.includes('bye') || lowerMessage.includes('goodbye')) {
        return "Goodbye! Feel free to come back anytime if you have more questions.";
    } else if (lowerMessage.includes('name')) {
        return "I'm your AI assistant! I don't have a specific name, but I'm here to help you with whatever you need.";
    } else if (lowerMessage.includes('joke')) {
        const jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a fake noodle? An impasta!",
            "Why did the math book look so sad? Because it had too many problems!",
            "What do you call a bear with no teeth? A gummy bear!"
        ];
        return jokes[Math.floor(Math.random() * jokes.length)];
    } else {
        const responses = [
            "That's interesting! Tell me more about that.",
            "I understand what you're saying. How can I help you with this?",
            "Thanks for sharing that with me. Is there anything specific you'd like to know?",
            "I'm here to help! Could you clarify what you're looking for?",
            "That's a great point! What would you like to explore further?"
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }
}

// Typing Indicator
function showTypingIndicator() {
    typingIndicator.classList.add('show');
    scrollToBottom();
}

function hideTypingIndicator() {
    typingIndicator.classList.remove('show');
}

// Utility Functions
function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

function updateCharCount() {
    const count = messageInput.value.length;
    charCount.textContent = `${count}/500`;
    
    // Update color based on count
    charCount.className = 'char-count';
    if (count > 400) {
        charCount.classList.add('warning');
    }
    if (count > 480) {
        charCount.classList.add('error');
    }
}

function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.classList.toggle('active', hasText);
}

function scrollToBottom() {
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 100);
}

function getCurrentTime() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes.toString().padStart(2, '0');
    return `${displayHours}:${displayMinutes} ${ampm}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Attachment Menu
function showAttachmentMenu() {
    // Create attachment menu
    const menu = document.createElement('div');
    menu.className = 'attachment-menu';
    menu.style.cssText = `
        position: absolute;
        bottom: 100%;
        left: 0;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        padding: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0;
        transform: translateY(10px) scale(0.95);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        z-index: 1000;
    `;
    
    menu.innerHTML = `
        <div class="attachment-option" data-type="file">
            <i class="fas fa-file"></i>
            <span>File</span>
        </div>
        <div class="attachment-option" data-type="image">
            <i class="fas fa-image"></i>
            <span>Image</span>
        </div>
        <div class="attachment-option" data-type="camera">
            <i class="fas fa-camera"></i>
            <span>Camera</span>
        </div>
    `;
    
    // Add styles for attachment options
    const style = document.createElement('style');
    style.textContent = `
        .attachment-option {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.2s ease;
            color: #374151;
        }
        .attachment-option:hover {
            background-color: #f3f4f6;
        }
        .attachment-option i {
            font-size: 1.2rem;
            color: #667eea;
        }
    `;
    document.head.appendChild(style);
    
    // Position menu
    const inputWrapper = attachBtn.closest('.input-wrapper');
    inputWrapper.style.position = 'relative';
    inputWrapper.appendChild(menu);
    
    // Animate in
    requestAnimationFrame(() => {
        menu.style.opacity = '1';
        menu.style.transform = 'translateY(0) scale(1)';
    });
    
    // Handle clicks
    menu.addEventListener('click', (e) => {
        const option = e.target.closest('.attachment-option');
        if (option) {
            const type = option.dataset.type;
            handleAttachment(type);
            hideAttachmentMenu(menu);
        }
    });
    
    // Hide on outside click
    setTimeout(() => {
        document.addEventListener('click', hideOnOutsideClick);
    }, 100);
    
    function hideOnOutsideClick(e) {
        if (!menu.contains(e.target) && !attachBtn.contains(e.target)) {
            hideAttachmentMenu(menu);
            document.removeEventListener('click', hideOnOutsideClick);
        }
    }
}

function hideAttachmentMenu(menu) {
    menu.style.opacity = '0';
    menu.style.transform = 'translateY(10px) scale(0.95)';
    setTimeout(() => {
        menu.remove();
    }, 300);
}

function handleAttachment(type) {
    const messages = {
        file: "File attachment feature would be implemented here!",
        image: "Image upload feature would be implemented here!",
        camera: "Camera access would be requested here!"
    };
    
    addUserMessage(`[${type.toUpperCase()}] ${messages[type]}`);
    
    // Simulate bot response
    setTimeout(() => {
        addBotMessage(`I see you want to attach a ${type}. In a real application, this would open the file picker or camera interface.`);
    }, 1000);
}

// Smooth scroll polyfill for better performance
function smoothScrollTo(element, target, duration = 300) {
    const start = element.scrollTop;
    const change = target - start;
    const startTime = performance.now();
    
    function animateScroll(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        
        element.scrollTop = start + change * easeOut;
        
        if (progress < 1) {
            requestAnimationFrame(animateScroll);
        }
    }
    
    requestAnimationFrame(animateScroll);
}

// Enhanced scroll to bottom with smooth animation
function smoothScrollToBottom() {
    const target = chatMessages.scrollHeight - chatMessages.clientHeight;
    smoothScrollTo(chatMessages, target, 500);
}

// Override the original scroll function
const originalScrollToBottom = scrollToBottom;
scrollToBottom = smoothScrollToBottom;

// Custom Glass Cursor
let cursor = null;

function createCursor() {
    cursor = document.createElement('div');
    cursor.className = 'custom-cursor';
    document.body.appendChild(cursor);
}

function updateCursor(e) {
    if (cursor) {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
    }
}

function handleCursorHover(e) {
    if (cursor) {
        const target = e.target;
        const computedStyle = getComputedStyle(target);
        const cursorType = computedStyle.cursor;
        
        // Remove all cursor classes first
        cursor.classList.remove('hand', 'selecting', 'scrolling');
        
        // Apply appropriate cursor class based on CSS cursor property
        if (cursorType === 'pointer' || 
            target.tagName === 'BUTTON' || 
            target.tagName === 'A' || 
            target.closest('button') || 
            target.closest('a')) {
            cursor.classList.add('hand');
        } else if (cursorType === 'text' || cursorType === 'vertical-text') {
            cursor.classList.add('selecting');
        } else if (cursorType === 'grab' || cursorType === 'grabbing' || 
                   cursorType === 'move' || cursorType === 'ns-resize' || 
                   cursorType === 'ew-resize' || cursorType === 'nw-resize' || 
                   cursorType === 'ne-resize' || cursorType === 'sw-resize' || 
                   cursorType === 'se-resize') {
            cursor.classList.add('scrolling');
        }
    }
}

// Initialize cursor
document.addEventListener('DOMContentLoaded', () => {
    createCursor();
    document.addEventListener('mousemove', updateCursor);
    document.addEventListener('mouseover', handleCursorHover);
    
    // Hide default cursor
    document.body.style.cursor = 'none';
    
    // Show default cursor on touch devices
    if ('ontouchstart' in window) {
        document.body.style.cursor = 'auto';
        if (cursor) cursor.style.display = 'none';
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        messageInput.value = '';
        autoResizeTextarea();
        updateCharCount();
        updateSendButton();
        messageInput.blur();
    }
});

// Chat History Functions
function updateHistoryItem() {
    const currentItem = historyList.querySelector('[data-session="current"]');
    if (currentItem) {
        const preview = currentItem.querySelector('.history-preview');
        const time = currentItem.querySelector('.history-time');
        
        preview.textContent = currentSession.lastMessage.length > 30 
            ? currentSession.lastMessage.substring(0, 30) + '...' 
            : currentSession.lastMessage;
        time.textContent = formatTime(currentSession.timestamp);
    }
}

function createNewSession() {
    // Save current session if it has messages
    if (currentSession.messages.length > 2) { // More than just the welcome message
        const sessionId = 'session_' + Date.now();
        const session = {
            ...currentSession,
            id: sessionId,
            title: `Chat ${chatSessions.length + 1}`
        };
        chatSessions.push(session);
        addHistoryItem(session);
    }
    
    // Create new current session
    currentSession = {
        id: 'current',
        title: 'Current Session',
        messages: [],
        lastMessage: 'Hello! I\'m your AI assistant...',
        timestamp: new Date()
    };
    
    // Clear chat messages
    chatMessages.innerHTML = `
        <div class="message bot-message">
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <p>Hello! I'm your AI assistant. How can I help you today?</p>
                </div>
                <div class="message-time">Just now</div>
            </div>
        </div>
    `;
    
    updateHistoryItem();
}

function addHistoryItem(session) {
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.dataset.session = session.id;
    
    historyItem.innerHTML = `
        <div class="history-icon">
            <i class="fas fa-comments"></i>
        </div>
        <div class="history-content">
            <div class="history-title">${session.title}</div>
            <div class="history-preview">${session.lastMessage.length > 30 
                ? session.lastMessage.substring(0, 30) + '...' 
                : session.lastMessage}</div>
        </div>
        <div class="history-time">${formatTime(session.timestamp)}</div>
    `;
    
    // Insert after the current session item
    const currentItem = historyList.querySelector('[data-session="current"]');
    if (currentItem) {
        currentItem.parentNode.insertBefore(historyItem, currentItem.nextSibling);
    } else {
        historyList.appendChild(historyItem);
    }
}

function loadChatSession(sessionId) {
    if (sessionId === 'current') return;
    
    const session = chatSessions.find(s => s.id === sessionId);
    if (!session) return;
    
    // Update active state
    document.querySelectorAll('.history-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`[data-session="${sessionId}"]`).classList.add('active');
    
    // Load messages
    chatMessages.innerHTML = '';
    session.messages.forEach(msg => {
        const messageElement = createMessageElement(msg.text, msg.type);
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translateY(0)';
        chatMessages.appendChild(messageElement);
    });
    
    scrollToBottom();
}

function clearChatHistory() {
    if (confirm('Are you sure you want to clear all chat history?')) {
        chatSessions = [];
        historyList.innerHTML = `
            <div class="history-item active" data-session="current">
                <div class="history-icon">
                    <i class="fas fa-comments"></i>
                </div>
                <div class="history-content">
                    <div class="history-title">Current Session</div>
                    <div class="history-preview">Hello! I'm your AI assistant...</div>
                </div>
                <div class="history-time">Just now</div>
            </div>
        `;
        
        // Reset current session
        currentSession = {
            id: 'current',
            title: 'Current Session',
            messages: [],
            lastMessage: 'Hello! I\'m your AI assistant...',
            timestamp: new Date()
        };
        
        // Clear chat messages
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <p>Hello! I'm your AI assistant. How can I help you today?</p>
                    </div>
                    <div class="message-time">Just now</div>
                </div>
            </div>
        `;
    }
}

function formatTime(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
}

// Dynamic Background Colors
function generateRandomColor() {
    const colors = [
        'rgba(255, 119, 198, 0.4)',   // Pink
        'rgba(120, 219, 255, 0.4)',   // Blue
        'rgba(120, 119, 198, 0.4)',   // Purple
        'rgba(255, 193, 7, 0.4)',     // Yellow
        'rgba(76, 175, 80, 0.4)',     // Green
        'rgba(244, 67, 54, 0.4)',     // Red
        'rgba(156, 39, 176, 0.4)',    // Deep Purple
        'rgba(0, 188, 212, 0.4)',     // Cyan
        'rgba(255, 87, 34, 0.4)',     // Deep Orange
        'rgba(103, 58, 183, 0.4)'     // Indigo
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

function createFloatingOrb() {
    const orb = document.createElement('div');
    orb.className = 'floating-orb';
    
    // Random position
    const x = Math.random() * 100;
    const y = Math.random() * 100;
    
    // Fixed size for consistency
    const size = 200 + Math.random() * 100;
    
    // Random color
    const color = generateRandomColor();
    
    // Random animation delay
    const delay = Math.random() * 10;
    
    orb.style.cssText = `
        position: fixed;
        width: ${size}px;
        height: ${size}px;
        background: radial-gradient(circle, ${color} 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(40px);
        opacity: 0.6;
        animation: orbFloat 20s ease-in-out infinite;
        animation-delay: -${delay}s;
        pointer-events: none;
        z-index: 1;
        top: ${y}%;
        left: ${x}%;
        transform-origin: center;
    `;
    
    return orb;
}

function updateOrbColors() {
    const orbs = document.querySelectorAll('.floating-orb');
    orbs.forEach(orb => {
        const color = generateRandomColor();
        orb.style.background = `radial-gradient(circle, ${color} 0%, transparent 70%)`;
    });
}

// Add loading animation for initial messages
window.addEventListener('load', () => {
    const messages = document.querySelectorAll('.message');
    messages.forEach((message, index) => {
        message.style.animationDelay = `${index * 0.2}s`;
    });
    
    // Create multiple floating orbs
    for (let i = 0; i < 6; i++) {
        const orb = createFloatingOrb();
        document.body.appendChild(orb);
    }
    
    // Update orb colors every 15 seconds
    setInterval(updateOrbColors, 15000);
    
    // Create new orbs periodically
    setInterval(() => {
        const existingOrbs = document.querySelectorAll('.floating-orb');
        if (existingOrbs.length < 8) {
            const newOrb = createFloatingOrb();
            document.body.appendChild(newOrb);
            
            // Remove old orbs to prevent too many
            setTimeout(() => {
                if (newOrb.parentNode) {
                    newOrb.remove();
                }
            }, 25000);
        }
    }, 12000);
}); 