// Socket.IO connection
const socket = io();

// Configuration state
let config = {
    video_source: null,
    conf_threshold: 0.4,
    alert_severities: ['Critical', 'High'],
    enable_email: true,
    enable_sms: true
};

let detectionRunning = false;
let accidentTypeChart = null;
let severityChart = null;
let uploadedVideoPath = null;
let liveSource = null;

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const configBtn = document.getElementById('configBtn');
const uploadBtn = document.getElementById('uploadBtn');
const liveBtn = document.getElementById('liveBtn');
const configModal = document.getElementById('configModal');
const uploadModal = document.getElementById('uploadModal');
const liveModal = document.getElementById('liveModal');
const closeModal = document.querySelector('.close');
const closeUploadModal = document.querySelector('.close-upload');
const closeLiveModal = document.querySelector('.close-live');
const saveConfigBtn = document.getElementById('saveConfig');
const videoCanvas = document.getElementById('videoCanvas');
const videoStatus = document.getElementById('videoStatus');
const ctx = videoCanvas.getContext('2d');
const uploadArea = document.getElementById('uploadArea');
const videoFileInput = document.getElementById('videoFileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadSuccess = document.getElementById('uploadSuccess');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const uploadedFileName = document.getElementById('uploadedFileName');
const startDetectionFromUpload = document.getElementById('startDetectionFromUpload');
const liveSuccess = document.getElementById('liveSuccess');
const liveSourceInfo = document.getElementById('liveSourceInfo');
const startLiveDetection = document.getElementById('startLiveDetection');
const rtspUrl = document.getElementById('rtspUrl');

// Sidebar navigation
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const section = item.dataset.section;
        
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');
        
        // Update active section
        document.querySelectorAll('.user-section').forEach(sec => sec.classList.remove('active'));
        document.getElementById(section).classList.add('active');
        
        // Load section data
        if (section === 'logs') loadLogs();
        if (section === 'sessions') loadSessions();
        if (section === 'statistics') loadStatistics();
        if (section === 'clips') loadClips();
        if (section === 'uploads') loadUploads();
        if (section === 'detection') loadUserStats();
    });
});

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('video_frame', (data) => {
    const img = new Image();
    img.onload = () => {
        videoCanvas.width = img.width;
        videoCanvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        videoStatus.style.display = 'none';
        videoCanvas.style.display = 'block';
    };
    img.src = 'data:image/jpeg;base64,' + data.frame;
});

socket.on('stats_update', (stats) => {
    document.getElementById('totalDetections').textContent = stats.total_detections;
    document.getElementById('criticalAlerts').textContent = stats.critical_alerts;
    document.getElementById('processingFPS').textContent = stats.fps;
});

socket.on('preview_started', (data) => {
    console.log(data.message);
    stopBtn.disabled = false;
    stopBtn.style.display = 'inline-block';
    videoStatus.textContent = 'Preview mode. Click Start Detection to begin analysis.';
});

socket.on('preview_stopped', (data) => {
    console.log(data.message);
});

socket.on('detection_started', (data) => {
    console.log(data.message);
    detectionRunning = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    stopBtn.style.display = 'inline-block';
    videoStatus.textContent = 'Loading video feed...';
});

socket.on('detection_stopped', (data) => {
    console.log(data.message);
    detectionRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    stopBtn.style.display = 'none';
    videoCanvas.style.display = 'none';
    videoStatus.style.display = 'block';
    videoStatus.textContent = 'Detection stopped. Select a video source to continue.';
});

socket.on('logs_cleared', (data) => {
    console.log(data.message);
    loadLogs();
});

socket.on('clip_deleted', (data) => {
    console.log('Clip deleted:', data.filename);
    loadClips();
});

socket.on('error', (data) => {
    console.error('Socket error:', data);
    alert('Error: ' + data.message);
    detectionRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(tabName).classList.add('active');
        
        if (tabName === 'logs') loadLogs();
        if (tabName === 'statistics') loadStatistics();
        if (tabName === 'clips') loadClips();
    });
});

// Configuration modal
configBtn.addEventListener('click', () => {
    configModal.style.display = 'block';
});

closeModal.addEventListener('click', () => {
    configModal.style.display = 'none';
});

// Upload modal
uploadBtn.addEventListener('click', () => {
    uploadModal.style.display = 'block';
    // Always reset to allow new uploads
    resetUploadModal();
});

closeUploadModal.addEventListener('click', () => {
    uploadModal.style.display = 'none';
    // Reset modal when closing
    resetUploadModal();
});

// Live detection modal
liveBtn.addEventListener('click', () => {
    liveModal.style.display = 'block';
    // Don't reset if already configured
    if (!liveSource) {
        resetLiveModal();
    }
});

closeLiveModal.addEventListener('click', () => {
    liveModal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === configModal) {
        configModal.style.display = 'none';
    }
    if (e.target === uploadModal) {
        uploadModal.style.display = 'none';
        resetUploadModal();
    }
    if (e.target === liveModal) {
        liveModal.style.display = 'none';
    }
});

// Reset live modal
function resetLiveModal() {
    document.querySelector('.live-options').style.display = 'grid';
    liveSuccess.style.display = 'none';
    rtspUrl.value = '';
}

// Reset upload modal
function resetUploadModal() {
    uploadArea.style.display = 'block';
    uploadProgress.style.display = 'none';
    uploadSuccess.style.display = 'none';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading... 0%';
}

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

// Click on upload area to browse files (but not on the label)
uploadArea.addEventListener('click', (e) => {
    // Don't trigger if clicking on the label itself
    if (e.target.tagName !== 'LABEL') {
        videoFileInput.click();
    }
});

videoFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
    // Reset input so same file can be selected again
    e.target.value = '';
});

// Handle file upload
function handleFileUpload(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid video file (MP4, AVI, MOV, MKV)');
        return;
    }

    // Show progress
    uploadArea.style.display = 'none';
    uploadProgress.style.display = 'block';

    const formData = new FormData();
    formData.append('video', file);

    const xhr = new XMLHttpRequest();

    // Progress tracking
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            progressFill.style.width = percentComplete + '%';
            progressText.textContent = `Uploading... ${Math.round(percentComplete)}%`;
        }
    });

    // Upload complete
    xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            if (response.status === 'success') {
                uploadedVideoPath = response.filepath;
                config.video_source = response.filepath;
                
                uploadProgress.style.display = 'none';
                uploadSuccess.style.display = 'block';
                uploadedFileName.textContent = file.name;
                
                // Enable start button
                startBtn.disabled = false;
                
                // Clear the uploaded path variable to allow new uploads
                uploadedVideoPath = response.filepath;
                
                // Refresh uploads list
                loadUploads();
            } else {
                alert('Upload failed: ' + response.message);
                resetUploadModal();
            }
        } else {
            alert('Upload failed. Please try again.');
            resetUploadModal();
        }
    });

    // Upload error
    xhr.addEventListener('error', () => {
        alert('Upload failed. Please try again.');
        resetUploadModal();
    });

    xhr.open('POST', '/upload_video');
    xhr.send(formData);
}

// Start detection from upload modal
startDetectionFromUpload.addEventListener('click', () => {
    uploadModal.style.display = 'none';
    // Reset modal for next upload
    resetUploadModal();
    // Automatically start detection
    if (config.video_source) {
        socket.emit('start_detection', config);
    }
});

// Live detection source selection
document.querySelectorAll('.select-live').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const source = e.target.dataset.source;
        
        if (source === '0') {
            // Webcam
            liveSource = 'webcam';
            config.video_source = 'webcam';
            
            document.querySelector('.live-options').style.display = 'none';
            liveSuccess.style.display = 'block';
            liveSourceInfo.textContent = 'Webcam (Device 0)';
            startBtn.disabled = false;
            
            // Start preview
            socket.emit('start_preview', {video_source: 'webcam'});
        } else if (source === 'rtsp') {
            // RTSP Stream
            const url = rtspUrl.value.trim();
            if (!url) {
                alert('Please enter RTSP stream URL');
                return;
            }
            
            if (!url.startsWith('rtsp://')) {
                alert('Invalid RTSP URL. Must start with rtsp://');
                return;
            }
            
            liveSource = url;
            config.video_source = url;
            
            document.querySelector('.live-options').style.display = 'none';
            liveSuccess.style.display = 'block';
            liveSourceInfo.textContent = url;
            startBtn.disabled = false;
            
            // Start preview
            socket.emit('start_preview', {video_source: url});
        }
    });
});

// Start live detection
startLiveDetection.addEventListener('click', () => {
    liveModal.style.display = 'none';
    startBtn.disabled = false;
    videoStatus.textContent = 'Preview mode. Click Start Detection to begin analysis.';
});

// Confidence threshold slider
document.getElementById('confThreshold').addEventListener('input', (e) => {
    document.getElementById('confValue').textContent = e.target.value;
});

// Save configuration
saveConfigBtn.addEventListener('click', async () => {
    config.conf_threshold = parseFloat(document.getElementById('confThreshold').value);
    
    config.alert_severities = Array.from(document.querySelectorAll('.checkbox-group input:checked'))
        .map(cb => cb.value);
    
    config.enable_email = document.getElementById('enableEmail').checked;
    config.enable_sms = document.getElementById('enableSMS').checked;
    
    configModal.style.display = 'none';
    alert('Configuration saved!');
});

// Start detection
startBtn.addEventListener('click', () => {
    if (config.video_source === null || config.video_source === undefined) {
        alert('Please select a video source first (Upload Video or Live Detection)');
        return;
    }
    console.log('Starting detection with config:', config);
    socket.emit('start_detection', config);
});

// Stop detection
stopBtn.addEventListener('click', () => {
    if (detectionRunning) {
        socket.emit('stop_detection');
    } else {
        socket.emit('stop_preview');
    }
});

// Load logs
async function loadLogs() {
    const response = await fetch('/get_logs');
    const logs = await response.json();
    
    const tbody = document.querySelector('#logsTable tbody');
    tbody.innerHTML = '';
    
    if (logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No logs found</td></tr>';
        return;
    }
    
    // Sort logs by frame_index descending (latest first)
    logs.sort((a, b) => b.frame_index - a.frame_index);
    
    logs.forEach(log => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = log.frame_index;
        row.insertCell(1).textContent = log.track_id;
        row.insertCell(2).textContent = log.accident_type;
        row.insertCell(3).textContent = log.severity;
        row.insertCell(4).textContent = log.bbox;
    });
}

document.getElementById('refreshLogs').addEventListener('click', loadLogs);

// Load statistics
async function loadStatistics() {
    const response = await fetch('/get_statistics');
    const data = await response.json();
    
    // Accident Type Chart
    const accidentCtx = document.getElementById('accidentTypeChart').getContext('2d');
    if (accidentTypeChart) accidentTypeChart.destroy();
    
    accidentTypeChart = new Chart(accidentCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(data.accident_types),
            datasets: [{
                label: 'Accident Types',
                data: Object.values(data.accident_types),
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Accident Types Distribution'
                }
            }
        }
    });
    
    // Severity Chart
    const severityCtx = document.getElementById('severityChart').getContext('2d');
    if (severityChart) severityChart.destroy();
    
    severityChart = new Chart(severityCtx, {
        type: 'pie',
        data: {
            labels: Object.keys(data.severities),
            datasets: [{
                data: Object.values(data.severities),
                backgroundColor: [
                    'rgba(239, 68, 68, 0.6)',
                    'rgba(251, 146, 60, 0.6)',
                    'rgba(250, 204, 21, 0.6)',
                    'rgba(34, 197, 94, 0.6)'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Severity Distribution'
                }
            }
        }
    });
}

// Load clips
async function loadClips() {
    const response = await fetch('/get_clips');
    const clips = await response.json();
    
    const clipsList = document.getElementById('clipsList');
    clipsList.innerHTML = '';
    
    if (clips.length === 0) {
        clipsList.innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">No clips available yet.</p>';
        return;
    }
    
    // Sort clips by name descending (latest first)
    clips.sort((a, b) => b.localeCompare(a));
    
    clips.forEach(clip => {
        const clipItem = document.createElement('div');
        clipItem.className = 'clip-item';
        clipItem.innerHTML = `
            <video controls>
                <source src="/clips/${currentUserId}/${clip}" type="video/mp4">
            </video>
            <p>${clip}</p>
        `;
        clipsList.appendChild(clipItem);
    });
}

document.getElementById('refreshClips').addEventListener('click', loadClips);

// Load user stats on page load
async function loadUserStats() {
    try {
        const response = await fetch('/user/stats');
        const stats = await response.json();
        
        document.getElementById('totalDetections').textContent = stats.total_detections;
        document.getElementById('criticalAlerts').textContent = stats.critical_alerts;
    } catch (error) {
        console.error('Error loading user stats:', error);
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    loadUserStats();
    startAutoRefresh();
    setupAutoRefreshToggle();
});

// Auto-refresh functionality
let autoRefreshInterval = null;
let autoRefreshEnabled = true;

function startAutoRefresh() {
    stopAutoRefresh();
    if (!autoRefreshEnabled) return;
    
    autoRefreshInterval = setInterval(() => {
        const activeSection = document.querySelector('.user-section.active');
        if (!activeSection) return;
        
        const sectionId = activeSection.id;
        
        if (sectionId === 'logs') {
            loadLogs();
        } else if (sectionId === 'sessions') {
            loadSessions();
        } else if (sectionId === 'clips') {
            loadClips();
        } else if (sectionId === 'uploads') {
            loadUploads();
        } else if (sectionId === 'detection') {
            loadUserStats();
        }
    }, 5000);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

// Setup toggle buttons
function setupAutoRefreshToggle() {
    const toggleBtn1 = document.getElementById('toggleAutoRefresh');
    const toggleBtn2 = document.getElementById('toggleAutoRefreshClips');
    
    function updateButtons() {
        [toggleBtn1, toggleBtn2].forEach(btn => {
            if (btn) {
                if (autoRefreshEnabled) {
                    btn.textContent = '✅ Auto-Refresh: ON';
                    btn.className = 'btn btn-success';
                } else {
                    btn.textContent = '❌ Auto-Refresh: OFF';
                    btn.className = 'btn btn-secondary';
                }
            }
        });
    }
    
    function toggleAutoRefresh() {
        autoRefreshEnabled = !autoRefreshEnabled;
        console.log('Toggle clicked, new state:', autoRefreshEnabled);
        updateButtons();
        
        if (autoRefreshEnabled) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    }
    
    if (toggleBtn1) {
        toggleBtn1.addEventListener('click', toggleAutoRefresh);
    }
    if (toggleBtn2) {
        toggleBtn2.addEventListener('click', toggleAutoRefresh);
    }
}

// Stop auto-refresh when page is hidden
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopAutoRefresh();
    } else if (autoRefreshEnabled) {
        startAutoRefresh();
    }
});

// Load sessions
async function loadSessions() {
    const response = await fetch('/user/sessions');
    const sessions = await response.json();
    
    const tbody = document.querySelector('#sessionsTable tbody');
    tbody.innerHTML = '';
    
    if (sessions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">No sessions found</td></tr>';
        return;
    }
    
    // Sessions are already sorted by start_time descending from backend
    sessions.forEach(session => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = session.id;
        row.insertCell(1).textContent = session.video_source;
        row.insertCell(2).textContent = session.start_time;
        row.insertCell(3).textContent = session.end_time;
        row.insertCell(4).textContent = session.total_detections;
        row.insertCell(5).textContent = session.critical_alerts;
        
        const statusCell = row.insertCell(6);
        const statusBadge = document.createElement('span');
        statusBadge.className = `status-badge ${session.status}`;
        statusBadge.textContent = session.status;
        statusCell.appendChild(statusBadge);
        
        const actionsCell = row.insertCell(7);
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn btn-secondary btn-sm';
        viewBtn.textContent = '👁️ View';
        viewBtn.onclick = () => viewSession(session.id);
        actionsCell.appendChild(viewBtn);
        
        if (session.status === 'active') {
            const stopBtn = document.createElement('button');
            stopBtn.className = 'btn btn-danger btn-sm';
            stopBtn.textContent = '⏹️ Stop';
            stopBtn.onclick = () => stopSession(session.id);
            actionsCell.appendChild(document.createTextNode(' '));
            actionsCell.appendChild(stopBtn);
        } else {
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn btn-danger btn-sm';
            deleteBtn.textContent = '🗑️ Delete';
            deleteBtn.onclick = () => deleteSession(session.id);
            actionsCell.appendChild(document.createTextNode(' '));
            actionsCell.appendChild(deleteBtn);
        }
    });
}

document.getElementById('refreshSessions').addEventListener('click', loadSessions);

async function stopSession(sessionId) {
    if (confirm('Are you sure you want to stop this session?')) {
        const response = await fetch(`/user/session/${sessionId}/stop`, {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            loadSessions();
            loadUserStats();
        } else {
            alert('Error: ' + result.message);
        }
    }
}

async function deleteSession(sessionId) {
    if (confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
        const response = await fetch(`/user/session/${sessionId}/delete`, {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            alert(result.message);
            loadSessions();
            loadUserStats();
        } else {
            alert('Error: ' + result.message);
        }
    }
}

async function viewSession(sessionId) {
    const response = await fetch(`/user/session/${sessionId}`);
    const session = await response.json();
    
    if (session.status === 'error') {
        alert('Error: ' + session.message);
        return;
    }
    
    const details = `
Session ID: ${session.id}
Video Source: ${session.video_source}
Start Time: ${session.start_time}
End Time: ${session.end_time}
Total Detections: ${session.total_detections}
Critical Alerts: ${session.critical_alerts}
Status: ${session.status}
    `;
    
    alert(details);
}

// Load user uploads
async function loadUploads() {
    const response = await fetch('/user/uploads');
    const uploads = await response.json();
    
    const uploadsList = document.getElementById('uploadsList');
    uploadsList.innerHTML = '';
    
    if (uploads.length === 0) {
        uploadsList.innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">No uploads found.</p>';
        return;
    }
    
    uploads.sort((a, b) => b.uploaded.localeCompare(a.uploaded));
    
    uploads.forEach(upload => {
        const sizeInMB = (upload.size / (1024 * 1024)).toFixed(2);
        
        const uploadItem = document.createElement('div');
        uploadItem.className = 'clip-item';
        uploadItem.innerHTML = `
            <video controls>
                <source src="/uploads/${currentUserId}/${upload.filename}" type="video/mp4">
            </video>
            <p><strong>Size:</strong> ${sizeInMB} MB</p>
            <p><strong>Uploaded:</strong> ${upload.uploaded}</p>
            <p>${upload.filename}</p>
            <button class="btn btn-success" onclick="runDetectionOnUpload('${upload.filename}')">▶️ Run Detection</button>
        `;
        uploadsList.appendChild(uploadItem);
    });
}

// Run detection on uploaded file
function runDetectionOnUpload(filename) {
    const filepath = `static/uploads/${currentUserId}/${filename}`;
    config.video_source = filepath;
    uploadedVideoPath = filepath;
    
    // Switch to detection section
    document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
    document.querySelector('[data-section="detection"]').classList.add('active');
    document.querySelectorAll('.user-section').forEach(sec => sec.classList.remove('active'));
    document.getElementById('detection').classList.add('active');
    
    // Start detection
    socket.emit('start_detection', config);
}

const refreshUploadsBtn = document.getElementById('refreshUploads');
if (refreshUploadsBtn) {
    refreshUploadsBtn.addEventListener('click', loadUploads);
}
