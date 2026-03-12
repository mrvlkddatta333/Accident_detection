// Sidebar navigation
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const section = item.dataset.section;
        
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');
        
        // Update active section
        document.querySelectorAll('.admin-section').forEach(sec => sec.classList.remove('active'));
        document.getElementById(section).classList.add('active');
        
        // Load section data
        if (section === 'users') loadUsers();
        if (section === 'sessions') loadSessions();
        if (section === 'logs') loadLogsAdmin();
        if (section === 'clips') loadClipsAdmin();
        if (section === 'uploads') loadUploadsAdmin();
        if (section === 'analytics') loadAnalytics();
        if (section === 'overview') loadOverview();
    });
});

// Load overview data
async function loadOverview() {
    const usersResponse = await fetch('/admin/users');
    const users = await usersResponse.json();
    
    const sessionsResponse = await fetch('/admin/sessions');
    const sessions = await sessionsResponse.json();
    
    document.getElementById('totalUsersOverview').textContent = users.length;
    
    const activeUsers = users.filter(u => u.is_active).length;
    document.getElementById('activeUsersOverview').textContent = activeUsers;
    
    const activeSessions = sessions.filter(s => s.status === 'active').length;
    document.getElementById('activeSessionsOverview').textContent = activeSessions;
    
    const totalDetections = sessions.reduce((sum, s) => sum + s.total_detections, 0);
    document.getElementById('totalDetectionsOverview').textContent = totalDetections;
    
    const criticalAlerts = sessions.reduce((sum, s) => sum + s.critical_alerts, 0);
    document.getElementById('criticalAlertsOverview').textContent = criticalAlerts;
    
    // Load recent activity
    loadRecentActivity();
    
    // Initialize charts
    initDetectionTrendChart();
    initUserActivityChart();
}

// Load recent activity
async function loadRecentActivity() {
    try {
        const response = await fetch('/admin/activity');
        const activities = await response.json();
        
        const activityList = document.getElementById('recentActivity');
        
        if (activities.length === 0) {
            activityList.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #999;">
                    <p>No recent activity</p>
                </div>
            `;
            return;
        }
        
        activityList.innerHTML = activities.map(activity => `
            <div class="activity-item">
                <div class="activity-icon ${activity.type}">${activity.icon}</div>
                <div class="activity-content">
                    <p>${activity.message}</p>
                    <span class="activity-time">${activity.time_ago}</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading activity:', error);
        document.getElementById('recentActivity').innerHTML = `
            <div style="text-align: center; padding: 40px; color: #ef4444;">
                <p>Error loading activity</p>
            </div>
        `;
    }
}

// Detection trend chart
function initDetectionTrendChart() {
    const ctx = document.getElementById('detectionTrendChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Detections',
                data: [12, 19, 15, 25, 22, 30, 28],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// User activity chart
function initUserActivityChart() {
    const ctx = document.getElementById('userActivityChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Active', 'Inactive', 'New'],
            datasets: [{
                data: [65, 25, 10],
                backgroundColor: ['#10b981', '#ef4444', '#667eea']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Load users
async function loadUsers() {
    const response = await fetch('/admin/users');
    const users = await response.json();
    
    const tbody = document.querySelector('#usersTable tbody');
    tbody.innerHTML = '';
    
    users.forEach(user => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${user.id}</td>
            <td>${user.username}</td>
            <td>${user.email}</td>
            <td><span class="role-badge ${user.role}">${user.role}</span></td>
            <td><span class="status-badge ${user.is_active ? 'active' : 'inactive'}">${user.is_active ? 'Active' : 'Inactive'}</span></td>
            <td>${user.created_at}</td>
            <td>${user.last_login}</td>
            <td>
                ${user.role !== 'admin' ? `
                    <button class="action-btn toggle" onclick="toggleUser(${user.id})">
                        ${user.is_active ? 'Deactivate' : 'Activate'}
                    </button>
                    <button class="action-btn delete" onclick="deleteUser(${user.id})">Delete</button>
                ` : '<em>Admin</em>'}
            </td>
        `;
    });
}

// Toggle user status
async function toggleUser(userId) {
    if (!confirm('Are you sure you want to toggle this user\'s status?')) return;
    
    const response = await fetch(`/admin/user/${userId}/toggle`, {method: 'POST'});
    const result = await response.json();
    
    if (result.status === 'success') {
        alert(result.message);
        loadUsers();
    } else {
        alert(result.message);
    }
}

// Delete user
async function deleteUser(userId) {
    if (!confirm('Are you sure you want to delete this user? This action cannot be undone.')) return;
    
    const response = await fetch(`/admin/user/${userId}/delete`, {method: 'POST'});
    const result = await response.json();
    
    if (result.status === 'success') {
        alert(result.message);
        loadUsers();
    } else {
        alert(result.message);
    }
}

// Load sessions
async function loadSessions() {
    const response = await fetch('/admin/sessions');
    const sessions = await response.json();
    
    const tbody = document.querySelector('#sessionsTable tbody');
    tbody.innerHTML = '';
    
    if (sessions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center;">No sessions found</td></tr>';
        return;
    }
    
    sessions.forEach(session => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${session.id}</td>
            <td>${session.username}</td>
            <td>${session.video_source}</td>
            <td>${session.start_time}</td>
            <td>${session.end_time}</td>
            <td>${session.total_detections}</td>
            <td>${session.critical_alerts}</td>
            <td><span class="status-badge ${session.status}">${session.status}</span></td>
            <td>
                <button class="action-btn toggle" onclick="viewSessionAdmin(${session.id})">👁️ View</button>
                ${session.status === 'active' ? `
                    <button class="action-btn delete" onclick="stopSessionAdmin(${session.id})">⏹️ Stop</button>
                ` : `
                    <button class="action-btn delete" onclick="deleteSessionAdmin(${session.id})">🗑️ Delete</button>
                `}
            </td>
        `;
    });
}

const refreshSessionsBtn = document.getElementById('refreshSessions');
if (refreshSessionsBtn) {
    refreshSessionsBtn.addEventListener('click', loadSessions);
}

const refreshSessionsAdminBtn = document.getElementById('refreshSessionsAdmin');
if (refreshSessionsAdminBtn) {
    refreshSessionsAdminBtn.addEventListener('click', loadSessions);
}

// Stop individual session
async function stopSessionAdmin(sessionId) {
    if (!confirm('Are you sure you want to stop this session?')) return;
    
    const response = await fetch(`/admin/session/${sessionId}/stop`, {method: 'POST'});
    const result = await response.json();
    
    if (result.status === 'success') {
        alert(result.message);
        loadSessions();
        loadOverview();
    } else {
        alert('Error: ' + result.message);
    }
}

// Delete individual session
async function deleteSessionAdmin(sessionId) {
    if (!confirm('Are you sure you want to delete this session? This action cannot be undone.')) return;
    
    const response = await fetch(`/admin/session/${sessionId}/delete`, {method: 'POST'});
    const result = await response.json();
    
    if (result.status === 'success') {
        alert(result.message);
        loadSessions();
        loadOverview();
    } else {
        alert('Error: ' + result.message);
    }
}

// View session details
async function viewSessionAdmin(sessionId) {
    const response = await fetch(`/admin/session/${sessionId}`);
    const session = await response.json();
    
    if (session.status === 'error') {
        alert('Error: ' + session.message);
        return;
    }
    
    const details = `
Session ID: ${session.id}
User: ${session.username}
Video Source: ${session.video_source}
Start Time: ${session.start_time}
End Time: ${session.end_time}
Total Detections: ${session.total_detections}
Critical Alerts: ${session.critical_alerts}
Status: ${session.status}
    `;
    
    alert(details);
}

// Stop all active sessions
const stopAllSessionsBtn = document.getElementById('stopAllSessions');
if (stopAllSessionsBtn) {
    stopAllSessionsBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to stop ALL active sessions? This will affect all users.')) return;
        
        const response = await fetch('/admin/sessions/stop-all', {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            alert(result.message);
            loadSessions();
            loadOverview();
        } else {
            alert('Error: ' + result.message);
        }
    });
}

// Delete all sessions
const deleteAllSessionsBtn = document.getElementById('deleteAllSessions');
if (deleteAllSessionsBtn) {
    deleteAllSessionsBtn.addEventListener('click', async () => {
        if (!confirm('⚠️ This will STOP all active sessions and DELETE ALL sessions. This action cannot be undone. Continue?')) return;
        
        const response = await fetch('/admin/sessions/delete-all', {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            alert(result.message);
            loadSessions();
            loadOverview();
        } else {
            alert('Error: ' + result.message);
        }
    });
}

// Load analytics
async function loadAnalytics() {
    const response = await fetch('/get_statistics');
    const data = await response.json();
    
    // Accident types chart
    const accidentCtx = document.getElementById('accidentTypesChart').getContext('2d');
    new Chart(accidentCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(data.accident_types),
            datasets: [{
                label: 'Occurrences',
                data: Object.values(data.accident_types),
                backgroundColor: '#667eea'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Severity chart
    const severityCtx = document.getElementById('severityChart').getContext('2d');
    new Chart(severityCtx, {
        type: 'pie',
        data: {
            labels: Object.keys(data.severities),
            datasets: [{
                data: Object.values(data.severities),
                backgroundColor: ['#ef4444', '#f59e0b', '#10b981', '#667eea']
            }]
        },
        options: {
            responsive: true
        }
    });
    
    // Monthly trends
    const trendsCtx = document.getElementById('monthlyTrendsChart').getContext('2d');
    new Chart(trendsCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Detections',
                data: [45, 52, 48, 65, 70, 85],
                borderColor: '#667eea',
                tension: 0.4
            }]
        },
        options: {
            responsive: true
        }
    });
}

// Load initial data
loadOverview();

// Auto-refresh for admin
let autoRefreshIntervalAdmin = null;
let autoRefreshEnabledAdmin = true;

function startAutoRefreshAdmin() {
    stopAutoRefreshAdmin();
    if (!autoRefreshEnabledAdmin) return;
    
    autoRefreshIntervalAdmin = setInterval(() => {
        const activeSection = document.querySelector('.admin-section.active');
        if (!activeSection) return;
        
        const sectionId = activeSection.id;
        
        if (sectionId === 'sessions') {
            loadSessions();
        } else if (sectionId === 'logs') {
            loadLogsAdmin();
        } else if (sectionId === 'clips') {
            loadClipsAdmin();
        } else if (sectionId === 'uploads') {
            loadUploadsAdmin();
        } else if (sectionId === 'overview') {
            loadOverview();
        } else if (sectionId === 'users') {
            loadUsers();
        }
    }, 5000);
}

function stopAutoRefreshAdmin() {
    if (autoRefreshIntervalAdmin) {
        clearInterval(autoRefreshIntervalAdmin);
        autoRefreshIntervalAdmin = null;
    }
}

// Setup admin toggle buttons
function setupAdminAutoRefreshToggle() {
    const toggleBtns = [
        document.getElementById('toggleAutoRefreshAdmin'),
        document.getElementById('toggleAutoRefreshAdminLogs'),
        document.getElementById('toggleAutoRefreshAdminClips')
    ];
    
    function updateAdminButtons() {
        toggleBtns.forEach(btn => {
            if (btn) {
                if (autoRefreshEnabledAdmin) {
                    btn.textContent = '✅ Auto-Refresh: ON';
                    btn.className = 'btn btn-success';
                } else {
                    btn.textContent = '❌ Auto-Refresh: OFF';
                    btn.className = 'btn btn-secondary';
                }
            }
        });
    }
    
    function toggleAdmin() {
        autoRefreshEnabledAdmin = !autoRefreshEnabledAdmin;
        console.log('Admin toggle clicked, new state:', autoRefreshEnabledAdmin);
        updateAdminButtons();
        
        if (autoRefreshEnabledAdmin) {
            startAutoRefreshAdmin();
        } else {
            stopAutoRefreshAdmin();
        }
    }
    
    toggleBtns.forEach(btn => {
        if (btn) {
            btn.addEventListener('click', toggleAdmin);
        }
    });
}

// Initialize admin auto-refresh
setupAdminAutoRefreshToggle();
startAutoRefreshAdmin();

// Stop auto-refresh when page is hidden
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopAutoRefreshAdmin();
    } else if (autoRefreshEnabledAdmin) {
        startAutoRefreshAdmin();
    }
});

// Load logs for admin
async function loadLogsAdmin() {
    const response = await fetch('/admin/logs');
    const logs = await response.json();
    
    const tbody = document.querySelector('#logsTableAdmin tbody');
    tbody.innerHTML = '';
    
    if (logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No logs found</td></tr>';
        return;
    }
    
    logs.forEach(log => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = log.frame_index;
        row.insertCell(1).textContent = log.track_id;
        row.insertCell(2).textContent = log.accident_type;
        
        const severityCell = row.insertCell(3);
        const severityBadge = document.createElement('span');
        severityBadge.className = `status-badge ${log.severity.toLowerCase()}`;
        severityBadge.textContent = log.severity;
        severityCell.appendChild(severityBadge);
        
        row.insertCell(4).textContent = log.bbox;
    });
}

const refreshLogsAdminBtn = document.getElementById('refreshLogsAdmin');
if (refreshLogsAdminBtn) {
    refreshLogsAdminBtn.addEventListener('click', loadLogsAdmin);
}

const clearLogsAdminBtn = document.getElementById('clearLogsAdmin');
if (clearLogsAdminBtn) {
    clearLogsAdminBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear all logs? This action cannot be undone.')) {
            const response = await fetch('/clear_logs', {method: 'POST'});
            const result = await response.json();
            
            if (result.status === 'success') {
                alert(result.message);
                loadLogsAdmin();
            } else {
                alert('Error: ' + result.message);
            }
        }
    });
}

// Load clips for admin
let allClips = [];

async function loadClipsAdmin() {
    const response = await fetch('/admin/clips');
    allClips = await response.json();
    
    // Populate user filter
    populateClipUserFilter();
    
    // Display clips
    displayClips(allClips);
}

function populateClipUserFilter() {
    const userFilter = document.getElementById('clipUserFilter');
    if (!userFilter) return;
    
    // Get unique users from clips
    const users = [...new Set(allClips.map(c => c.username))];
    
    // Clear existing options except "All Users"
    userFilter.innerHTML = '<option value="">All Users</option>';
    
    // Add user options
    users.forEach(username => {
        const option = document.createElement('option');
        option.value = username;
        option.textContent = username;
        userFilter.appendChild(option);
    });
}

function displayClips(clips) {
    const clipsList = document.getElementById('clipsListAdmin');
    if (!clipsList) return;
    
    clipsList.innerHTML = '';
    
    if (clips.length === 0) {
        clipsList.innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">No clips available yet.</p>';
        return;
    }
    
    clips.forEach(clip => {
        const clipItem = document.createElement('div');
        clipItem.className = 'clip-item';
        clipItem.innerHTML = `
            <video controls>
                <source src="/clips/${clip.user_id}/${clip.filename}" type="video/mp4">
            </video>
            <p><strong>User:</strong> ${clip.username}</p>
            <p>${clip.filename}</p>
            <button class="btn btn-danger" onclick="deleteClipAdmin(${clip.user_id}, '${clip.filename}')">🗑️ Delete</button>
        `;
        clipsList.appendChild(clipItem);
    });
}

// Filter clips by user
const clipUserFilter = document.getElementById('clipUserFilter');
if (clipUserFilter) {
    clipUserFilter.addEventListener('change', (e) => {
        const selectedUser = e.target.value;
        
        if (selectedUser === '') {
            displayClips(allClips);
        } else {
            const filteredClips = allClips.filter(clip => clip.username === selectedUser);
            displayClips(filteredClips);
        }
    });
}

// Search clips
const clipSearch = document.getElementById('clipSearch');
if (clipSearch) {
    clipSearch.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const selectedUser = clipUserFilter ? clipUserFilter.value : '';
        
        let filteredClips = allClips;
        
        // Filter by user first
        if (selectedUser !== '') {
            filteredClips = filteredClips.filter(clip => clip.username === selectedUser);
        }
        
        // Then filter by search term
        if (searchTerm !== '') {
            filteredClips = filteredClips.filter(clip => clip.filename.toLowerCase().includes(searchTerm));
        }
        
        displayClips(filteredClips);
    });
}

const refreshClipsAdminBtn = document.getElementById('refreshClipsAdmin');
if (refreshClipsAdminBtn) {
    refreshClipsAdminBtn.addEventListener('click', loadClipsAdmin);
}

// Delete all clips
const deleteAllClipsBtn = document.getElementById('deleteAllClips');
if (deleteAllClipsBtn) {
    deleteAllClipsBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to delete ALL video clips? This action cannot be undone.')) return;
        
        const response = await fetch('/admin/clips/delete-all', {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            alert(result.message);
            loadClipsAdmin();
        } else {
            alert('Error: ' + result.message);
        }
    });
}

async function deleteClipAdmin(userId, filename) {
    if (confirm(`Delete ${filename}?`)) {
        const response = await fetch(`/admin/clip/${userId}/${filename}/delete`, {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            alert(result.message);
            loadClipsAdmin();
        } else {
            alert('Error: ' + result.message);
        }
    }
}

// Load uploads for admin
let allUploads = [];

async function loadUploadsAdmin() {
    const response = await fetch('/admin/uploads');
    allUploads = await response.json();
    
    populateUploadUserFilter();
    displayUploads(allUploads);
}

function populateUploadUserFilter() {
    const userFilter = document.getElementById('uploadUserFilter');
    if (!userFilter) return;
    
    const users = [...new Set(allUploads.map(u => u.user))];
    userFilter.innerHTML = '<option value="">All Users</option>';
    
    users.forEach(username => {
        const option = document.createElement('option');
        option.value = username;
        option.textContent = username;
        userFilter.appendChild(option);
    });
}

function displayUploads(uploads) {
    const uploadsList = document.getElementById('uploadsListAdmin');
    if (!uploadsList) return;
    
    uploadsList.innerHTML = '';
    
    if (uploads.length === 0) {
        uploadsList.innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">No uploads found.</p>';
        return;
    }
    
    uploads.forEach(upload => {
        const sizeInMB = (upload.size / (1024 * 1024)).toFixed(2);
        
        const uploadItem = document.createElement('div');
        uploadItem.className = 'clip-item';
        uploadItem.innerHTML = `
            <video controls>
                <source src="/uploads/${upload.user_id}/${upload.filename}" type="video/mp4">
            </video>
            <p><strong>User:</strong> ${upload.user}</p>
            <p><strong>Size:</strong> ${sizeInMB} MB</p>
            <p><strong>Uploaded:</strong> ${upload.uploaded}</p>
            <p>${upload.filename}</p>
            <button class="btn btn-danger" onclick="deleteUploadAdmin(${upload.user_id}, '${upload.filename}')">🗑️ Delete</button>
        `;
        uploadsList.appendChild(uploadItem);
    });
}

const uploadUserFilter = document.getElementById('uploadUserFilter');
if (uploadUserFilter) {
    uploadUserFilter.addEventListener('change', (e) => {
        const selectedUser = e.target.value;
        
        if (selectedUser === '') {
            displayUploads(allUploads);
        } else {
            const filteredUploads = allUploads.filter(u => u.user === selectedUser);
            displayUploads(filteredUploads);
        }
    });
}

const uploadSearch = document.getElementById('uploadSearch');
if (uploadSearch) {
    uploadSearch.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const selectedUser = uploadUserFilter ? uploadUserFilter.value : '';
        
        let filteredUploads = allUploads;
        
        if (selectedUser !== '') {
            filteredUploads = filteredUploads.filter(u => u.user === selectedUser);
        }
        
        if (searchTerm !== '') {
            filteredUploads = filteredUploads.filter(u => u.filename.toLowerCase().includes(searchTerm));
        }
        
        displayUploads(filteredUploads);
    });
}

const refreshUploadsAdminBtn = document.getElementById('refreshUploadsAdmin');
if (refreshUploadsAdminBtn) {
    refreshUploadsAdminBtn.addEventListener('click', loadUploadsAdmin);
}

const deleteAllUploadsBtn = document.getElementById('deleteAllUploads');
if (deleteAllUploadsBtn) {
    deleteAllUploadsBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to delete ALL uploaded videos? This action cannot be undone.')) return;
        
        const response = await fetch('/admin/uploads/delete-all', {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            alert(result.message);
            loadUploadsAdmin();
        } else {
            alert('Error: ' + result.message);
        }
    });
}

async function deleteUploadAdmin(userId, filename) {
    if (confirm(`Delete ${filename}?`)) {
        const response = await fetch(`/admin/upload/${userId}/${filename}/delete`, {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            alert(result.message);
            loadUploadsAdmin();
        } else {
            alert('Error: ' + result.message);
        }
    }
}
