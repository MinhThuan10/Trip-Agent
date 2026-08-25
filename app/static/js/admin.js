const API_BASE = "/api/v1/admin";

function switchTab(tabId, element) {
    document.querySelectorAll('.tab-content-section').forEach(el => el.style.display = 'none');
    document.getElementById(tabId).style.display = 'block';
    document.querySelectorAll('.sidebar .nav-link').forEach(el => el.classList.remove('active'));
    if(element) element.classList.add('active');
}

async function loadAirports() {
    try {
        const res = await fetch(`${API_BASE}/airports?limit=100`);
        const json = await res.json();
        if (json.success) {
            const tbody = document.querySelector('#airportsTable tbody');
            tbody.innerHTML = json.data.map(a => `
                <tr>
                    <td><strong>${a.iata_code}</strong></td>
                    <td>${a.airport_name}</td>
                    <td>${a.city || ''}</td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadAirlines() {
    try {
        const res = await fetch(`${API_BASE}/airlines?limit=100`);
        const json = await res.json();
        if (json.success) {
            const tbody = document.querySelector('#airlinesTable tbody');
            tbody.innerHTML = json.data.map(a => `
                <tr>
                    <td><strong>${a.iata_code}</strong></td>
                    <td>${a.icao_code || ''}</td>
                    <td>${a.airline_name}</td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadRAGDocs() {
    try {
        const res = await fetch(`${API_BASE}/rag/documents`);
        const json = await res.json();
        if (json.success) {
            const tbody = document.querySelector('#ragTable tbody');
            if (json.documents.length === 0) {
                tbody.innerHTML = `<tr><td colspan="4" class="text-center text-muted">Chưa có tài liệu nào được upload.</td></tr>`;
            } else {
                tbody.innerHTML = json.documents.map(d => `
                    <tr>
                        <td><span class="badge bg-info text-dark">${d.category}</span></td>
                        <td><i class="fas fa-file-pdf text-danger me-2"></i>${d.file_name}</td>
                        <td><strong>${d.chunk_count}</strong> chunks</td>
                        <td class="text-end">
                            <a href="${API_BASE}/rag/download?file_name=${encodeURIComponent(d.file_name)}" class="btn btn-sm btn-outline-primary me-1" target="_blank">
                                <i class="fas fa-download me-1"></i> Tải về
                            </a>
                            <button class="btn btn-sm btn-outline-danger" onclick="deleteRAGDoc('${d.file_name}')">
                                <i class="fas fa-trash-alt me-1"></i> Xóa
                            </button>
                        </td>
                    </tr>
                `).join('');
            }

            // Populate category select
            const select = document.getElementById('categorySelect');
            let options = json.categories.map(c => `<option value="${c}">${c}</option>`).join('');
            options += `<option value="__new__">+ Thêm danh mục mới...</option>`;
            select.innerHTML = options;
        }
    } catch (e) {
        console.error(e);
    }
}

async function deleteRAGDoc(fileName) {
    if (!confirm(`Bạn có chắc chắn muốn xóa tài liệu "${fileName}"? Thao tác này sẽ xóa file và toàn bộ vector chunks khỏi cơ sở dữ liệu.`)) {
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/rag/documents?file_name=${encodeURIComponent(fileName)}`, {
            method: 'DELETE'
        });
        const json = await res.json();
        if (res.ok && json.success) {
            alert(json.message);
            loadRAGDocs();
        } else {
            alert('Lỗi: ' + (json.detail || 'Không thể xóa tài liệu.'));
        }
    } catch (err) {
        console.error(err);
        alert('Lỗi kết nối tới server.');
    }
}

function checkCategoryInput(select) {
    const input = document.getElementById('newCategoryInput');
    if (select.value === '__new__') {
        input.style.display = 'block';
        input.required = true;
        input.focus();
    } else {
        input.style.display = 'none';
        input.required = false;
    }
}

document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('pdfFile');
    const select = document.getElementById('categorySelect');
    const newInput = document.getElementById('newCategoryInput');
    const spinner = document.getElementById('uploadSpinner');
    const submitBtn = document.getElementById('uploadSubmitBtn');

    let category = select.value;
    if (category === '__new__') {
        category = newInput.value.trim();
        if (!category) {
            alert('Vui lòng nhập tên danh mục mới.');
            return;
        }
    }

    if (fileInput.files.length === 0) {
        alert('Vui lòng chọn file PDF.');
        return;
    }

    const uploadedFile = fileInput.files[0];
    if (!uploadedFile.name.toLowerCase().endsWith('.pdf')) {
        alert('Chỉ cho phép tải lên file định dạng PDF (.pdf)!');
        return;
    }

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('category', category);

    spinner.style.display = 'inline-block';
    submitBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/rag/upload`, {
            method: 'POST',
            body: formData
        });
        const json = await res.json();
        if (res.ok && json.success) {
            alert(json.message);
            const modal = bootstrap.Modal.getInstance(document.getElementById('uploadModal'));
            modal.hide();
            document.getElementById('uploadForm').reset();
            document.getElementById('newCategoryInput').style.display = 'none';
            loadRAGDocs();
        } else {
            alert('Lỗi: ' + (json.detail || 'Không thể upload.'));
        }
    } catch (err) {
        console.error(err);
        alert('Lỗi kết nối tới server.');
    } finally {
        spinner.style.display = 'none';
        submitBtn.disabled = false;
    }
});

// Initial load
window.onload = () => {
    loadAirports();
    loadAirlines();
    loadRAGDocs();
};
