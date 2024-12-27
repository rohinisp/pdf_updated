// Show status message
function showStatus(message) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.classList.remove('d-none');
}

// Show error message
function showError(message) {
    const errorElement = document.getElementById('error');
    errorElement.textContent = message;
    errorElement.classList.remove('d-none');
    document.getElementById('status').classList.add('d-none');
}

// Clear messages
function clearMessages() {
    document.getElementById('status').classList.add('d-none');
    document.getElementById('error').classList.add('d-none');
}

// Extract text from URL
async function extractFromUrl() {
    clearMessages();
    const url = document.getElementById('documentUrl').value;
    const question = document.getElementById('question').value;

    if (!url) {
        showError('Please enter a URL');
        return;
    }

    showStatus('Processing document...');

    try {
        const response = await fetch('/api/process-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                url: url,
                question: question || undefined
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to extract content');
        }

        updateExtractResponse(data);
        if (data.answer) {
            updateAnswerSection(data);
        }
    } catch (error) {
        showError(error.message);
    }
}

// Extract text from file
async function extractFromFile() {
    clearMessages();
    const fileInput = document.getElementById('pdfFile');
    const question = document.getElementById('question').value;

    if (!fileInput.files.length) {
        showError('Please select a PDF file');
        return;
    }

    const file = fileInput.files[0];
    if (file.type !== 'application/pdf') {
        showError('Please select a valid PDF file');
        return;
    }

    showStatus('Processing PDF...');

    const formData = new FormData();
    formData.append('file', file);
    if (question) {
        formData.append('question', question);
    }

    try {
        const response = await fetch('/api/process-pdf', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to extract content');
        }

        updateExtractResponse(data);
        if (data.answer) {
            updateAnswerSection(data);
        }
    } catch (error) {
        showError(error.message);
    }
}

// Copy extracted text to clipboard
function copyText() {
    const textarea = document.getElementById('extractedText');
    textarea.select();
    document.execCommand('copy');
    showStatus('Text copied to clipboard!');
}

// Update the extract response
function updateExtractResponse(data) {
    const textArea = document.getElementById('extractedText');
    textArea.value = data.text;

    // Calculate metrics
    const textLength = data.text.length;
    const textKB = (new TextEncoder().encode(data.text).length / 1024).toFixed(2);

    // Create performance metrics display
    const metrics = [];
    metrics.push(`${textKB}KB extracted`);

    if (data.metadata?.processing_time) {
        metrics.push(`${data.metadata.processing_time.toFixed(1)}s processing time`);
    }

    showStatus(`Content processed successfully! ${metrics.join(' • ')}`);
}

// Update answer section
function updateAnswerSection(data) {
    const answerSection = document.getElementById('answerSection');
    const answerContent = document.getElementById('answerContent').querySelector('.card-body');

    if (data.answer) {
        answerSection.classList.remove('d-none');
        answerContent.innerHTML = `
            <p class="mb-2">${data.answer.answer}</p>
            <p class="text-muted small mb-1">Confidence: ${(data.answer.confidence * 100).toFixed(1)}%</p>
            <div class="mt-2">
                <small class="text-muted">Context:</small>
                <p class="small">${data.answer.context}</p>
            </div>
            ${data.answer.reasoning ? `
            <div class="mt-2">
                <small class="text-muted">Reasoning:</small>
                <p class="small">${data.answer.reasoning}</p>
            </div>` : ''}
        `;
    } else {
        answerSection.classList.add('d-none');
        answerContent.innerHTML = '';
    }
}