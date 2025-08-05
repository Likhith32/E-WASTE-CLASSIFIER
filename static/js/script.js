// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name-display');

    if (uploadArea) {
        // Trigger file input click when the area is clicked
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Add visual feedback for drag-and-drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                updateFileName(files[0]);
            }
        });
    }

    if (fileInput) {
        // Update file name when selected via browse
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                updateFileName(fileInput.files[0]);
            }
        });
    }

    function updateFileName(file) {
        if (file) {
            fileNameDisplay.textContent = `Selected: ${file.name}`;
        }
    }
});