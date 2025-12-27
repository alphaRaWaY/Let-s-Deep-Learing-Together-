const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const uploadBtn = document.getElementById('uploadBtn');

imageInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            document.getElementById('prompt').style.display = 'none';
            uploadBtn.disabled = false;
        }
        reader.readAsDataURL(file);
    }
});

uploadBtn.addEventListener('click', async () => {
    const formData = new FormData();
    formData.append('file', imageInput.files[0]);

    uploadBtn.innerText = '检测中...';
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const result = await response.json();
    
    document.getElementById('resultCard').style.display = 'block';
    document.getElementById('prediction').innerText = result.prediction;
    uploadBtn.innerText = '开始智能检测';
});