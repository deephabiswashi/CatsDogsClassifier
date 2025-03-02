// Show image preview on file selection
document.getElementById('image').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('preview');
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
});

// Handle form submission via AJAX
document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        if(data.error) {
            resultDiv.innerText = 'Error: ' + data.error;
        } else {
            resultDiv.innerText = `Prediction: ${data.result} (Confidence: ${data.confidence.toFixed(2)})`;
        }
    })
    .catch(err => {
        document.getElementById('result').innerText = 'An error occurred.';
        console.error(err);
    });
});
