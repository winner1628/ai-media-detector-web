// Global variables
let model;
const IMAGE_SIZE = 224; // Model expects 224x224 images

// Initialize app (load model on page load)
async function init() {
    try {
        updateStatus("Loading AI model... (first run: ~10s)");
        updateProgress(20);

        // Load pre-trained AI image detector (TensorFlow.js model)
        // This model is a lightweight version of facebook/ai-image-detector (converted to TF.js)
        model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/ai-image-detector/model.json');
        
        updateStatus("Model loaded! Ready to upload images");
        updateProgress(100);
        enableDetectButton();
    } catch (error) {
        updateStatus(`Error loading model: ${error.message}`);
        console.error(error);
    }
}

// File upload handler
document.getElementById('image-upload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
        updateStatus("Error: Only JPG/PNG images are supported");
        return;
    }

    // Preview image
    const preview = document.getElementById('image-preview');
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;

    // Enable detect button
    enableDetectButton();
    updateStatus("Image uploaded! Click 'Run AI Detection'");
});

// Detect button handler
document.getElementById('detect-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];
    if (!file || !model) return;

    try {
        updateStatus("Processing image...");
        updateProgress(0);

        // Step 1: Load and preprocess image
        const image = await loadAndPreprocessImage(file);
        updateProgress(30);

        // Step 2: Run AI detection
        updateStatus("Analyzing image for AI generation...");
        const prediction = await predictImage(image);
        updateProgress(80);

        // Step 3: Display results
        displayResults(prediction);
        updateProgress(100);
        updateStatus("Detection complete!");

    } catch (error) {
        updateStatus(`Detection error: ${error.message}`);
        console.error(error);
    }
});

// Load and preprocess image for model
async function loadAndPreprocessImage(file) {
    // Convert file to HTML image element
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await new Promise((resolve) => img.onload = resolve);

    // Resize and preprocess image (match model requirements)
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]) // Resize to 224x224
            .toFloat()
            .div(tf.scalar(255.0)) // Normalize to 0-1
            .expandDims(); // Add batch dimension (model expects [1, 224, 224, 3])
        return tensor;
    });
}

// Run prediction on preprocessed image
async function predictImage(tensor) {
    // Run model prediction
    const predictions = await model.predict(tensor).data();
    
    // Model outputs: [prob_ai, prob_real] (0-1)
    const probAI = predictions[0];
    const probReal = predictions[1];
    const confidence = Math.max(probAI, probReal) * 100;

    // Determine result
    let result = probAI > probReal ? "AI-generated" : "Real";
    
    return {
        result: result,
        confidence: parseFloat(confidence.toFixed(2))
    };
}

// Display results to user
function displayResults(prediction) {
    const resultsSection = document.querySelector('.results-section');
    const resultText = document.getElementById('result-text');
    const confidenceText = document.getElementById('confidence-text');

    // Show results section
    resultsSection.hidden = false;

    // Set result text (color-coded)
    resultText.textContent = prediction.result;
    resultText.className = prediction.result === "AI-generated" ? "ai-generated" : "real";

    // Set confidence text
    confidenceText.textContent = `Confidence: ${prediction.confidence}%`;
}

// Helper: Update progress bar
function updateProgress(percent) {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.setProperty('--progress', `${percent}%`);
    progressBar.innerHTML = `<div style="width: ${percent}%; height: 100%; background-color: #0078D7;"></div>`;
}

// Helper: Update status text
function updateStatus(text) {
    document.getElementById('status').textContent = text;
}

// Helper: Enable detect button
function enableDetectButton() {
    document.getElementById('detect-btn').disabled = false;
}

// Initialize app when page loads
window.onload = init;
