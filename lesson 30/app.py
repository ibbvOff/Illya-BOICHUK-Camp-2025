import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image, UnidentifiedImageError
import io

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="Object recognition in photos",
    description="Upload a photo - find out what's in it",
)

try:
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

class Prediction(BaseModel):
    label: str
    score: float

class PredictionsResponse(BaseModel):
    results: list[Prediction]

@app.get("/", response_class=HTMLResponse, tags=["Home page"])
def main_page():
    return """
<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>Object recognizer</title>
  <style>
    body {
      font-family: "Segoe UI", Arial, sans-serif;
      background: linear-gradient(135deg, #667eea, #764ba2);
      min-height: 100vh;
      margin: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      color: #fff;
      padding: 40px 20px;
    }
    h1 {
      margin-bottom: 10px;
    }
    p {
      margin-bottom: 30px;
      opacity: 0.9;
    }
    .upload-area {
      position: relative;
      border: 3px dashed #fff;
      border-radius: 15px;
      padding: 50px 20px;
      text-align: center;
      background: rgba(255, 255, 255, 0.1);
      cursor: pointer;
      max-width: 400px;
      width: 100%;
      margin: 0 auto;
      transition: 0.3s;
    }
    .upload-area:hover {
      background: rgba(255, 255, 255, 0.2);
      transform: scale(1.02);
    }
    .upload-area input[type=file] {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
      cursor: pointer;
      z-index: 2;
    }
    .upload-content {
      position: relative;
      z-index: 1;
      pointer-events: none;
    }
    .upload-icon {
      font-size: 50px;
      margin-bottom: 15px;
    }
    .upload-text {
      font-weight: bold;
      font-size: 20px;
      margin-bottom: 5px;
    }
    .upload-hint {
      font-size: 14px;
      opacity: 0.8;
    }
    #preview {
      margin-top: 30px;
      max-width: 400px;
      display: none;
    }
    #preview img {
      width: 100%;
      border-radius: 15px;
      box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    #results {
      margin-top: 25px;
      font-size: 16px;
      max-width: 400px;
      width: 100%;
    }
    .result-top {
      background: #fff;
      color: #333;
      font-weight: bold;
      font-size: 20px;
      padding: 12px;
      border-radius: 12px;
      margin-bottom: 15px;
      text-align: center;
      box-shadow: 0 3px 10px rgba(0,0,0,0.2);
      animation: fadeIn 0.6s ease;
    }
    .result-item {
      background: rgba(255,255,255,0.9);
      color: #333;
      padding: 10px;
      border-radius: 10px;
      margin: 5px 0;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
      animation: fadeIn 0.6s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <h1>Object recognizer</h1>
  <p>Upload any photo and find out what's in it</p>

  <div class="upload-area">
    <input type="file" id="fileInput" accept="image/*">
    <div class="upload-content">
      <div class="upload-text">Click here to select a photo</div>
      <div class="upload-hint">or just drag a photo here</div>
    </div>
  </div>

  <div id="preview">
    <h3>Selected photo:</h3>
    <img id="previewImg" src="" alt="Preview">
  </div>

  <div id="results"></div>

  <script>
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const previewImg = document.getElementById("previewImg");
    const results = document.getElementById("results");

    fileInput.addEventListener("change", async (e) => {
      if (e.target.files.length > 0) {
        const file = e.target.files[0];

        const reader = new FileReader();
        reader.onload = function(evt) {
          previewImg.src = evt.target.result;
          preview.style.display = "block";
        };
        reader.readAsDataURL(file);
        const formData = new FormData();
        formData.append("file", file);

        results.innerHTML = "Recognition...";

        try {
          const response = await fetch("http://127.0.0.1:8000/classify", {
            method: "POST",
            body: formData
          });
          const data = await response.json();

          if (data.results) {
            results.innerHTML = "";
            const top = data.results[0];
            results.innerHTML += `
              <div class="result-top">
                ${top.label} — ${Math.round(top.score * 100)}%
              </div>
            `;
            data.results.slice(1).forEach(res => {
              results.innerHTML += `
                <div class="result-item">
                  ${res.label} — ${Math.round(res.score * 100)}%
                </div>
              `;
            });
          } else {
            results.innerHTML = "Error: " + JSON.stringify(data);
          }
        } catch (err) {
          results.innerHTML = "An error occurred while requesting the API";
        }
      }
    });
  </script>
</body>
</html>
    """

@app.post("/classify", response_model=PredictionsResponse, tags=["Recognition"])
async def classify_image(file: UploadFile = File(...)):
    """Recognizes objects in uploaded photos"""

    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recognition model is not available. Please try again later."
        )
    
    if file.content_type not in ALLOWED_MIMES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only photos in the following formats are supported: JPG, PNG, WEBP"
        )
    
    contents = await file.read(MAX_BYTES + 1)
    
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty. Please select another photo."
        )
    
    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is too large. Maximum size: {MAX_BYTES // (1024*1024)} MB"
        )

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to open photo. Please ensure it is a valid image."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error occurred while processing the photo."
        )

    try:
        raw_results = classifier(img)
    except Exception as e:
        print(f"Classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error occurred while analyzing the photo. Please try again."
        )

    if not raw_results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to obtain analysis results."
        )

    top3 = raw_results[:3] if len(raw_results) >= 3 else raw_results
    results = [
        {
            "label": result["label"], 
            "score": round(float(result["score"]), 4)
        } 
        for result in top3
    ]
    
    return {"results": results}

@app.get("/health", tags=["System"])
async def health_check():
    """Check system status"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)