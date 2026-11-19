# Downloads the two OCR models kinSin uses to read text in photos.
# Run from the project root:   powershell -ExecutionPolicy Bypass -File scripts\download_ocr_models.ps1
# Then start the server as usual; it prints "Photo reading: on".

$ErrorActionPreference = 'Stop'
$dir = 'models/ocr'
$base = 'https://ocrs-models.s3-accelerate.amazonaws.com'

New-Item -ItemType Directory -Force -Path $dir | Out-Null

foreach ($file in 'text-detection.onnx', 'text-recognition.onnx') {
    $target = Join-Path $dir $file
    Write-Host "Downloading $file ..."
    # curl.exe (not the PowerShell alias): -L follows redirects, --fail stops on HTTP errors
    curl.exe -L --fail -o $target "$base/$file"
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -ErrorAction SilentlyContinue $target
        throw "Download of $file failed"
    }
}

Write-Host "Done. The models are in $dir"
