# Downloads OpenAI's Whisper speech-to-text model for kinSin's voice input.
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\download_whisper.ps1          (base, ~290 MB: the default)
#   powershell -ExecutionPolicy Bypass -File scripts\download_whisper.ps1 tiny     (~150 MB: fastest, least accurate)
#   powershell -ExecutionPolicy Bypass -File scripts\download_whisper.ps1 small    (~970 MB: slower, more accurate)
# Then start the server as usual; it prints "Voice input: on".

param([ValidateSet('tiny', 'base', 'small')][string]$Size = 'base')

$ErrorActionPreference = 'Stop'
$dir = 'models/whisper'
$base = "https://huggingface.co/openai/whisper-$Size/resolve/main"

New-Item -ItemType Directory -Force -Path $dir | Out-Null

foreach ($file in 'config.json', 'tokenizer.json', 'model.safetensors') {
    $target = Join-Path $dir $file
    Write-Host "Downloading whisper-$Size/$file ..."
    # curl.exe (not the PowerShell alias): -L follows redirects, --fail stops on HTTP errors
    curl.exe -L --fail -o $target "$base/$file"
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -ErrorAction SilentlyContinue $target
        throw "Download of $file failed"
    }
}

Write-Host "Done. Whisper $Size is in $dir"
