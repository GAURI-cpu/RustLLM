#!/usr/bin/env sh
# Downloads the two OCR models kinSin uses to read text in photos.
# Run from the project root:   sh scripts/download_ocr_models.sh
# Then start the server as usual; it prints "Photo reading: on".
set -eu

dir="models/ocr"
base="https://ocrs-models.s3-accelerate.amazonaws.com"

mkdir -p "$dir"
for file in text-detection.onnx text-recognition.onnx; do
    echo "Downloading $file ..."
    if ! curl -L --fail -o "$dir/$file" "$base/$file"; then
        rm -f "$dir/$file"
        echo "Download of $file failed" >&2
        exit 1
    fi
done

echo "Done. The models are in $dir"
