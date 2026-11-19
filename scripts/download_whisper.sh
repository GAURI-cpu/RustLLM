#!/usr/bin/env sh
# Downloads OpenAI's Whisper speech-to-text model for kinSin's voice input.
# Run from the project root:
#   sh scripts/download_whisper.sh          (base, ~290 MB: the default)
#   sh scripts/download_whisper.sh tiny     (~150 MB: fastest, least accurate)
#   sh scripts/download_whisper.sh small    (~970 MB: slower, more accurate)
# Then start the server as usual; it prints "Voice input: on".
set -eu

size="${1:-base}"
case "$size" in
    tiny|base|small) ;;
    *) echo "size must be tiny, base or small (got: $size)" >&2; exit 1 ;;
esac

dir="models/whisper"
base="https://huggingface.co/openai/whisper-$size/resolve/main"

mkdir -p "$dir"
for file in config.json tokenizer.json model.safetensors; do
    echo "Downloading whisper-$size/$file ..."
    if ! curl -L --fail -o "$dir/$file" "$base/$file"; then
        rm -f "$dir/$file"
        echo "Download of $file failed" >&2
        exit 1
    fi
done

echo "Done. Whisper $size is in $dir"
