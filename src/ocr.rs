//! Reading the text in photos (OCR), with the `ocrs` engine.
//!
//! Needs two model files in one folder (see scripts/download_ocr_models.*):
//!   text-detection.onnx    finds where the words are
//!   text-recognition.onnx  reads them

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use image::RgbImage;
use image::imageops::FilterType;
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;

const DETECTION_FILE: &str = "text-detection.onnx";
const RECOGNITION_FILE: &str = "text-recognition.onnx";
/// Photos bigger than this (longest side, in pixels) are shrunk first: OCR is slow on huge images.
const MAX_SIDE: u32 = 2000;
const NOT_A_PHOTO: &str = "this file is not a photo kinSin can open (try JPG or PNG)";

pub struct Ocr {
    engine: OcrEngine,
}

impl Ocr {
    /// Are both model files in `dir`?
    pub fn models_present(dir: &Path) -> bool {
        dir.join(DETECTION_FILE).is_file() && dir.join(RECOGNITION_FILE).is_file()
    }

    pub fn load(dir: &Path) -> Result<Ocr> {
        let detection_path = dir.join(DETECTION_FILE);
        let recognition_path = dir.join(RECOGNITION_FILE);
        let detection_model = Model::load_file(&detection_path)
            .with_context(|| format!("could not load the OCR detection model {}", detection_path.display()))?;
        let recognition_model = Model::load_file(&recognition_path)
            .with_context(|| format!("could not load the OCR recognition model {}", recognition_path.display()))?;
        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection_model),
            recognition_model: Some(recognition_model),
            ..Default::default()
        })
        .context("could not start the OCR engine")?;
        Ok(Ocr { engine })
    }

    /// The text in a photo (JPG, PNG, ...), trimmed. Can take a few seconds: call it off the async runtime.
    pub fn read_text(&self, bytes: &[u8]) -> Result<String> {
        let img = prepare_image(bytes)?;
        let source = ImageSource::from_bytes(img.as_raw(), img.dimensions()).map_err(|e| anyhow!("{e}"))?;
        let input = self.engine.prepare_input(source).context("could not prepare the photo for reading")?;
        let text = self.engine.get_text(&input).context("could not read the text in the photo")?;
        Ok(text.trim().to_string())
    }
}

/// Decode the photo and shrink it so its longest side is at most MAX_SIDE (keeping its shape).
fn prepare_image(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.is_empty() {
        return Err(anyhow!(NOT_A_PHOTO));
    }
    let img = image::load_from_memory(bytes).map_err(|_| anyhow!(NOT_A_PHOTO))?;
    let img = if img.width().max(img.height()) > MAX_SIDE {
        img.resize(MAX_SIDE, MAX_SIDE, FilterType::Triangle)
    } else {
        img
    };
    Ok(img.into_rgb8())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        RgbImage::new(width, height)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        bytes
    }

    #[test]
    fn non_image_bytes_fail() {
        let err = prepare_image(b"this is not a picture").unwrap_err();
        assert!(err.to_string().contains("not a photo"));
    }

    #[test]
    fn empty_bytes_fail() {
        let err = prepare_image(&[]).unwrap_err();
        assert!(err.to_string().contains("not a photo"));
    }

    #[test]
    fn big_photo_is_shrunk() {
        let img = prepare_image(&png(4000, 1000)).unwrap();
        assert_eq!(img.dimensions(), (2000, 500));
    }

    #[test]
    fn small_photo_stays_the_same() {
        let img = prepare_image(&png(800, 600)).unwrap();
        assert_eq!(img.dimensions(), (800, 600));
    }

    #[test]
    fn missing_models_are_detected() {
        assert!(!Ocr::models_present(Path::new("no/such/folder")));
    }
}
