# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Initial release (M1+M2)**: faithful YOLO26 detection reproduction — architecture with
  verified weight alignment (2,572,280 params, strict load, bit-identical to official output),
  inference (.pt/.onnx, E2E NMS-free & NMS paths), pt→onnx export, COCO evaluation
  (40.27 / 40.89 vs official 40.1 / 40.9), ultralytics-style logging, progress bar and
  runs/ result layout.
