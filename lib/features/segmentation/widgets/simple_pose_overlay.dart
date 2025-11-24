import 'dart:math';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class SimplePoseOverlay extends StatelessWidget {
  const SimplePoseOverlay({
    super.key,
    required this.poseDetections,
    required this.flipHorizontal,
    required this.flipVertical,
    this.modelInputSize = 640.0,
  });

  final List<YOLOResult> poseDetections;
  final bool flipHorizontal;
  final bool flipVertical;
  final double modelInputSize;

  @override
  Widget build(BuildContext context) {
    if (poseDetections.isEmpty) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _SimplePosePainter(
          poseDetections: poseDetections,
          flipHorizontal: flipHorizontal,
          flipVertical: flipVertical,
          modelInputSize: modelInputSize,
        ),
      ),
    );
  }
}

class _SimplePosePainter extends CustomPainter {
  _SimplePosePainter({
    required this.poseDetections,
    required this.flipHorizontal,
    required this.flipVertical,
    required this.modelInputSize,
  });

  final List<YOLOResult> poseDetections;
  final bool flipHorizontal;
  final bool flipVertical;
  final double modelInputSize;

  // Confidence thresholds
  static const double _noseConfidence = 0.15;
  static const double _eyeConfidence = 0.25;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;

    // Draw fat orange border for debugging
    final borderPaint = Paint()
      ..color = Colors.orange
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10.0;
    canvas.drawRect(Offset.zero & size, borderPaint);

    final poseDetection = _pickPrimaryPose(poseDetections, size);
    if (poseDetection == null) return;

    final keypoints = poseDetection.keypoints;
    if (keypoints == null || keypoints.isEmpty) return;

    final nose = _mapPosePoint(
      detection: poseDetection,
      index: 0,
      viewSize: size,
    );
    final leftEye = _mapPosePoint(
      detection: poseDetection,
      index: 1,
      viewSize: size,
    );
    final rightEye = _mapPosePoint(
      detection: poseDetection,
      index: 2,
      viewSize: size,
    );

    final nosePaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.fill;

    final leftEyePaint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.fill;

    final rightEyePaint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.fill;

    if (nose != null && nose.confidence >= _noseConfidence) {
      canvas.drawCircle(nose.imagePosition, 8.0, nosePaint);
    }

    if (leftEye != null && leftEye.confidence >= _eyeConfidence) {
      canvas.drawCircle(leftEye.imagePosition, 6.0, leftEyePaint);
    }

    if (rightEye != null && rightEye.confidence >= _eyeConfidence) {
      canvas.drawCircle(rightEye.imagePosition, 6.0, rightEyePaint);
    }
  }

  YOLOResult? _pickPrimaryPose(List<YOLOResult> list, Size size) {
    return list.first;
  }

  _PosePoint? _mapPosePoint({
    required YOLOResult detection,
    required int index,
    required Size viewSize,
  }) {
    final keypoints = detection.keypoints;
    final confidences = detection.keypointConfidences;
    if (keypoints == null || index < 0 || index >= keypoints.length) {
      return null;
    }
    final confidence = (confidences != null && index < confidences.length)
        ? confidences[index]
        : 1.0;
    if (confidence.isNaN) return null;

    final point = keypoints[index];
    if (!point.x.isFinite || !point.y.isFinite) return null;

    // Keypoints are likely in Model Input Coordinates (e.g. 640x640) with letterboxing.
    // BoundingBox is in Image Coordinates.
    // We need to transform Keypoints from Model Space to Image Space.

    final box = detection.boundingBox;
    final nBox = detection.normalizedBox;

    if (box.width <= 0 || nBox.width <= 0) return null;

    // 1. Infer Image Size
    final imgW = box.width / nBox.width;
    final imgH = box.height / nBox.height;

    // 2. Assume Model Input Size (Standard YOLO is 640)
    final modelSize = modelInputSize;

    // 3. Calculate Letterboxing parameters used by the model
    final scale = modelSize / max(imgW, imgH);
    final paddingX = (modelSize - imgW * scale) / 2.0;
    final paddingY = (modelSize - imgH * scale) / 2.0;

    // 4. Transform Keypoint (Model -> Image)
    // x_model = (x_img * scale) + paddingX
    // x_img = (x_model - paddingX) / scale
    double imgX = (point.x - paddingX) / scale;
    double imgY = (point.y - paddingY) / scale;

    // 5. Normalize (Image -> 0..1)
    double normX = imgX / imgW;
    double normY = imgY / imgH;

    // 6. Scale to View (0..1 -> View) using BoxFit.cover
    // We assume the camera preview fills the screen (BoxFit.cover), cropping if necessary.
    final scaleX = viewSize.width / imgW;
    final scaleY = viewSize.height / imgH;
    final scaleView = max(scaleX, scaleY); // BoxFit.cover

    // Calculate offsets to center the image in the view
    final dx = (viewSize.width - imgW * scaleView) / 2.0;
    final dy = (viewSize.height - imgH * scaleView) / 2.0;

    double screenX = normX * imgW * scaleView + dx;
    double screenY = normY * imgH * scaleView + dy;

    // 7. Apply Flip (if needed)
    if (flipHorizontal) {
      screenX = viewSize.width - screenX;
    }

    // Vertical flip removed as per user testing

    final imagePosition = Offset(screenX, screenY);

    return _PosePoint(imagePosition: imagePosition, confidence: confidence);
  }

  @override
  bool shouldRepaint(covariant _SimplePosePainter oldDelegate) {
    return oldDelegate.poseDetections != poseDetections ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical;
  }
}

class _PosePoint {
  const _PosePoint({required this.imagePosition, required this.confidence});

  final Offset imagePosition;
  final double confidence;
}
