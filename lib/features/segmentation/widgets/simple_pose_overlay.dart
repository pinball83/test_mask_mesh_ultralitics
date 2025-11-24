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

    // Draw blue bounding box
    _drawBoundingBox(canvas, poseDetection, size);

    // Draw landmarks
    _drawLandmarks(canvas, poseDetection, size);
  }

  YOLOResult? _pickPrimaryPose(List<YOLOResult> list, Size size) {
    return list.first;
  }

  void _drawBoundingBox(
    Canvas canvas,
    YOLOResult detection,
    Size viewSize,
  ) {
    final nBox = detection.normalizedBox;

    var left = nBox.left * viewSize.width;
    var top = nBox.top * viewSize.height;
    var right = nBox.right * viewSize.width;
    var bottom = nBox.bottom * viewSize.height;

    final rect = Rect.fromLTRB(left, top, right, bottom);
    final boxPaint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawRect(rect, boxPaint);
  }

  void _drawLandmarks(
    Canvas canvas,
    YOLOResult detection,
    Size viewSize,
  ) {
    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.isEmpty) return;

    final nose = _mapPosePoint(detection, 0, viewSize);
    final leftEye = _mapPosePoint(detection, 1, viewSize);
    final rightEye = _mapPosePoint(detection, 2, viewSize);

    if (nose != null && nose.confidence >= _noseConfidence) {
      canvas.drawCircle(nose.imagePosition, 8.0, Paint()..color = Colors.red);
    }
    if (leftEye != null && leftEye.confidence >= _eyeConfidence) {
      canvas.drawCircle(
        leftEye.imagePosition,
        6.0,
        Paint()..color = Colors.green,
      );
    }
    if (rightEye != null && rightEye.confidence >= _eyeConfidence) {
      canvas.drawCircle(
        rightEye.imagePosition,
        6.0,
        Paint()..color = Colors.blue,
      );
    }
  }

  _PosePoint? _mapPosePoint(
    YOLOResult detection,
    int index,
    Size viewSize,
  ) {
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

    // 4. Transform Keypoint (Model -> Image)
    double imgX = (point.x) / scale;
    double imgY = (point.y) / scale;

    // 5. Normalize (Image -> 0..1)
    double normX = imgX / imgW;
    double normY = imgY / imgH;

    double screenX = normX * viewSize.width;
    double screenY = normY * viewSize.height;

    // 3. Apply Flip
    if (flipHorizontal) {
      screenX = viewSize.width - screenX;
    }

    return _PosePoint(
      imagePosition: Offset(screenX, screenY),
      confidence: confidence,
    );
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
