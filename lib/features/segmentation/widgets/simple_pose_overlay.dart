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

    return LayoutBuilder(
      builder: (context, constraints) {
        final size = Size(constraints.maxWidth, constraints.maxHeight);
        return Stack(
          children: [
            IgnorePointer(
              child: CustomPaint(
                painter: _SimplePosePainter(
                  poseDetections: poseDetections,
                  flipHorizontal: flipHorizontal,
                  flipVertical: flipVertical,
                  modelInputSize: modelInputSize,
                ),
                size: size,
              ),
            ),
            if (poseDetections.isNotEmpty)
              Positioned(
                top: 50,
                right: 10,
                child: _DebugInfoOverlay(
                  detection: poseDetections.first,
                  viewSize: size,
                  modelInputSize: modelInputSize,
                  flipHorizontal: flipHorizontal,
                  flipVertical: flipVertical,
                ),
              ),
          ],
        );
      },
    );
  }
}

class _DebugInfoOverlay extends StatelessWidget {
  const _DebugInfoOverlay({
    required this.detection,
    required this.viewSize,
    required this.modelInputSize,
    required this.flipHorizontal,
    required this.flipVertical,
  });

  final YOLOResult detection;
  final Size viewSize;
  final double modelInputSize;
  final bool flipHorizontal;
  final bool flipVertical;

  @override
  Widget build(BuildContext context) {
    final box = detection.boundingBox;
    final nBox = detection.normalizedBox;
    final keypoints = detection.keypoints;

    String keypointInfo = 'No Keypoints';
    if (keypoints != null && keypoints.isNotEmpty) {
      final nose = keypoints[0];
      keypointInfo =
          'Nose: (${nose.x.toStringAsFixed(1)}, ${nose.y.toStringAsFixed(1)})';
    }

    // Calculations from _SimplePosePainter
    double imgW = 0;
    double imgH = 0;
    double scale = 0;

    if (nBox.width > 0 && nBox.height > 0) {
      imgW = box.width / nBox.width;
      imgH = box.height / nBox.height;
      scale = modelInputSize / max(imgW, imgH);
    }

    return Container(
      padding: const EdgeInsets.all(8),
      color: Colors.black.withOpacity(0.7),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'View: ${viewSize.width.toStringAsFixed(0)}x${viewSize.height.toStringAsFixed(0)}',
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
          Text(
            'Box: ${box.width.toStringAsFixed(0)}x${box.height.toStringAsFixed(0)} @ (${box.left.toStringAsFixed(0)}, ${box.top.toStringAsFixed(0)})',
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
          Text(
            'NBox: ${nBox.width.toStringAsFixed(2)}x${nBox.height.toStringAsFixed(2)}',
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
          Text(
            'Img: ${detection.imageSize != null ? "${detection.imageSize!.width.toStringAsFixed(0)}x${detection.imageSize!.height.toStringAsFixed(0)} (Native)" : "${imgW.toStringAsFixed(0)}x${imgH.toStringAsFixed(0)} (Inferred)"}',
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
          Text(
            'Scale: ${scale.toStringAsFixed(3)}',
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
          Text(
            keypointInfo,
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
          Text(
            'Flip: H=$flipHorizontal, V=$flipVertical',
            style: const TextStyle(color: Colors.white, fontSize: 10),
          ),
        ],
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
    // canvas.drawRect(Offset.zero & size, borderPaint);

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

  void _drawBoundingBox(Canvas canvas, YOLOResult detection, Size viewSize) {
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

  void _drawLandmarks(Canvas canvas, YOLOResult detection, Size viewSize) {
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

  _PosePoint? _mapPosePoint(YOLOResult detection, int index, Size viewSize) {
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

    double? imageW = detection.imageSize?.width;
    double? imageH = detection.imageSize?.height;
    if (imageW == null || imageH == null) {
      final box = detection.boundingBox;
      final nBox = detection.normalizedBox;
      if (box.width <= 0 || nBox.width <= 0) return null;
      imageW = box.width / nBox.width;
      imageH = box.height / nBox.height;
    }

    // Keypoints are reported normalized (0..1). Convert to image pixel space
    // when values look normalized; otherwise assume pixel coordinates.
    var kpX = point.x;
    var kpY = point.y;
    if (kpX.abs() <= 1.2 && kpY.abs() <= 1.2) {
      kpX = kpX * imageW;
      kpY = kpY * imageH;
    }

    final scale = max(viewSize.width / imageW, viewSize.height / imageH);
    final scaledW = imageW * scale;
    final scaledH = imageH * scale;
    final dx = (viewSize.width - scaledW) / 2.0;
    final dy = (viewSize.height - scaledH) / 2.0;

    double screenX = (kpX * scale) + dx;
    double screenY = (kpY * scale) + dy;

    // 3. Apply Flip
    if (flipHorizontal) {
      screenX = viewSize.width - screenX;
    }

    if (flipVertical) {
      screenY = viewSize.height - screenY;
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
