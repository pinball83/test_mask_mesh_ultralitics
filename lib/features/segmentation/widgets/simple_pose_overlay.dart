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
  static const int _logThrottleMs = 350;
  static DateTime? _lastLog;

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

    final imageSize = _resolveImageSize(poseDetection);
    if (imageSize == null) return;

    // Compute cover scale and offsets once per frame so boxes and keypoints
    // share the exact same mapping as the camera preview.
    final scale = max(size.width / imageSize.width, size.height / imageSize.height);
    final scaledW = imageSize.width * scale;
    final scaledH = imageSize.height * scale;
    final dx = (size.width - scaledW) / 2.0;
    final dy = (size.height - scaledH) / 2.0;

    // Draw blue bounding box
    _drawBoundingBox(
      canvas,
      poseDetection,
      size,
      scale: scale,
      dx: dx,
      dy: dy,
      imageSize: imageSize,
    );

    // Draw landmarks
    _drawLandmarks(
      canvas,
      poseDetection,
      size,
      scale: scale,
      dx: dx,
      dy: dy,
      imageSize: imageSize,
    );
  }

  YOLOResult? _pickPrimaryPose(List<YOLOResult> list, Size size) {
    return list.first;
  }

  void _drawBoundingBox(
    Canvas canvas,
    YOLOResult detection,
    Size viewSize, {
    required double scale,
    required double dx,
    required double dy,
    required Size imageSize,
  }) {
    double left = detection.normalizedBox.left * imageSize.width * scale + dx;
    double right = detection.normalizedBox.right * imageSize.width * scale + dx;
    double top = detection.normalizedBox.top * imageSize.height * scale + dy;
    double bottom = detection.normalizedBox.bottom * imageSize.height * scale + dy;

    if (flipHorizontal) {
      final origLeft = left;
      left = viewSize.width - right;
      right = viewSize.width - origLeft;
    }
    if (flipVertical) {
      final origTop = top;
      top = viewSize.height - bottom;
      bottom = viewSize.height - origTop;
    }

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
    Size viewSize, {
    required double scale,
    required double dx,
    required double dy,
    required Size imageSize,
  }) {
    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.isEmpty) return;

    final nose = _mapPosePoint(
      detection: detection,
      index: 0,
      viewSize: viewSize,
      scale: scale,
      dx: dx,
      dy: dy,
      imageSize: imageSize,
    );
    final leftEye = _mapPosePoint(
      detection: detection,
      index: 1,
      viewSize: viewSize,
      scale: scale,
      dx: dx,
      dy: dy,
      imageSize: imageSize,
    );
    final rightEye = _mapPosePoint(
      detection: detection,
      index: 2,
      viewSize: viewSize,
      scale: scale,
      dx: dx,
      dy: dy,
      imageSize: imageSize,
    );

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

    _logDebugMapping(
      viewSize: viewSize,
      imageSize: imageSize,
      scale: scale,
      dx: dx,
      dy: dy,
      detection: detection,
      nose: nose,
      leftEye: leftEye,
      rightEye: rightEye,
    );
  }

  _PosePoint? _mapPosePoint({
    required YOLOResult detection,
    required int index,
    required Size viewSize,
    required double scale,
    required double dx,
    required double dy,
    required Size imageSize,
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

    double x = point.x;
    double y = point.y;

    // Build the view-space bounding box (post-flip) for containment checks.
    double boxLeft = detection.normalizedBox.left * imageSize.width * scale + dx;
    double boxRight = detection.normalizedBox.right * imageSize.width * scale + dx;
    double boxTop = detection.normalizedBox.top * imageSize.height * scale + dy;
    double boxBottom = detection.normalizedBox.bottom * imageSize.height * scale + dy;
    if (flipHorizontal) {
      final origLeft = boxLeft;
      boxLeft = viewSize.width - boxRight;
      boxRight = viewSize.width - origLeft;
    }
    if (flipVertical) {
      final origTop = boxTop;
      boxTop = viewSize.height - boxBottom;
      boxBottom = viewSize.height - origTop;
    }
    final viewBox = Rect.fromLTRB(boxLeft, boxTop, boxRight, boxBottom);
    final boxCenter = viewBox.center;

    // Candidate 1: assume full-image normalized keypoints.
    Offset? candidate1;
    double cand1Dist = double.infinity;
    double cx1 = x;
    double cy1 = y;
    if (x >= 0.0 && x <= 1.2 && y >= 0.0 && y <= 1.2) {
      cx1 = x.clamp(0.0, 1.0) * imageSize.width;
      cy1 = y.clamp(0.0, 1.0) * imageSize.height;
    }
    cx1 = (cx1 * scale) + dx;
    cy1 = (cy1 * scale) + dy;
    if (flipHorizontal) cx1 = viewSize.width - cx1;
    if (flipVertical) cy1 = viewSize.height - cy1;
    candidate1 = Offset(cx1, cy1);
    cand1Dist = (candidate1 - boxCenter).distanceSquared;

    // Candidate 2: if keypoints looked normalized, assume they were relative
    // to the bounding box. Map that to image pixels, then to view space.
    Offset? candidate2;
    double cand2Dist = double.infinity;
    if (x >= 0.0 && x <= 1.2 && y >= 0.0 && y <= 1.2) {
      double bx = (detection.normalizedBox.left + x.clamp(0.0, 1.0) * detection.normalizedBox.width) * imageSize.width;
      double by = (detection.normalizedBox.top + y.clamp(0.0, 1.0) * detection.normalizedBox.height) * imageSize.height;
      bx = (bx * scale) + dx;
      by = (by * scale) + dy;
      if (flipHorizontal) bx = viewSize.width - bx;
      if (flipVertical) by = viewSize.height - by;
      candidate2 = Offset(bx, by);
      cand2Dist = (candidate2 - boxCenter).distanceSquared;
    }

    // Pick a candidate that lands inside the box; otherwise choose the closer one.
    Offset chosen;
    if (candidate1 != null && viewBox.contains(candidate1)) {
      chosen = candidate1;
    } else if (candidate2 != null && viewBox.contains(candidate2)) {
      chosen = candidate2;
    } else {
      chosen = (cand2Dist < cand1Dist && candidate2 != null) ? candidate2 : candidate1!;
    }

    return _PosePoint(
      imagePosition: chosen,
      confidence: confidence,
    );
  }

  // Resolve the source image size, preferring explicit imageSize and falling
  // back to inferring from the absolute vs normalized bounding box.
  Size? _resolveImageSize(YOLOResult detection) {
    final img = detection.imageSize;
    if (img != null && img.width > 0 && img.height > 0) {
      return Size(img.width, img.height);
    }

    final bb = detection.boundingBox;
    final nb = detection.normalizedBox;
    if (bb.width > 0 && nb.width > 0 && bb.height > 0 && nb.height > 0) {
      final w = bb.width / nb.width;
      final h = bb.height / nb.height;
      if (w.isFinite && h.isFinite && w > 0 && h > 0) {
        return Size(w, h);
      }
    }
    return null;
  }

  void _logDebugMapping({
    required Size viewSize,
    required Size imageSize,
    required double scale,
    required double dx,
    required double dy,
    required YOLOResult detection,
    _PosePoint? nose,
    _PosePoint? leftEye,
    _PosePoint? rightEye,
  }) {
    final now = DateTime.now();
    if (_lastLog != null &&
        now.difference(_lastLog!).inMilliseconds < _logThrottleMs) {
      return;
    }
    _lastLog = now;

    final nb = detection.normalizedBox;
    final bb = detection.boundingBox;

    debugPrint(
      [
        'POSE MAP — view=${viewSize.width.toStringAsFixed(0)}x${viewSize.height.toStringAsFixed(0)}'
            ' img=${imageSize.width.toStringAsFixed(0)}x${imageSize.height.toStringAsFixed(0)}'
            ' scale=${scale.toStringAsFixed(3)} dx=${dx.toStringAsFixed(1)} dy=${dy.toStringAsFixed(1)}',
        'nb=L${nb.left.toStringAsFixed(3)} T${nb.top.toStringAsFixed(3)} '
            'R${nb.right.toStringAsFixed(3)} B${nb.bottom.toStringAsFixed(3)}',
        'bb=L${bb.left.toStringAsFixed(1)} T${bb.top.toStringAsFixed(1)} '
            'R${bb.right.toStringAsFixed(1)} B${bb.bottom.toStringAsFixed(1)}',
        if (nose != null)
          'nose=${nose.imagePosition.dx.toStringAsFixed(1)},${nose.imagePosition.dy.toStringAsFixed(1)} '
              'conf=${nose.confidence.toStringAsFixed(2)}',
        if (leftEye != null)
          'lEye=${leftEye.imagePosition.dx.toStringAsFixed(1)},${leftEye.imagePosition.dy.toStringAsFixed(1)} '
              'conf=${leftEye.confidence.toStringAsFixed(2)}',
        if (rightEye != null)
          'rEye=${rightEye.imagePosition.dx.toStringAsFixed(1)},${rightEye.imagePosition.dy.toStringAsFixed(1)} '
              'conf=${rightEye.confidence.toStringAsFixed(2)}',
        'flips: H=$flipHorizontal V=$flipVertical',
      ].join(' | '),
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
