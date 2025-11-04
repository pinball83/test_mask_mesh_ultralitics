import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

/// Pose-driven overlay that draws facial debug markers and a mustache sprite.
/// The name is kept for backwards compatibility with the previous shader-based implementation.
class SelfieShaderOverlay extends StatefulWidget {
  const SelfieShaderOverlay({
    super.key,
    required this.detections,
    required this.poseDetections,
    required this.flipHorizontal,
    required this.flipVertical,
    this.mustacheAsset = 'assets/images/mustash.png',
    this.mustacheAlpha = 1.0,
    this.debugPose = false,
    this.showMustache = true,
  });

  final List<YOLOResult> detections;
  final List<YOLOResult> poseDetections;
  final bool flipHorizontal;
  final bool flipVertical;
  final String mustacheAsset;
  final double mustacheAlpha;
  final bool debugPose;
  final bool showMustache;

  @override
  State<SelfieShaderOverlay> createState() => _SelfieShaderOverlayState();
}

class _SelfieShaderOverlayState extends State<SelfieShaderOverlay> {
  ui.Image? _mustacheImage;
  ImageStream? _mustacheStream;
  ImageStreamListener? _mustacheListener;

  double? _srcW;
  double? _srcH;

  @override
  void initState() {
    super.initState();
    _resolveMustache();
    _updateSourceSize();
  }

  @override
  void didUpdateWidget(covariant SelfieShaderOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.mustacheAsset != widget.mustacheAsset) {
      _resolveMustache();
    }
    if (!identical(oldWidget.detections, widget.detections)) {
      _updateSourceSize();
    }
  }

  @override
  void dispose() {
    _disposeMustacheStream();
    super.dispose();
  }

  void _resolveMustache() {
    _disposeMustacheStream();
    final provider = AssetImage(widget.mustacheAsset);
    final stream = provider.resolve(ImageConfiguration.empty);
    _mustacheStream = stream;
    _mustacheListener = ImageStreamListener((imageInfo, _) {
      if (mounted) {
        setState(() => _mustacheImage = imageInfo.image);
      }
    });
    stream.addListener(_mustacheListener!);
  }

  void _disposeMustacheStream() {
    if (_mustacheStream != null && _mustacheListener != null) {
      _mustacheStream!.removeListener(_mustacheListener!);
    }
    _mustacheStream = null;
    _mustacheListener = null;
  }

  void _updateSourceSize() {
    setState(() {
      _srcW = _estimateSourceWidth(widget.detections);
      _srcH = _estimateSourceHeight(widget.detections);
    });
  }

  double? _estimateSourceWidth(List<YOLOResult> list) {
    for (final d in list) {
      final nb = d.normalizedBox;
      final bb = d.boundingBox;
      if (nb.width > 0 && bb.width > 0 && nb.width.isFinite) {
        final candidate = bb.width / nb.width;
        if (candidate.isFinite && candidate > 0) return candidate;
      }
    }
    return null;
  }

  double? _estimateSourceHeight(List<YOLOResult> list) {
    for (final d in list) {
      final nb = d.normalizedBox;
      final bb = d.boundingBox;
      if (nb.height > 0 && bb.height > 0 && nb.height.isFinite) {
        final candidate = bb.height / nb.height;
        if (candidate.isFinite && candidate > 0) return candidate;
      }
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    if (widget.poseDetections.isEmpty && !widget.debugPose) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _PoseOverlayPainter(
          detections: widget.detections,
          poseDetections: widget.poseDetections,
          mustacheImage: _mustacheImage,
          mustacheAlpha: widget.mustacheAlpha.clamp(0.0, 1.0).toDouble(),
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
          sourceWidth: _srcW,
          sourceHeight: _srcH,
          debugPose: widget.debugPose,
          showMustache: widget.showMustache,
        ),
      ),
    );
  }
}

class _PoseOverlayPainter extends CustomPainter {
  _PoseOverlayPainter({
    required this.detections,
    required this.poseDetections,
    required this.mustacheImage,
    required this.mustacheAlpha,
    required this.flipHorizontal,
    required this.flipVertical,
    required this.sourceWidth,
    required this.sourceHeight,
    required this.debugPose,
    required this.showMustache,
  });

  final List<YOLOResult> detections;
  final List<YOLOResult> poseDetections;
  final ui.Image? mustacheImage;
  final double mustacheAlpha;
  final bool flipHorizontal;
  final bool flipVertical;
  final double? sourceWidth;
  final double? sourceHeight;
  final bool debugPose;
  final bool showMustache;

  static const double _minMustacheWidth = 24;
  static const double _maxMustacheWidthFactor = 0.6;
  static const double _fallbackWidthFactor = 0.36;
  static const double _eyeWidthMultiplier = 1.08;
  static const double _noseConfidence = 0.15;
  static const double _eyeConfidence = 0.25;
  static const double _candidateConfidence = 0.05;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;

    final transform = _LetterboxTransform.fromView(
      viewSize: size,
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
    );

    final poseDetection = _pickPrimaryPose(poseDetections, size);
    final maskDetection = _pickPrimaryMask(detections);

    Rect? faceRectImage;
    Offset? noseImage;
    Offset? leftEyeImage;
    Offset? rightEyeImage;
    Offset? noseBridgeStartImage;
    Offset? noseBridgeEndImage;
    Offset? upperLipImage;

    if (poseDetection != null) {
      final landmarks = _extractPoseLandmarks(
        detection: poseDetection,
        transform: transform,
        seedRect: faceRectImage,
      );
      faceRectImage = landmarks.faceRect ?? faceRectImage;
      noseImage = landmarks.nose;
      leftEyeImage = landmarks.leftEye;
      rightEyeImage = landmarks.rightEye;
      noseBridgeStartImage = landmarks.noseBridgeStart;
      noseBridgeEndImage = landmarks.noseBridgeEnd ?? noseImage;
      upperLipImage = landmarks.upperLipCenter;
    }

    if (faceRectImage == null && maskDetection != null) {
      faceRectImage = transform.viewRectToImage(maskDetection.boundingBox);
    }

    final faceRectView = _mapRectToCanvas(faceRectImage, transform, size);
    final noseView = _mapPointToCanvas(noseImage, transform, size);
    final leftEyeView = _mapPointToCanvas(leftEyeImage, transform, size);
    final rightEyeView = _mapPointToCanvas(rightEyeImage, transform, size);
    final noseBridgeStartView = _mapPointToCanvas(
      noseBridgeStartImage,
      transform,
      size,
    );
    final noseBridgeEndView = _mapPointToCanvas(
      noseBridgeEndImage,
      transform,
      size,
    );
    final upperLipView = _mapPointToCanvas(upperLipImage, transform, size);

    _drawMustache(
      canvas,
      size,
      faceRectView: faceRectView,
      leftEye: leftEyeView,
      rightEye: rightEyeView,
      nose: noseView,
      noseBridgeEnd: noseBridgeEndView,
    );

    if (debugPose) {
      _drawDebug(
        canvas,
        faceRect: faceRectView,
        leftEye: leftEyeView,
        rightEye: rightEyeView,
        nose: noseView,
        noseBridgeStart: noseBridgeStartView,
        noseBridgeEnd: noseBridgeEndView,
        upperLip: upperLipView,
      );
    }
  }

  void _drawMustache(
    Canvas canvas,
    Size size, {
    Rect? faceRectView,
    Offset? leftEye,
    Offset? rightEye,
    Offset? nose,
    Offset? noseBridgeEnd,
  }) {
    if (!showMustache) return;
    final image = mustacheImage;
    if (image == null) return;

    final width = _resolveMustacheWidth(
      faceRectView: faceRectView,
      leftEye: leftEye,
      rightEye: rightEye,
      viewSize: size,
    );
    final height = width * (image.height / image.width);

    double rotation = 0;
    if (leftEye != null && rightEye != null) {
      rotation = math.atan2(rightEye.dy - leftEye.dy, rightEye.dx - leftEye.dx);
    }

    final anchor =
        nose ??
        noseBridgeEnd ??
        faceRectView?.center ??
        Offset(size.width / 2, size.height / 2);
    final offsetX = -math.sin(rotation) * (height / 2);
    final offsetY = math.cos(rotation) * (height / 2);
    final center = Offset(anchor.dx + offsetX, anchor.dy + offsetY);

    canvas.save();
    canvas.translate(center.dx, center.dy);
    if (rotation != 0) {
      canvas.rotate(rotation);
    }

    final rect = Rect.fromCenter(
      center: Offset.zero,
      width: width,
      height: height,
    );

    paintImage(
      canvas: canvas,
      rect: rect,
      image: image,
      fit: BoxFit.contain,
      filterQuality: FilterQuality.high,
      colorFilter: mustacheAlpha < 0.99
          ? ColorFilter.mode(
              Colors.white.withValues(alpha: mustacheAlpha),
              BlendMode.modulate,
            )
          : null,
    );

    canvas.restore();
  }

  void _drawDebug(
    Canvas canvas, {
    Rect? faceRect,
    Offset? leftEye,
    Offset? rightEye,
    Offset? nose,
    Offset? noseBridgeStart,
    Offset? noseBridgeEnd,
    Offset? upperLip,
  }) {
    final facePaint = Paint()
      ..color = Colors.limeAccent.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    final eyeOutline = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    final eyeFill = Paint()
      ..color = Colors.cyanAccent.withValues(alpha: 0.35)
      ..style = PaintingStyle.fill;
    final helperLine = Paint()
      ..color = Colors.amberAccent.withValues(alpha: 0.8)
      ..strokeWidth = 1.4;
    final noseFill = Paint()
      ..color = Colors.deepOrangeAccent
      ..style = PaintingStyle.fill;
    final noseCross = Paint()
      ..color = Colors.deepOrangeAccent
      ..strokeWidth = 1.1;
    final lipPaint = Paint()
      ..color = Colors.purpleAccent.withValues(alpha: 0.7)
      ..style = PaintingStyle.fill;

    if (faceRect != null) {
      canvas.drawRect(faceRect, facePaint);
    }

    if (leftEye != null) {
      canvas.drawCircle(leftEye, 7, eyeOutline);
      canvas.drawCircle(leftEye, 3, eyeFill);
    }
    if (rightEye != null) {
      canvas.drawCircle(rightEye, 7, eyeOutline);
      canvas.drawCircle(rightEye, 3, eyeFill);
    }
    if (leftEye != null && rightEye != null) {
      canvas.drawLine(leftEye, rightEye, helperLine);
    }

    if (nose != null) {
      canvas.drawCircle(nose, 5, noseFill);
      canvas.drawLine(
        Offset(nose.dx - 6, nose.dy),
        Offset(nose.dx + 6, nose.dy),
        noseCross,
      );
      canvas.drawLine(
        Offset(nose.dx, nose.dy - 6),
        Offset(nose.dx, nose.dy + 6),
        noseCross,
      );
    }

    if (noseBridgeStart != null && noseBridgeEnd != null) {
      canvas.drawLine(noseBridgeStart, noseBridgeEnd, helperLine);
    }

    if (upperLip != null) {
      canvas.drawCircle(upperLip, 4, lipPaint);
    }
  }

  double _resolveMustacheWidth({
    required Rect? faceRectView,
    required Offset? leftEye,
    required Offset? rightEye,
    required Size viewSize,
  }) {
    double width;
    if (leftEye != null && rightEye != null) {
      final eyeSpan = (leftEye - rightEye).distance;
      width = eyeSpan * _eyeWidthMultiplier;
    } else if (faceRectView != null) {
      width = faceRectView.width * _fallbackWidthFactor;
    } else {
      width = viewSize.width * _fallbackWidthFactor;
    }

    if (!width.isFinite || width <= 0) {
      width = viewSize.width * 0.3;
    }

    return width.clamp(
      _minMustacheWidth,
      viewSize.width * _maxMustacheWidthFactor,
    );
  }

  YOLOResult? _pickPrimaryMask(List<YOLOResult> list) {
    YOLOResult? best;
    var bestArea = -1.0;
    for (final d in list) {
      final mask = d.mask;
      final box = d.boundingBox;
      if (mask == null || mask.isEmpty || box.isEmpty) continue;
      final area = (box.width * box.height).abs();
      if (area > bestArea) {
        bestArea = area;
        best = d;
      }
    }
    return best;
  }

  YOLOResult? _pickPrimaryPose(List<YOLOResult> list, Size size) {
    YOLOResult? best;
    var bestScore = double.negativeInfinity;
    for (final d in list) {
      final keypoints = d.keypoints;
      if (keypoints == null || keypoints.isEmpty) continue;
      final box = d.boundingBox;
      if (box.isEmpty) continue;
      final area = box.width.abs() * box.height.abs();
      if (area <= 0) continue;
      final cx = box.left + box.width / 2;
      final cy = box.top + box.height / 2;
      final dx = cx - size.width / 2;
      final dy = cy - size.height / 2;
      final distPenalty = dx * dx + dy * dy;
      final score = area * 0.5 - distPenalty * 2.0;
      if (score > bestScore) {
        bestScore = score;
        best = d;
      }
    }
    return best;
  }

  Rect? _mapRectToCanvas(Rect? rect, _LetterboxTransform transform, Size size) {
    if (rect == null || rect.isEmpty) return null;
    final topLeft = _mapPointToCanvas(rect.topLeft, transform, size);
    final bottomRight = _mapPointToCanvas(rect.bottomRight, transform, size);
    if (topLeft == null || bottomRight == null) return null;
    final left = math.min(topLeft.dx, bottomRight.dx);
    final right = math.max(topLeft.dx, bottomRight.dx);
    final top = math.min(topLeft.dy, bottomRight.dy);
    final bottom = math.max(topLeft.dy, bottomRight.dy);
    if (right <= left || bottom <= top) return null;
    return Rect.fromLTRB(left, top, right, bottom);
  }

  Offset? _mapPointToCanvas(
    Offset? imagePoint,
    _LetterboxTransform transform,
    Size size,
  ) {
    if (imagePoint == null) return null;
    final viewPoint = transform.imageToView(imagePoint);
    var dx = viewPoint.dx;
    var dy = viewPoint.dy;
    if (flipHorizontal) dx = size.width - dx;
    if (flipVertical) dy = size.height - dy;
    if (!dx.isFinite || !dy.isFinite) return null;
    return Offset(dx.clamp(0.0, size.width), dy.clamp(0.0, size.height));
  }

  Rect _clampRectToImage(Rect rect, _LetterboxTransform transform) {
    final left = rect.left.clamp(0.0, transform.srcWidth);
    final top = rect.top.clamp(0.0, transform.srcHeight);
    final right = rect.right.clamp(0.0, transform.srcWidth);
    final bottom = rect.bottom.clamp(0.0, transform.srcHeight);
    if (right <= left || bottom <= top) {
      return Rect.zero;
    }
    return Rect.fromLTRB(left, top, right, bottom);
  }

  _PoseLandmarks _extractPoseLandmarks({
    required YOLOResult detection,
    required _LetterboxTransform transform,
    Rect? seedRect,
  }) {
    Rect? faceRect = seedRect;
    if (faceRect == null && !detection.boundingBox.isEmpty) {
      faceRect = _clampRectToImage(
        transform.viewRectToImage(detection.boundingBox),
        transform,
      );
    }

    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.isEmpty) {
      return _PoseLandmarks(faceRect: faceRect);
    }

    final points = List<_PosePoint?>.generate(
      keypoints.length,
      (index) => _mapPosePoint(
        detection: detection,
        index: index,
        transform: transform,
      ),
      growable: false,
    );

    Offset? noseBridgeEnd;
    Offset? nose;
    final nosePoint = _pointWithThreshold(points, 0, _noseConfidence);
    if (nosePoint != null) {
      noseBridgeEnd = nosePoint.imagePosition;
      nose = nosePoint.imagePosition;
    }

    Offset? noseBridgeStart;
    Offset? leftEyePosition;
    Offset? rightEyePosition;
    final leftEyePoint = _pointWithThreshold(points, 1, _eyeConfidence);
    final rightEyePoint = _pointWithThreshold(points, 2, _eyeConfidence);
    if (leftEyePoint != null) {
      leftEyePosition = leftEyePoint.imagePosition;
    }
    if (rightEyePoint != null) {
      rightEyePosition = rightEyePoint.imagePosition;
    }
    if (leftEyePosition != null && rightEyePosition != null) {
      noseBridgeStart = Offset(
        (leftEyePosition.dx + rightEyePosition.dx) / 2,
        (leftEyePosition.dy + rightEyePosition.dy) / 2,
      );
    }

    Offset? upperLipCenter;
    if (noseBridgeEnd != null && faceRect != null) {
      final verticalLimit = faceRect.height * 0.35;
      final horizontalLimit = faceRect.width * 0.45;
      final candidates = <_PosePoint>[];
      for (final pt in points) {
        if (pt == null || pt.confidence < _candidateConfidence) continue;
        final pos = pt.imagePosition;
        if (pos.dy <= noseBridgeEnd.dy) continue;
        if (pos.dy > noseBridgeEnd.dy + verticalLimit) continue;
        if ((pos.dx - noseBridgeEnd.dx).abs() > horizontalLimit) continue;
        candidates.add(pt);
      }

      if (candidates.isNotEmpty) {
        candidates.sort(
          (a, b) => a.imagePosition.dx.compareTo(b.imagePosition.dx),
        );
        final left = candidates.first.imagePosition;
        final right = candidates.last.imagePosition;
        if (candidates.length >= 2 &&
            (right.dx - left.dx).abs() > faceRect.width * 0.05) {
          upperLipCenter = Offset(
            (left.dx + right.dx) / 2,
            (left.dy + right.dy) / 2,
          );
        } else {
          final avgX =
              candidates.fold<double>(0, (sum, e) => sum + e.imagePosition.dx) /
              candidates.length;
          final avgY =
              candidates.fold<double>(0, (sum, e) => sum + e.imagePosition.dy) /
              candidates.length;
          upperLipCenter = Offset(avgX, avgY);
        }
      }
    }

    return _PoseLandmarks(
      faceRect: faceRect,
      noseBridgeStart: noseBridgeStart,
      noseBridgeEnd: noseBridgeEnd,
      upperLipCenter: upperLipCenter,
      leftEye: leftEyePosition,
      rightEye: rightEyePosition,
      nose: nose,
    );
  }

  _PosePoint? _mapPosePoint({
    required YOLOResult detection,
    required int index,
    required _LetterboxTransform transform,
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

    final mapped = _convertPosePoint(
      detection: detection,
      point: keypoints[index],
      transform: transform,
    );
    if (mapped == null) return null;

    final imageClamped = Offset(
      mapped.dx.clamp(0.0, transform.srcWidth),
      mapped.dy.clamp(0.0, transform.srcHeight),
    );
    return _PosePoint(imagePosition: imageClamped, confidence: confidence);
  }

  Offset? _convertPosePoint({
    required YOLOResult detection,
    required Point point,
    required _LetterboxTransform transform,
  }) {
    final looksNormalized =
        point.x >= 0.0 && point.x <= 1.0 && point.y >= 0.0 && point.y <= 1.0;
    if (looksNormalized && !detection.boundingBox.isEmpty) {
      final bboxImage = transform.viewRectToImage(detection.boundingBox);
      return Offset(
        bboxImage.left + point.x * bboxImage.width,
        bboxImage.top + point.y * bboxImage.height,
      );
    }

    if (point.x.isFinite && point.y.isFinite) {
      final viewPoint = Offset(point.x.toDouble(), point.y.toDouble());
      final imagePoint = transform.viewToImage(viewPoint);
      return Offset(
        imagePoint.dx.clamp(0.0, transform.srcWidth),
        imagePoint.dy.clamp(0.0, transform.srcHeight),
      );
    }
    return null;
  }

  _PosePoint? _pointWithThreshold(
    List<_PosePoint?> list,
    int index,
    double threshold,
  ) {
    if (index < 0 || index >= list.length) return null;
    final pt = list[index];
    if (pt == null) return null;
    if (pt.confidence < threshold) return null;
    if (!pt.imagePosition.dx.isFinite || !pt.imagePosition.dy.isFinite) {
      return null;
    }
    return pt;
  }

  @override
  bool shouldRepaint(covariant _PoseOverlayPainter oldDelegate) {
    return oldDelegate.mustacheImage != mustacheImage ||
        oldDelegate.mustacheAlpha != mustacheAlpha ||
        oldDelegate.detections != detections ||
        oldDelegate.poseDetections != poseDetections ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical ||
        oldDelegate.sourceWidth != sourceWidth ||
        oldDelegate.sourceHeight != sourceHeight ||
        oldDelegate.debugPose != debugPose;
  }
}

class _LetterboxTransform {
  const _LetterboxTransform({
    required this.srcWidth,
    required this.srcHeight,
    required this.scale,
    required this.dx,
    required this.dy,
  });

  final double srcWidth;
  final double srcHeight;
  final double scale;
  final double dx;
  final double dy;

  Offset imageToView(Offset imagePoint) {
    return Offset(dx + imagePoint.dx * scale, dy + imagePoint.dy * scale);
  }

  Offset viewToImage(Offset viewPoint) {
    return Offset((viewPoint.dx - dx) / scale, (viewPoint.dy - dy) / scale);
  }

  Rect viewRectToImage(Rect rect) {
    return Rect.fromLTRB(
      (rect.left - dx) / scale,
      (rect.top - dy) / scale,
      (rect.right - dx) / scale,
      (rect.bottom - dy) / scale,
    );
  }

  static _LetterboxTransform fromView({
    required Size viewSize,
    required double? sourceWidth,
    required double? sourceHeight,
  }) {
    final srcW = (sourceWidth ?? viewSize.width)
        .clamp(1.0, double.infinity)
        .toDouble();
    final srcH = (sourceHeight ?? viewSize.height)
        .clamp(1.0, double.infinity)
        .toDouble();
    final scale = math.max(viewSize.width / srcW, viewSize.height / srcH);
    final scaledW = srcW * scale;
    final scaledH = srcH * scale;
    final dx = (viewSize.width - scaledW) / 2.0;
    final dy = (viewSize.height - scaledH) / 2.0;
    return _LetterboxTransform(
      srcWidth: srcW,
      srcHeight: srcH,
      scale: scale,
      dx: dx,
      dy: dy,
    );
  }
}

class _PoseLandmarks {
  const _PoseLandmarks({
    required this.faceRect,
    this.noseBridgeStart,
    this.noseBridgeEnd,
    this.upperLipCenter,
    this.leftEye,
    this.rightEye,
    this.nose,
  });

  final Rect? faceRect;
  final Offset? noseBridgeStart;
  final Offset? noseBridgeEnd;
  final Offset? upperLipCenter;
  final Offset? leftEye;
  final Offset? rightEye;
  final Offset? nose;
}

class _PosePoint {
  const _PosePoint({required this.imagePosition, required this.confidence});

  final Offset imagePosition;
  final double confidence;
}
