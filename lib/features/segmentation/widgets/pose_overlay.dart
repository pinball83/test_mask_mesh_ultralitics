import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

/// Pose-driven overlay that draws facial debug markers and a mustache sprite.
/// The name is kept for backwards compatibility with the previous shader-based implementation.
class PoseOverlay extends StatefulWidget {
  const PoseOverlay({
    super.key,
    required this.poseDetections,
    required this.flipHorizontal,
    required this.flipVertical,
    this.mustacheAsset = 'assets/images/must.png',
    this.mustacheAlpha = 1.0,
    this.debugPose = false,
    this.showMustache = true,
  });

  final List<YOLOResult> poseDetections;
  final bool flipHorizontal;
  final bool flipVertical;
  final String mustacheAsset;
  final double mustacheAlpha;
  final bool debugPose;
  final bool showMustache;

  @override
  State<PoseOverlay> createState() => _PoseOverlayState();
}

class _PoseOverlayState extends State<PoseOverlay> {
  ui.Image? _mustacheImage;
  ImageStream? _mustacheStream;
  ImageStreamListener? _mustacheListener;

  @override
  void initState() {
    super.initState();
    _resolveMustache();
  }

  @override
  void didUpdateWidget(covariant PoseOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.mustacheAsset != widget.mustacheAsset) {
      _resolveMustache();
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

  @override
  Widget build(BuildContext context) {
    if (widget.poseDetections.isEmpty && !widget.debugPose) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _PoseOverlayPainter(
          poseDetections: widget.poseDetections,
          mustacheImage: _mustacheImage,
          mustacheAlpha: widget.mustacheAlpha.clamp(0.0, 1.0).toDouble(),
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
          debugPose: widget.debugPose,
          showMustache: widget.showMustache,
        ),
      ),
    );
  }
}

class _PoseOverlayPainter extends CustomPainter {
  _PoseOverlayPainter({
    required this.poseDetections,
    required this.mustacheImage,
    required this.mustacheAlpha,
    required this.flipHorizontal,
    required this.flipVertical,
    required this.debugPose,
    required this.showMustache,
  });

  final List<YOLOResult> poseDetections;
  final ui.Image? mustacheImage;
  final double mustacheAlpha;
  final bool flipHorizontal;
  final bool flipVertical;
  final bool debugPose;
  final bool showMustache;

  // Mustache sizing constants
  static const double _minMustacheWidth = 24.0;
  static const double _maxMustacheWidthFactor = 0.8;
  static const double _fallbackWidthFactor = 0.5;
  static const double _fallbackWidthFactorUnsafe = 0.3;
  static const double _eyeWidthMultiplier = 1.8;

  // Confidence thresholds
  static const double _noseConfidence = 0.15;
  static const double _eyeConfidence = 0.25;
  static const double _candidateConfidence = 0.05;

  // Mustache positioning correction (in view space, scaled)
  // Small adjustment to center mustache mask on nose. Set to 0.0 if not needed.
  static const double _mustacheCorrectionFactor = 0.0;

  // Upper lip detection limits (relative to face rect)
  static const double _upperLipVerticalLimitFactor = 0.35;
  static const double _upperLipHorizontalLimitFactor = 0.45;
  static const double _upperLipMinWidthFactor = 0.05;

  // Pose selection scoring
  static const double _poseScoreAreaWeight = 0.5;
  static const double _poseScoreDistancePenalty = 2.0;

  // Alpha threshold for color filter
  static const double _alphaThreshold = 0.99;

  // Debug paint constants
  static const double _debugFaceStrokeWidth = 1.2;
  static const double _debugEyeOutlineRadius = 7.0;
  static const double _debugEyeFillRadius = 3.0;
  static const double _debugEyeStrokeWidth = 2.0;
  static const double _debugNoseRadius = 5.0;
  static const double _debugNoseCrossSize = 6.0;
  static const double _debugNoseStrokeWidth = 1.1;
  static const double _debugLipRadius = 4.0;
  static const double _debugHelperStrokeWidth = 1.4;

  // Debug colors
  static const Color _debugFaceColor = Colors.limeAccent;
  static const double _debugFaceAlpha = 0.6;
  static const Color _debugEyeColor = Colors.cyanAccent;
  static const double _debugEyeAlpha = 0.35;
  static const Color _debugHelperColor = Colors.amberAccent;
  static const double _debugHelperAlpha = 0.8;
  static const Color _debugNoseColor = Colors.deepOrangeAccent;
  static const Color _debugLipColor = Colors.purpleAccent;
  static const double _debugLipAlpha = 0.7;

  // Debug logging throttle (milliseconds)
  static const int _debugLogThrottleMs = 500;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;

    final poseDetection = _pickPrimaryPose(poseDetections, size);

    final landmarks = poseDetection != null
        ? _extractPoseLandmarks(detection: poseDetection, viewSize: size)
        : null;

    final viewPoints = _ViewPoints(
      faceRect: landmarks?.faceRect,
      nose: landmarks?.nose,
      leftEye: landmarks?.leftEye,
      rightEye: landmarks?.rightEye,
      noseBridgeStart: landmarks?.noseBridgeStart,
      noseBridgeEnd: landmarks?.noseBridgeEnd ?? landmarks?.nose,
      upperLip: landmarks?.upperLipCenter,
    );

    if (showMustache) {
      _drawMustache(canvas, size, viewPoints: viewPoints);
    }

    if (debugPose) {
       _logDebugPoseMapping(
        size: size,
        pose: poseDetection,
        viewPoints: viewPoints,
      );
      _drawDebug(canvas, viewPoints);
      _drawRawDebug(canvas, size, poseDetection);
    }
  }

  void _drawRawDebug(Canvas canvas, Size size, YOLOResult? pose) {
    if (pose == null) return;

    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final pointPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.fill;

    // Draw raw bounding box
    final bbox = pose.boundingBox;
    if (!bbox.isEmpty) {
      var left = bbox.left;
      var right = bbox.right;
      var top = bbox.top;
      var bottom = bbox.bottom;

      // If normalized, scale to size
      if (left >= 0 && left <= 1.0 && right >= 0 && right <= 1.0) {
        left *= size.width;
        right *= size.width;
        top *= size.height;
        bottom *= size.height;
      }

      if (flipHorizontal) {
        final originalLeft = left;
        left = size.width - right;
        right = size.width - originalLeft;
      }

      final rect = Rect.fromLTRB(left, top, right, bottom);
      canvas.drawRect(rect, paint);
    }

    // Draw raw keypoints
    final keypoints = pose.keypoints;
    if (keypoints != null) {
      for (final kp in keypoints) {
        double x = kp.x;
        double y = kp.y;

        // If normalized, scale to size
        if (x >= 0 && x <= 1.0 && y >= 0 && y <= 1.0) {
          x *= size.width;
          y *= size.height;
        }

        if (flipHorizontal) {
          x = size.width - x;
        }

        canvas.drawCircle(Offset(x, y), 4.0, pointPaint);
      }
    }
  }

  void _drawMustache(
    Canvas canvas,
    Size size, {
    required _ViewPoints viewPoints,
  }) {
    final image = mustacheImage;
    if (image == null) return;

    final width = _resolveMustacheWidth(viewPoints: viewPoints, viewSize: size);
    final height = width * (image.height / image.width);

    final rotation = _calculateRotation(
      viewPoints.leftEye,
      viewPoints.rightEye,
    );
    final anchor = _selectAnchor(viewPoints, size);
    final center = _calculateMustacheCenter(anchor: anchor);

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
      colorFilter: mustacheAlpha < _alphaThreshold
          ? ColorFilter.mode(
              Colors.white.withValues(alpha: mustacheAlpha),
              BlendMode.modulate,
            )
          : null,
    );

    canvas.restore();
  }

  double _calculateRotation(Offset? leftEye, Offset? rightEye) {
    if (leftEye == null || rightEye == null) return 0.0;
    return math.atan2(rightEye.dy - leftEye.dy, rightEye.dx - leftEye.dx);
  }

  Offset _selectAnchor(_ViewPoints viewPoints, Size size) {
    final nose = viewPoints.nose;
    final upperLip = viewPoints.upperLip;

    // Prefer anchoring the mask between the nose tip and the upper lip so the
    // sprite follows the actual mouth region instead of the face center.
    if (nose != null && upperLip != null) {
      // Blend slightly closer to the nose so the mask sits just above the lip.
      const t = 0.6;
      return Offset(
        upperLip.dx + (nose.dx - upperLip.dx) * t,
        upperLip.dy + (nose.dy - upperLip.dy) * t,
      );
    }

    // Fallbacks preserve previous behaviour.
    return upperLip ??
        nose ??
        viewPoints.noseBridgeEnd ??
        viewPoints.faceRect?.center ??
        Offset(size.width / 2, size.height / 2);
  }

  Offset _calculateMustacheCenter({required Offset anchor}) {
    // Skip correction if it's zero
    if (_mustacheCorrectionFactor == 0.0) {
      return anchor;
    }

    final correction = Offset(
      _mustacheCorrectionFactor,
      _mustacheCorrectionFactor,
    );
    return Offset(anchor.dx - correction.dx, anchor.dy - correction.dy);
  }

  void _drawDebug(Canvas canvas, _ViewPoints points) {
    final facePaint = Paint()
      ..color = _debugFaceColor.withValues(alpha: _debugFaceAlpha)
      ..style = PaintingStyle.stroke
      ..strokeWidth = _debugFaceStrokeWidth;

    final eyeOutline = Paint()
      ..color = _debugEyeColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = _debugEyeStrokeWidth;

    final eyeFill = Paint()
      ..color = _debugEyeColor.withValues(alpha: _debugEyeAlpha)
      ..style = PaintingStyle.fill;

    final helperLine = Paint()
      ..color = _debugHelperColor.withValues(alpha: _debugHelperAlpha)
      ..strokeWidth = _debugHelperStrokeWidth;

    final noseFill = Paint()
      ..color = _debugNoseColor
      ..style = PaintingStyle.fill;

    final noseCross = Paint()
      ..color = _debugNoseColor
      ..strokeWidth = _debugNoseStrokeWidth;

    final lipPaint = Paint()
      ..color = _debugLipColor.withValues(alpha: _debugLipAlpha)
      ..style = PaintingStyle.fill;

    if (points.faceRect != null) {
      canvas.drawRect(points.faceRect!, facePaint);
    }

    if (points.leftEye != null) {
      canvas.drawCircle(points.leftEye!, _debugEyeOutlineRadius, eyeOutline);
      canvas.drawCircle(points.leftEye!, _debugEyeFillRadius, eyeFill);
    }
    if (points.rightEye != null) {
      canvas.drawCircle(points.rightEye!, _debugEyeOutlineRadius, eyeOutline);
      canvas.drawCircle(points.rightEye!, _debugEyeFillRadius, eyeFill);
    }
    if (points.leftEye != null && points.rightEye != null) {
      canvas.drawLine(points.leftEye!, points.rightEye!, helperLine);
    }

    if (points.nose != null) {
      canvas.drawCircle(points.nose!, _debugNoseRadius, noseFill);
      final nose = points.nose!;
      canvas.drawLine(
        Offset(nose.dx - _debugNoseCrossSize, nose.dy),
        Offset(nose.dx + _debugNoseCrossSize, nose.dy),
        noseCross,
      );
      canvas.drawLine(
        Offset(nose.dx, nose.dy - _debugNoseCrossSize),
        Offset(nose.dx, nose.dy + _debugNoseCrossSize),
        noseCross,
      );
    }

    if (points.noseBridgeStart != null && points.noseBridgeEnd != null) {
      canvas.drawLine(
        points.noseBridgeStart!,
        points.noseBridgeEnd!,
        helperLine,
      );
    }

    if (points.upperLip != null) {
      canvas.drawCircle(points.upperLip!, _debugLipRadius, lipPaint);
    }
  }

  double _resolveMustacheWidth({
    required _ViewPoints viewPoints,
    required Size viewSize,
  }) {
    double width;
    if (viewPoints.leftEye != null && viewPoints.rightEye != null) {
      final eyeSpan = (viewPoints.leftEye! - viewPoints.rightEye!).distance;
      width = eyeSpan * _eyeWidthMultiplier;
    } else if (viewPoints.faceRect != null) {
      width = viewPoints.faceRect!.width * _fallbackWidthFactor;
    } else {
      width = viewSize.width * _fallbackWidthFactor;
    }

    if (!width.isFinite || width <= 0) {
      width = viewSize.width * _fallbackWidthFactorUnsafe;
    }

    return width.clamp(
      _minMustacheWidth,
      viewSize.width * _maxMustacheWidthFactor,
    );
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
      final center = Offset(box.left + box.width / 2, box.top + box.height / 2);
      final viewCenter = Offset(size.width / 2, size.height / 2);
      final dx = center.dx - viewCenter.dx;
      final dy = center.dy - viewCenter.dy;
      final distSquared = dx * dx + dy * dy;
      final score =
          area * _poseScoreAreaWeight - distSquared * _poseScoreDistancePenalty;
      if (score > bestScore) {
        bestScore = score;
        best = d;
      }
    }
    return best;
  }

  static DateTime? _lastLog;
  void _logDebugPoseMapping({
    required Size size,
    required YOLOResult? pose,
    required _ViewPoints viewPoints,
  }) {
    final now = DateTime.now();
    if (_lastLog != null &&
        now.difference(_lastLog!).inMilliseconds < _debugLogThrottleMs) {
      return;
    }
    _lastLog = now;

    final bb = pose?.boundingBox;
    final nb = pose?.normalizedBox;
    final kp = pose?.keypoints;
    final kc = pose?.keypointConfidences;
    final noseRaw = (kp != null && kp.isNotEmpty) ? kp[0] : null;
    final noseConf = (kc != null && kc.isNotEmpty) ? kc[0] : null;

    debugPrint(
      [
        'POSE DEBUG — view=${size.width.toStringAsFixed(0)}x${size.height.toStringAsFixed(0)}',
        if (bb != null)
          'poseBB(view)=L${bb.left.toStringAsFixed(1)} T${bb.top.toStringAsFixed(1)} '
              'W${bb.width.toStringAsFixed(1)} H${bb.height.toStringAsFixed(1)}',
        if (nb != null)
          'poseNB(norm)=L${nb.left.toStringAsFixed(3)} T${nb.top.toStringAsFixed(3)} '
              'R${nb.right.toStringAsFixed(3)} B${nb.bottom.toStringAsFixed(3)}',
        if (noseRaw != null)
          'noseRaw=(${noseRaw.x.toStringAsFixed(3)}, ${noseRaw.y.toStringAsFixed(3)}) '
              '${noseConf != null ? 'conf=${noseConf.toStringAsFixed(2)}' : ''}',
        if (viewPoints.nose != null)
          'noseView=${viewPoints.nose!.dx.toStringAsFixed(1)}, ${viewPoints.nose!.dy.toStringAsFixed(1)}',
        if (viewPoints.faceRect != null)
          'faceRectView=L${viewPoints.faceRect!.left.toStringAsFixed(1)} '
              'T${viewPoints.faceRect!.top.toStringAsFixed(1)} '
              'W${viewPoints.faceRect!.width.toStringAsFixed(1)} '
              'H${viewPoints.faceRect!.height.toStringAsFixed(1)}',
        'flips: h=$flipHorizontal v=$flipVertical',
      ].join(' | '),
    );
  }

  _PoseLandmarks _extractPoseLandmarks({
    required YOLOResult detection,
    required Size viewSize,
  }) {
    Rect? faceRect;
    if (!detection.boundingBox.isEmpty) {
      var left = detection.boundingBox.left;
      var right = detection.boundingBox.right;
      var top = detection.boundingBox.top;
      var bottom = detection.boundingBox.bottom;

      // If normalized, scale to size
      if (left >= 0 && left <= 1.0 && right >= 0 && right <= 1.0) {
        left *= viewSize.width;
        right *= viewSize.width;
        top *= viewSize.height;
        bottom *= viewSize.height;
      }

      if (flipHorizontal) {
        final originalLeft = left;
        left = viewSize.width - right;
        right = viewSize.width - originalLeft;
      }

      faceRect = Rect.fromLTRB(left, top, right, bottom);
    }

    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.isEmpty) {
      return _PoseLandmarks(faceRect: faceRect);
    }

    final points = List<_PosePoint?>.generate(
      keypoints.length,
      (index) =>
          _mapPosePoint(detection: detection, index: index, viewSize: viewSize),
      growable: false,
    );

    final nosePoint = _pointWithThreshold(points, 0, _noseConfidence);
    final leftEyePoint = _pointWithThreshold(points, 1, _eyeConfidence);
    final rightEyePoint = _pointWithThreshold(points, 2, _eyeConfidence);

    final nose = nosePoint?.imagePosition;
    final noseBridgeEnd = nose;
    final leftEye = leftEyePoint?.imagePosition;
    final rightEye = rightEyePoint?.imagePosition;

    final noseBridgeStart = (leftEye != null && rightEye != null)
        ? Offset((leftEye.dx + rightEye.dx) / 2, (leftEye.dy + rightEye.dy) / 2)
        : null;

    final upperLipCenter = _findUpperLipCenter(
      points: points,
      noseBridgeEnd: noseBridgeEnd,
      faceRect: faceRect,
    );

    return _PoseLandmarks(
      faceRect: faceRect,
      noseBridgeStart: noseBridgeStart,
      noseBridgeEnd: noseBridgeEnd,
      upperLipCenter: upperLipCenter,
      leftEye: leftEye,
      rightEye: rightEye,
      nose: nose,
    );
  }

  Offset? _findUpperLipCenter({
    required List<_PosePoint?> points,
    required Offset? noseBridgeEnd,
    required Rect? faceRect,
  }) {
    if (noseBridgeEnd == null || faceRect == null) return null;

    final verticalLimit = faceRect.height * _upperLipVerticalLimitFactor;
    final horizontalLimit = faceRect.width * _upperLipHorizontalLimitFactor;
    final minWidth = faceRect.width * _upperLipMinWidthFactor;

    final candidates = <_PosePoint>[];
    for (final pt in points) {
      if (pt == null || pt.confidence < _candidateConfidence) continue;
      final pos = pt.imagePosition;
      if (pos.dy <= noseBridgeEnd.dy) continue;
      if (pos.dy > noseBridgeEnd.dy + verticalLimit) continue;
      if ((pos.dx - noseBridgeEnd.dx).abs() > horizontalLimit) continue;
      candidates.add(pt);
    }

    if (candidates.isEmpty) return null;

    candidates.sort((a, b) => a.imagePosition.dx.compareTo(b.imagePosition.dx));
    final left = candidates.first.imagePosition;
    final right = candidates.last.imagePosition;

    if (candidates.length >= 2 && (right.dx - left.dx).abs() > minWidth) {
      return Offset((left.dx + right.dx) / 2, (left.dy + right.dy) / 2);
    }

    // Average of all candidates
    final avgX =
        candidates.fold<double>(0, (sum, e) => sum + e.imagePosition.dx) /
        candidates.length;
    final avgY =
        candidates.fold<double>(0, (sum, e) => sum + e.imagePosition.dy) /
        candidates.length;
    return Offset(avgX, avgY);
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

    double x = point.x;
    double y = point.y;

    // If normalized, scale to size
    if (x >= 0 && x <= 1.0 && y >= 0 && y <= 1.0) {
      x *= viewSize.width;
      y *= viewSize.height;
    }

    if (flipHorizontal) {
      x = viewSize.width - x;
    }

    final imagePosition = Offset(x, y);

    return _PosePoint(imagePosition: imagePosition, confidence: confidence);
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
        oldDelegate.poseDetections != poseDetections ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical ||
        oldDelegate.debugPose != debugPose;
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

class _ViewPoints {
  const _ViewPoints({
    this.faceRect,
    this.nose,
    this.leftEye,
    this.rightEye,
    this.noseBridgeStart,
    this.noseBridgeEnd,
    this.upperLip,
  });

  final Rect? faceRect;
  final Offset? nose;
  final Offset? leftEye;
  final Offset? rightEye;
  final Offset? noseBridgeStart;
  final Offset? noseBridgeEnd;
  final Offset? upperLip;
}
