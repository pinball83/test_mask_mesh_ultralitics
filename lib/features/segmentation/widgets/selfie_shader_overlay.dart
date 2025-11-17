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
    this.mustacheAsset = 'assets/images/must.png',
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
      final size = _estimateSourceSize(widget.detections);
      _srcW = size?.width;
      _srcH = size?.height;
    });
  }

  Size? _estimateSourceSize(List<YOLOResult> list) {
    // Prefer segmentation detections (with non-empty mask) for estimating
    // original source size, since their boxes are the reference used by
    // the background overlay.
    final ordered = [
      ...list.where((d) => (d.mask?.isNotEmpty ?? false)),
      ...list.where((d) => !(d.mask?.isNotEmpty ?? false)),
    ];

    double? width;
    double? height;

    for (final d in ordered) {
      final nb = d.normalizedBox;
      final bb = d.boundingBox;

      if (width == null && nb.width > 0 && bb.width > 0 && nb.width.isFinite) {
        final candidate = bb.width / nb.width;
        if (candidate.isFinite && candidate > 0) {
          width = candidate;
        }
      }

      if (height == null &&
          nb.height > 0 &&
          bb.height > 0 &&
          nb.height.isFinite) {
        final candidate = bb.height / nb.height;
        if (candidate.isFinite && candidate > 0) {
          height = candidate;
        }
      }

      if (width != null && height != null) break;
    }

    if (width != null && height != null) {
      return Size(width, height);
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

  // Keypoint position correction (in view space, pixels)
  // This correction compensates for offset between detected keypoints and actual
  // facial feature positions. Set to Offset.zero if keypoints are already accurate.
  static const Offset _keypointCorrectionView = Offset.zero;

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

    final transform = _LetterboxTransform.fromView(
      viewSize: size,
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
    );

    final poseDetection = _pickPrimaryPose(poseDetections, size);
    final maskDetection = _pickPrimaryMask(detections);

    final landmarks = poseDetection != null
        ? _extractPoseLandmarks(
            detection: poseDetection,
            transform: transform,
            actualViewSize: size,
          )
        : null;

    final faceRectImage =
        landmarks?.faceRect ??
        (maskDetection != null
            ? transform.viewRectToImage(maskDetection.boundingBox)
            : null);

    final (bool autoFlipH, bool autoFlipV) = _detectAutoMirror(
      size: size,
      transform: transform,
      reference: maskDetection ?? poseDetection,
    );
    final effectiveFlipH = flipHorizontal ^ autoFlipH;
    final effectiveFlipV = flipVertical ^ autoFlipV;

    final viewPoints = _ViewPoints(
      faceRect: _mapRectToCanvas(
        faceRectImage,
        transform,
        size,
        effectiveFlipH,
        effectiveFlipV,
      ),
      nose: _mapPointToCanvas(
        landmarks?.nose,
        transform,
        size,
        effectiveFlipH,
        effectiveFlipV,
      ),
      leftEye: _mapPointToCanvas(
        landmarks?.leftEye,
        transform,
        size,
        effectiveFlipH,
        effectiveFlipV,
      ),
      rightEye: _mapPointToCanvas(
        landmarks?.rightEye,
        transform,
        size,
        effectiveFlipH,
        effectiveFlipV,
      ),
      noseBridgeStart: _mapPointToCanvas(
        landmarks?.noseBridgeStart,
        transform,
        size,
        effectiveFlipH,
        effectiveFlipV,
      ),
      noseBridgeEnd: _mapPointToCanvas(
        landmarks?.noseBridgeEnd ?? landmarks?.nose,
        transform,
        size,
        effectiveFlipH,
        effectiveFlipV,
      ),
      upperLip: _mapPointToCanvas(
        landmarks?.upperLipCenter,
        transform,
        size,
        effectiveFlipH,
        effectiveFlipV,
      ),
    );

    if (debugPose) {
      _logDebugPoseMapping(
        size: size,
        transform: transform,
        pose: poseDetection,
        mask: maskDetection,
        viewPoints: viewPoints,
        effectiveFlipH: effectiveFlipH,
        effectiveFlipV: effectiveFlipV,
      );
    }

    if (showMustache) {
      _drawMustache(canvas, size, transform: transform, viewPoints: viewPoints);
    }

    if (debugPose) {
      _drawDebug(canvas, viewPoints);
    }
  }

  void _drawMustache(
    Canvas canvas,
    Size size, {
    required _LetterboxTransform transform,
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
    final center = _calculateMustacheCenter(
      anchor: anchor,
      transform: transform,
    );

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
    return viewPoints.nose ??
        viewPoints.noseBridgeEnd ??
        viewPoints.faceRect?.center ??
        Offset(size.width / 2, size.height / 2);
  }

  Offset _calculateMustacheCenter({
    required Offset anchor,
    required _LetterboxTransform transform,
  }) {
    // Skip correction if it's zero
    if (_mustacheCorrectionFactor == 0.0) {
      return anchor;
    }

    final correction = Offset(
      _mustacheCorrectionFactor * transform.scale,
      _mustacheCorrectionFactor * transform.scale,
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
    required _LetterboxTransform transform,
    required YOLOResult? pose,
    required YOLOResult? mask,
    required _ViewPoints viewPoints,
    required bool effectiveFlipH,
    required bool effectiveFlipV,
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
        'src=${transform.srcWidth.toStringAsFixed(1)}x${transform.srcHeight.toStringAsFixed(1)} '
            'scale=${transform.scale.toStringAsFixed(4)} '
            'dx=${transform.dx.toStringAsFixed(1)} dy=${transform.dy.toStringAsFixed(1)}',
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
        if (mask?.boundingBox != null)
          'segBB(view)=L${mask!.boundingBox.left.toStringAsFixed(1)} '
              'T${mask.boundingBox.top.toStringAsFixed(1)} '
              'W${mask.boundingBox.width.toStringAsFixed(1)} '
              'H${mask.boundingBox.height.toStringAsFixed(1)}',
        'flips: h=$flipHorizontal v=$flipVertical eff(h:$effectiveFlipH v:$effectiveFlipV)',
      ].join(' | '),
    );
  }

  Rect? _mapRectToCanvas(
    Rect? rect,
    _LetterboxTransform transform,
    Size size,
    bool effectiveFlipH,
    bool effectiveFlipV,
  ) {
    if (rect == null || rect.isEmpty) return null;
    final topLeft = _mapPointToCanvas(
      rect.topLeft,
      transform,
      size,
      effectiveFlipH,
      effectiveFlipV,
    );
    final bottomRight = _mapPointToCanvas(
      rect.bottomRight,
      transform,
      size,
      effectiveFlipH,
      effectiveFlipV,
    );
    if (topLeft == null || bottomRight == null) return null;
    final left = math.min(topLeft.dx, bottomRight.dx);
    final right = math.max(topLeft.dx, bottomRight.dx);
    final top = math.min(topLeft.dy, bottomRight.dy);
    final bottom = math.max(topLeft.dy, bottomRight.dy);
    if (right <= left || bottom <= top) return null;
    return Rect.fromLTRB(left, top, right, bottom);
  }

  Offset? _applyKeypointCorrection(
    Offset? imagePosition,
    _LetterboxTransform transform,
  ) {
    if (imagePosition == null) return null;

    // Skip correction if it's zero to avoid unnecessary transformations
    if (_keypointCorrectionView.dx == 0.0 &&
        _keypointCorrectionView.dy == 0.0) {
      return imagePosition;
    }

    // Convert to view space, apply correction, then convert back to image space
    final viewPoint = transform.imageToView(imagePosition);
    final correctedView = Offset(
      viewPoint.dx - _keypointCorrectionView.dx,
      viewPoint.dy - _keypointCorrectionView.dy,
    );
    return transform.viewToImage(correctedView);
  }

  Offset? _mapPointToCanvas(
    Offset? imagePoint,
    _LetterboxTransform transform,
    Size size,
    bool effectiveFlipH,
    bool effectiveFlipV,
  ) {
    if (imagePoint == null) return null;
    // Apply flips in IMAGE space to match mask overlay semantics, then map.
    final flipped = Offset(
      effectiveFlipH ? (transform.srcWidth - imagePoint.dx) : imagePoint.dx,
      effectiveFlipV ? (transform.srcHeight - imagePoint.dy) : imagePoint.dy,
    );
    final viewPoint = transform.imageToView(flipped);
    if (!viewPoint.dx.isFinite || !viewPoint.dy.isFinite) return null;
    return Offset(
      viewPoint.dx.clamp(0.0, size.width),
      viewPoint.dy.clamp(0.0, size.height),
    );
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
    required Size actualViewSize,
  }) {
    Rect? faceRect;
    if (!detection.boundingBox.isEmpty) {
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
        viewSize: actualViewSize,
      ),
      growable: false,
    );

    final nosePoint = _pointWithThreshold(points, 0, _noseConfidence);
    final leftEyePoint = _pointWithThreshold(points, 1, _eyeConfidence);
    final rightEyePoint = _pointWithThreshold(points, 2, _eyeConfidence);

    // Apply correction in view space to be independent of source image size
    final nose = _applyKeypointCorrection(nosePoint?.imagePosition, transform);
    final noseBridgeEnd = nose;

    final leftEye = _applyKeypointCorrection(
      leftEyePoint?.imagePosition,
      transform,
    );

    final rightEye = _applyKeypointCorrection(
      rightEyePoint?.imagePosition,
      transform,
    );

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

  (bool, bool) _detectAutoMirror({
    required Size size,
    required _LetterboxTransform transform,
    required YOLOResult? reference,
  }) {
    final ref = reference;
    if (ref == null) return (false, false);
    if (ref.normalizedBox.isEmpty || ref.boundingBox.isEmpty) return (false, false);

    Rect _mapNormToView(Rect n) {
      final l = transform.dx + (n.left.clamp(0.0, 1.0) * transform.srcWidth) * transform.scale;
      final t = transform.dy + (n.top.clamp(0.0, 1.0) * transform.srcHeight) * transform.scale;
      final r = transform.dx + (n.right.clamp(0.0, 1.0) * transform.srcWidth) * transform.scale;
      final b = transform.dy + (n.bottom.clamp(0.0, 1.0) * transform.srcHeight) * transform.scale;
      final left = l < r ? l : r;
      final right = l < r ? r : l;
      final top = t < b ? t : b;
      final bottom = t < b ? b : t;
      return Rect.fromLTRB(left, top, right, bottom);
    }

    Rect _mirrorH(Rect r) => Rect.fromLTRB(
          size.width - r.right,
          r.top,
          size.width - r.left,
          r.bottom,
        );
    Rect _mirrorV(Rect r) => Rect.fromLTRB(
          r.left,
          size.height - r.bottom,
          r.right,
          size.height - r.top,
        );

    double _l1(Rect a, Rect b) {
      return (a.left - b.left).abs() +
          (a.top - b.top).abs() +
          (a.right - b.right).abs() +
          (a.bottom - b.bottom).abs();
    }

    final predicted = _mapNormToView(ref.normalizedBox);
    final actual = ref.boundingBox;

    final h = _mirrorH(predicted);
    final v = _mirrorV(predicted);
    final hv = _mirrorV(h);

    final dBase = _l1(predicted, actual);
    final dH = _l1(h, actual);
    final dV = _l1(v, actual);
    final dHV = _l1(hv, actual);

    double min = dBase;
    var flipH = false;
    var flipV = false;
    if (dH + 0.5 < min) {
      min = dH;
      flipH = true;
      flipV = false;
    }
    if (dV + 0.5 < min) {
      min = dV;
      flipH = false;
      flipV = true;
    }
    if (dHV + 0.5 < min) {
      min = dHV;
      flipH = true;
      flipV = true;
    }
    return (flipH, flipV);
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
    required _LetterboxTransform transform,
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

    final mapped = _convertPosePoint(
      detection: detection,
      point: keypoints[index],
      transform: transform,
      viewSize: viewSize,
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
    required Size viewSize,
  }) {
    if (!point.x.isFinite || !point.y.isFinite) return null;

    // Check if normalized (0-1), treat as relative to bounding box
    final isNormalized =
        point.x >= 0.0 && point.x <= 1.0 && point.y >= 0.0 && point.y <= 1.0;

    if (isNormalized && !detection.boundingBox.isEmpty) {
      final bboxImage = transform.viewRectToImage(detection.boundingBox);
      return Offset(
        bboxImage.left + point.x * bboxImage.width,
        bboxImage.top + point.y * bboxImage.height,
      );
    }

    // Check if coordinates are in view space range
    // (YOLO pose keypoints are typically in view space, not image space)
    if (point.x >= 0 &&
        point.x <= viewSize.width &&
        point.y >= 0 &&
        point.y <= viewSize.height) {
      final imagePoint = transform.viewToImage(
        Offset(point.x.toDouble(), point.y.toDouble()),
      );
      return Offset(
        imagePoint.dx.clamp(0.0, transform.srcWidth),
        imagePoint.dy.clamp(0.0, transform.srcHeight),
      );
    }

    // Fallback: if coordinates are very large, might be in image space
    final maxExpectedCoord =
        math.max(transform.srcWidth, transform.srcHeight) * 2;
    if (point.x >= 0 &&
        point.x <= maxExpectedCoord &&
        point.y >= 0 &&
        point.y <= maxExpectedCoord) {
      return Offset(
        point.x.clamp(0.0, transform.srcWidth),
        point.y.clamp(0.0, transform.srcHeight),
      );
    }

    // Last resort: treat as view space and convert
    final imagePoint = transform.viewToImage(
      Offset(point.x.toDouble(), point.y.toDouble()),
    );
    return Offset(
      imagePoint.dx.clamp(0.0, transform.srcWidth),
      imagePoint.dy.clamp(0.0, transform.srcHeight),
    );
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
