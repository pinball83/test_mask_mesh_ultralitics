import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

/// GPU-accelerated overlay that uses `assets/shaders/selfie_shader.frag`
/// to replace background by segmentation mask and draw a mustache overlay.
class SelfieShaderOverlay extends StatefulWidget {
  const SelfieShaderOverlay({
    super.key,
    required this.detections,
    required this.poseDetections,
    required this.flipHorizontal,
    required this.flipVertical,
    this.mode = 1.0, // 0 off, 1 mask, 2 replace bg (same in shader)
    this.mustacheAlpha = 1.0,
    this.mustacheScale = 0.12,
    this.backgroundAsset = 'assets/images/bg_image.jpg',
    this.mustacheAsset = 'assets/images/mustash.png',
    this.debugPose = false,
  });

  final List<YOLOResult> detections;
  final List<YOLOResult> poseDetections;
  final bool flipHorizontal;
  final bool flipVertical;

  final double mode;
  final double mustacheAlpha;
  final double mustacheScale;
  final String backgroundAsset;
  final String mustacheAsset;
  final bool debugPose;

  @override
  State<SelfieShaderOverlay> createState() => _SelfieShaderOverlayState();
}

class _SelfieShaderOverlayState extends State<SelfieShaderOverlay> {
  ui.FragmentProgram? _program;
  ui.Image? _bgImage;
  ui.Image? _mustacheImage;
  ui.Image? _maskImage; // Combined or primary mask to pass to shader

  ImageStream? _bgStream;
  ImageStreamListener? _bgListener;
  ImageStream? _stashStream;
  ImageStreamListener? _stashListener;

  // Cache last mask source dims for uniforms.
  double? _srcW;
  double? _srcH;

  @override
  void initState() {
    super.initState();
    _loadProgram();
    _resolveImages();
    _rebuildMaskImage();
  }

  @override
  void didUpdateWidget(covariant SelfieShaderOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.backgroundAsset != widget.backgroundAsset ||
        oldWidget.mustacheAsset != widget.mustacheAsset) {
      _resolveImages();
    }
    if (!identical(oldWidget.detections, widget.detections)) {
      _rebuildMaskImage();
    }
  }

  @override
  void dispose() {
    _disposeImageStreams();
    super.dispose();
  }

  Future<void> _loadProgram() async {
    try {
      final p = await ui.FragmentProgram.fromAsset(
        'assets/shaders/selfie_shader.frag',
      );
      if (mounted) setState(() => _program = p);
    } catch (e) {
      // Silently ignore; painter will skip if null.
    }
  }

  void _resolveImages() {
    _disposeImageStreams();

    // Background
    final bgProvider = AssetImage(widget.backgroundAsset);
    final bgStream = bgProvider.resolve(ImageConfiguration.empty);
    _bgStream = bgStream;
    _bgListener = ImageStreamListener((imageInfo, _) {
      if (mounted) setState(() => _bgImage = imageInfo.image);
    });
    bgStream.addListener(_bgListener!);

    // Mustache
    final stashProvider = AssetImage(widget.mustacheAsset);
    final stashStream = stashProvider.resolve(ImageConfiguration.empty);
    _stashStream = stashStream;
    _stashListener = ImageStreamListener((imageInfo, _) {
      if (mounted) setState(() => _mustacheImage = imageInfo.image);
    });
    stashStream.addListener(_stashListener!);
  }

  void _disposeImageStreams() {
    if (_bgStream != null && _bgListener != null) {
      _bgStream!.removeListener(_bgListener!);
    }
    if (_stashStream != null && _stashListener != null) {
      _stashStream!.removeListener(_stashListener!);
    }
    _bgStream = null;
    _bgListener = null;
    _stashStream = null;
    _stashListener = null;
  }

  void _rebuildMaskImage() {
    // Pick the largest detection with a valid mask and use it as texture.
    final detection = _pickPrimary(widget.detections);
    if (detection == null ||
        detection.mask == null ||
        detection.mask!.isEmpty) {
      if (mounted) setState(() => _maskImage = null);
      return;
    }
    final mask = detection.mask!;
    _srcW = _estimateSourceWidth(widget.detections);
    _srcH = _estimateSourceHeight(widget.detections);

    _maskFrom2D(mask).then((img) {
      if (!mounted) return;
      setState(() => _maskImage = img);
    });
  }

  YOLOResult? _pickPrimary(List<YOLOResult> list) {
    YOLOResult? best;
    var bestArea = -1.0;
    for (final d in list) {
      final m = d.mask;
      final b = d.boundingBox;
      if (m == null || m.isEmpty || b.isEmpty) continue;
      final area = (b.width * b.height).abs();
      if (area > bestArea) {
        bestArea = area;
        best = d;
      }
    }
    return best;
  }

  double? _estimateSourceWidth(List<YOLOResult> list) {
    for (final d in list) {
      final nb = d.normalizedBox;
      final b = d.boundingBox;
      final w = nb.width;
      if (w > 0 && b.width > 0 && w.isFinite) {
        final candidate = b.width / w;
        if (candidate.isFinite && candidate > 0) return candidate;
      }
    }
    return null;
  }

  double? _estimateSourceHeight(List<YOLOResult> list) {
    for (final d in list) {
      final nb = d.normalizedBox;
      final b = d.boundingBox;
      final h = nb.height;
      if (h > 0 && b.height > 0 && h.isFinite) {
        final candidate = b.height / h;
        if (candidate.isFinite && candidate > 0) return candidate;
      }
    }
    return null;
  }

  Future<ui.Image> _maskFrom2D(List<List<double>> mask) async {
    final h = mask.length;
    final w = mask.first.length;
    final bytes = Uint8List(w * h * 4);
    var i = 0;
    for (var y = 0; y < h; y++) {
      final row = mask[y];
      for (var x = 0; x < w; x++) {
        final v = (row[x].clamp(0.0, 1.0) * 255).toInt();
        bytes[i++] = v; // R
        bytes[i++] = v; // G
        bytes[i++] = v; // B
        bytes[i++] = 255; // A
      }
    }

    final buffer = await ImmutableBuffer.fromUint8List(bytes);
    final desc = ImageDescriptor.raw(
      buffer,
      width: w,
      height: h,
      pixelFormat: PixelFormat.rgba8888,
    );
    final codec = await desc.instantiateCodec();
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  @override
  Widget build(BuildContext context) {
    final program = _program;
    final bg = _bgImage;
    final stash = _mustacheImage;
    final mask = _maskImage;
    if (program == null || bg == null || stash == null || mask == null) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _SelfieShaderPainter(
          program: program,
          background: bg,
          mustache: stash,
          mask: mask,
          detections: widget.detections,
          poseDetections: widget.poseDetections,
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
          mode: widget.mode,
          mustacheAlpha: widget.mustacheAlpha,
          mustacheScale: widget.mustacheScale,
          srcW: _srcW,
          srcH: _srcH,
          debugPose: widget.debugPose,
        ),
      ),
    );
  }
}

class _SelfieShaderPainter extends CustomPainter {
  _SelfieShaderPainter({
    required this.program,
    required this.background,
    required this.mustache,
    required this.mask,
    required this.detections,
    required this.poseDetections,
    required this.flipHorizontal,
    required this.flipVertical,
    required this.mode,
    required this.mustacheAlpha,
    required this.mustacheScale,
    required this.srcW,
    required this.srcH,
    required this.debugPose,
  });

  final ui.FragmentProgram program;
  final ui.Image background;
  final ui.Image mustache;
  final ui.Image mask;
  final List<YOLOResult> detections;
  final List<YOLOResult> poseDetections;
  final bool flipHorizontal;
  final bool flipVertical;
  final double mode;
  final double mustacheAlpha;
  final double mustacheScale;
  final double? srcW;
  final double? srcH;
  final bool debugPose;

  static const double _noseThreshold = 0.15;
  static const double _eyeThreshold = 0.25;
  static const double _candidateThreshold = 0.05;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;

    final face = _resolveFaceData(size);
    final shader = program.fragmentShader();

    shader.setImageSampler(0, mask);
    shader.setImageSampler(1, background);
    shader.setImageSampler(2, mustache);

    int i = 0;
    shader.setFloat(i++, size.width);
    shader.setFloat(i++, size.height);
    shader.setFloat(i++, mode);
    shader.setFloat(i++, flipHorizontal ? 1.0 : 0.0);
    shader.setFloat(i++, flipVertical ? 1.0 : 0.0);

    shader.setFloat(i++, face.imageWidth);
    shader.setFloat(i++, face.imageHeight);
    shader.setFloat(i++, 0.0); // uRotation placeholder

    if (face.hasFace) {
      final rect = face.faceRect!;
      shader.setFloat(i++, 1.0);
      shader.setFloat(i++, rect.left);
      shader.setFloat(i++, rect.top);
      shader.setFloat(i++, rect.right);
      shader.setFloat(i++, rect.bottom);
    } else {
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
    }

    if (face.noseBridgeStart != null && face.noseBridgeEnd != null) {
      shader.setFloat(i++, 2.0);
      shader.setFloat(i++, face.noseBridgeStart!.dx);
      shader.setFloat(i++, face.noseBridgeStart!.dy);
      shader.setFloat(i++, face.noseBridgeEnd!.dx);
      shader.setFloat(i++, face.noseBridgeEnd!.dy);
    } else {
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
    }

    if (face.upperLipCenter != null) {
      shader.setFloat(i++, 1.0);
      shader.setFloat(i++, face.upperLipCenter!.dx);
      shader.setFloat(i++, face.upperLipCenter!.dy);
    } else {
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
      shader.setFloat(i++, 0.0);
    }

    shader.setFloat(i++, mustacheScale);
    shader.setFloat(i++, mustacheAlpha);

    final paint = Paint()..shader = shader;
    canvas.drawRect(Offset.zero & size, paint);

    if (debugPose) _paintPoseDebug(canvas, size, face);
  }

  void _paintPoseDebug(Canvas canvas, Size size, _FaceUniformData face) {
    final transform = face.transform;
    final faceRect = _mapRectToCanvas(face.faceRect, transform, size);
    final leftEye = _mapPointToCanvas(face.leftEye, transform, size);
    final rightEye = _mapPointToCanvas(face.rightEye, transform, size);
    final nose = _mapPointToCanvas(face.nose, transform, size);
    final noseBridgeStart = _mapPointToCanvas(
      face.noseBridgeStart,
      transform,
      size,
    );
    final noseBridgeEnd = _mapPointToCanvas(
      face.noseBridgeEnd,
      transform,
      size,
    );

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

  Rect? _mapRectToCanvas(
    Rect? imageRect,
    _LetterboxTransform transform,
    Size size,
  ) {
    if (imageRect == null || imageRect.isEmpty) return null;
    final topLeft = _mapPointToCanvas(imageRect.topLeft, transform, size);
    final bottomRight = _mapPointToCanvas(
      imageRect.bottomRight,
      transform,
      size,
    );
    if (topLeft == null || bottomRight == null) return null;
    final left = math.min(topLeft.dx, bottomRight.dx);
    final right = math.max(topLeft.dx, bottomRight.dx);
    final top = math.min(topLeft.dy, bottomRight.dy);
    final bottom = math.max(topLeft.dy, bottomRight.dy);
    if (right <= left || bottom <= top) return null;
    return Rect.fromLTRB(left, top, right, bottom);
  }

  _FaceUniformData _resolveFaceData(Size canvasSize) {
    final transform = _LetterboxTransform.fromView(
      viewSize: canvasSize,
      sourceWidth: srcW,
      sourceHeight: srcH,
    );
    final imageWidth = transform.srcWidth;
    final imageHeight = transform.srcHeight;

    final poseDetection = _pickPrimaryPose(poseDetections, canvasSize);
    final maskDetection = _pickPrimaryMask(detections);
    final baseDetection = poseDetection ?? maskDetection;

    Rect? faceRect;
    if (baseDetection != null && !baseDetection.boundingBox.isEmpty) {
      faceRect = _clampRectToImage(
        transform.viewRectToImage(baseDetection.boundingBox),
        transform,
      );
    }

    Offset? noseBridgeStart;
    Offset? noseBridgeEnd;
    Offset? upperLipCenter;
    Offset? leftEye;
    Offset? rightEye;
    Offset? nose;

    if (poseDetection != null) {
      final landmarks = _extractPoseLandmarks(
        detection: poseDetection,
        canvasSize: canvasSize,
        transform: transform,
        seedRect: faceRect,
      );
      faceRect ??= landmarks.faceRect;
      noseBridgeStart = landmarks.noseBridgeStart;
      noseBridgeEnd = landmarks.noseBridgeEnd;
      upperLipCenter = landmarks.upperLipCenter;
      leftEye = landmarks.leftEye;
      rightEye = landmarks.rightEye;
      nose = landmarks.nose;
    }

    if (faceRect == null &&
        maskDetection != null &&
        !maskDetection.boundingBox.isEmpty) {
      faceRect = _clampRectToImage(
        transform.viewRectToImage(maskDetection.boundingBox),
        transform,
      );
    }

    if (faceRect == Rect.zero) {
      faceRect = null;
    }

    if (noseBridgeEnd == null && faceRect != null) {
      noseBridgeEnd = faceRect.center;
    }
    nose ??= noseBridgeEnd;
    if (noseBridgeStart == null && noseBridgeEnd != null && faceRect != null) {
      noseBridgeStart = Offset(
        noseBridgeEnd.dx,
        math.max(faceRect.top, noseBridgeEnd.dy - faceRect.height * 0.18),
      );
    }
    if (upperLipCenter == null && noseBridgeEnd != null && faceRect != null) {
      upperLipCenter = Offset(
        noseBridgeEnd.dx,
        math.min(faceRect.bottom, noseBridgeEnd.dy + faceRect.height * 0.12),
      );
    }

    return _FaceUniformData(
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      transform: transform,
      faceRect: faceRect,
      noseBridgeStart: noseBridgeStart,
      noseBridgeEnd: noseBridgeEnd,
      upperLipCenter: upperLipCenter,
      leftEye: leftEye,
      rightEye: rightEye,
      nose: nose,
    );
  }

  YOLOResult? _pickPrimaryMask(List<YOLOResult> list) {
    YOLOResult? best;
    var bestArea = -1.0;
    for (final d in list) {
      final mask = d.mask;
      if (mask == null || mask.isEmpty) continue;
      final b = d.boundingBox;
      if (b.isEmpty) continue;
      final area = (b.width * b.height).abs();
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
    required Size canvasSize,
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
        canvasSize: canvasSize,
        transform: transform,
      ),
      growable: false,
    );

    Offset? noseBridgeEnd;
    Offset? nose;
    final nosePoint = _pointWithThreshold(points, 0, _noseThreshold);
    if (nosePoint != null) {
      noseBridgeEnd = nosePoint.imagePosition;
      nose = nosePoint.imagePosition;
    }

    Offset? noseBridgeStart;
    Offset? leftEyePosition;
    Offset? rightEyePosition;
    final leftEyePoint = _pointWithThreshold(points, 1, _eyeThreshold);
    final rightEyePoint = _pointWithThreshold(points, 2, _eyeThreshold);
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
        if (pt == null || pt.confidence < _candidateThreshold) continue;
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
    required Size canvasSize,
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
    );
    if (mapped == null) return null;

    final viewClamped = Offset(
      mapped.dx.clamp(0.0, canvasSize.width),
      mapped.dy.clamp(0.0, canvasSize.height),
    );
    final imagePoint = transform.viewToImage(mapped);
    final imageClamped = Offset(
      imagePoint.dx.clamp(0.0, transform.srcWidth),
      imagePoint.dy.clamp(0.0, transform.srcHeight),
    );
    return _PosePoint(
      viewPosition: viewClamped,
      imagePosition: imageClamped,
      confidence: confidence,
    );
  }

  Offset? _convertPosePoint({
    required YOLOResult detection,
    required Point point,
  }) {
    final looksNormalized =
        point.x >= 0.0 && point.x <= 1.0 && point.y >= 0.0 && point.y <= 1.0;
    if (looksNormalized && !detection.boundingBox.isEmpty) {
      final bbox = detection.boundingBox;
      return Offset(
        bbox.left + point.x * bbox.width,
        bbox.top + point.y * bbox.height,
      );
    }

    if (point.x.isFinite && point.y.isFinite) {
      return Offset(point.x.toDouble(), point.y.toDouble());
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
    if (!pt.viewPosition.dx.isFinite || !pt.viewPosition.dy.isFinite) {
      return null;
    }
    if (!pt.imagePosition.dx.isFinite || !pt.imagePosition.dy.isFinite) {
      return null;
    }
    return pt;
  }

  @override
  bool shouldRepaint(covariant _SelfieShaderPainter oldDelegate) {
    return oldDelegate.background != background ||
        oldDelegate.mustache != mustache ||
        oldDelegate.mask != mask ||
        oldDelegate.detections != detections ||
        oldDelegate.poseDetections != poseDetections ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical ||
        oldDelegate.mode != mode ||
        oldDelegate.mustacheAlpha != mustacheAlpha ||
        oldDelegate.mustacheScale != mustacheScale ||
        oldDelegate.srcW != srcW ||
        oldDelegate.srcH != srcH ||
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

  Size get srcSize => Size(srcWidth, srcHeight);

  Offset imageToView(Offset imagePoint) {
    return Offset(dx + imagePoint.dx * scale, dy + imagePoint.dy * scale);
  }

  Offset viewToImage(Offset viewPoint) {
    return Offset((viewPoint.dx - dx) / scale, (viewPoint.dy - dy) / scale);
  }

  Rect imageRectToView(Rect rect) {
    return Rect.fromLTRB(
      dx + rect.left * scale,
      dy + rect.top * scale,
      dx + rect.right * scale,
      dy + rect.bottom * scale,
    );
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
    final srcW = (sourceWidth ?? viewSize.width).clamp(1.0, double.infinity);
    final srcH = (sourceHeight ?? viewSize.height).clamp(1.0, double.infinity);
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

class _FaceUniformData {
  const _FaceUniformData({
    required this.imageWidth,
    required this.imageHeight,
    required this.transform,
    required this.faceRect,
    required this.noseBridgeStart,
    required this.noseBridgeEnd,
    required this.upperLipCenter,
    this.leftEye,
    this.rightEye,
    this.nose,
  });

  final double imageWidth;
  final double imageHeight;
  final _LetterboxTransform transform;
  final Rect? faceRect;
  final Offset? noseBridgeStart;
  final Offset? noseBridgeEnd;
  final Offset? upperLipCenter;
  final Offset? leftEye;
  final Offset? rightEye;
  final Offset? nose;

  bool get hasFace => faceRect != null && !faceRect!.isEmpty;
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
  const _PosePoint({
    required this.viewPosition,
    required this.imagePosition,
    required this.confidence,
  });

  final Offset viewPosition;
  final Offset imagePosition;
  final double confidence;
}
