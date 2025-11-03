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
  }

  _FaceUniformData _resolveFaceData(Size canvasSize) {
    final imageWidth = canvasSize.width > 0 ? canvasSize.width : (srcW ?? 1.0);
    final imageHeight = canvasSize.height > 0
        ? canvasSize.height
        : (srcH ?? 1.0);

    final poseDetection = _pickPrimaryPose(poseDetections, canvasSize);
    final maskDetection = _pickPrimaryMask(detections);
    final baseDetection = poseDetection ?? maskDetection;

    Rect? faceRect;
    if (baseDetection != null && !baseDetection.boundingBox.isEmpty) {
      faceRect = _clampRectToCanvas(baseDetection.boundingBox, canvasSize);
    }

    Offset? noseBridgeStart;
    Offset? noseBridgeEnd;
    Offset? upperLipCenter;

    if (poseDetection != null) {
      final landmarks = _extractPoseLandmarks(
        detection: poseDetection,
        canvasSize: canvasSize,
        seedRect: faceRect,
      );
      faceRect ??= landmarks.faceRect;
      noseBridgeStart = landmarks.noseBridgeStart;
      noseBridgeEnd = landmarks.noseBridgeEnd;
      upperLipCenter = landmarks.upperLipCenter;
    }

    if (faceRect == null &&
        maskDetection != null &&
        !maskDetection.boundingBox.isEmpty) {
      faceRect = _clampRectToCanvas(maskDetection.boundingBox, canvasSize);
    }

    if (faceRect == Rect.zero) {
      faceRect = null;
    }

    if (noseBridgeEnd == null && faceRect != null) {
      noseBridgeEnd = faceRect.center;
    }
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
      faceRect: faceRect,
      noseBridgeStart: noseBridgeStart,
      noseBridgeEnd: noseBridgeEnd,
      upperLipCenter: upperLipCenter,
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

  Rect _clampRectToCanvas(Rect rect, Size size) {
    final left = rect.left.clamp(0.0, size.width);
    final top = rect.top.clamp(0.0, size.height);
    final right = rect.right.clamp(0.0, size.width);
    final bottom = rect.bottom.clamp(0.0, size.height);
    if (right <= left || bottom <= top) {
      return Rect.zero;
    }
    return Rect.fromLTRB(left, top, right, bottom);
  }

  _PoseLandmarks _extractPoseLandmarks({
    required YOLOResult detection,
    required Size canvasSize,
    Rect? seedRect,
  }) {
    Rect? faceRect = seedRect;
    if (faceRect == null && !detection.boundingBox.isEmpty) {
      faceRect = _clampRectToCanvas(detection.boundingBox, canvasSize);
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
      ),
      growable: false,
    );

    Offset? noseBridgeEnd;
    final nose = _pointWithThreshold(points, 0, _noseThreshold);
    if (nose != null) {
      noseBridgeEnd = nose.position;
    }

    Offset? noseBridgeStart;
    final leftEye = _pointWithThreshold(points, 1, _eyeThreshold);
    final rightEye = _pointWithThreshold(points, 2, _eyeThreshold);
    if (leftEye != null && rightEye != null) {
      noseBridgeStart = Offset(
        (leftEye.position.dx + rightEye.position.dx) / 2,
        (leftEye.position.dy + rightEye.position.dy) / 2,
      );
    }

    Offset? upperLipCenter;
    if (noseBridgeEnd != null && faceRect != null) {
      final verticalLimit = faceRect.height * 0.35;
      final horizontalLimit = faceRect.width * 0.45;
      final candidates = <_PosePoint>[];
      for (final pt in points) {
        if (pt == null || pt.confidence < _candidateThreshold) continue;
        final pos = pt.position;
        if (pos.dy <= noseBridgeEnd.dy) continue;
        if (pos.dy > noseBridgeEnd.dy + verticalLimit) continue;
        if ((pos.dx - noseBridgeEnd.dx).abs() > horizontalLimit) continue;
        candidates.add(pt);
      }

      if (candidates.isNotEmpty) {
        candidates.sort((a, b) => a.position.dx.compareTo(b.position.dx));
        final left = candidates.first;
        final right = candidates.last;
        if (candidates.length >= 2 &&
            (right.position.dx - left.position.dx).abs() >
                faceRect.width * 0.05) {
          upperLipCenter = Offset(
            (left.position.dx + right.position.dx) / 2,
            (left.position.dy + right.position.dy) / 2,
          );
        } else {
          final avgX =
              candidates.fold<double>(0, (sum, e) => sum + e.position.dx) /
              candidates.length;
          final avgY =
              candidates.fold<double>(0, (sum, e) => sum + e.position.dy) /
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
    );
  }

  _PosePoint? _mapPosePoint({
    required YOLOResult detection,
    required int index,
    required Size canvasSize,
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

    final clamped = Offset(
      mapped.dx.clamp(0.0, canvasSize.width),
      mapped.dy.clamp(0.0, canvasSize.height),
    );
    return _PosePoint(position: clamped, confidence: confidence);
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
    if (!pt.position.dx.isFinite || !pt.position.dy.isFinite) return null;
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
        oldDelegate.srcH != srcH;
  }
}

class _FaceUniformData {
  const _FaceUniformData({
    required this.imageWidth,
    required this.imageHeight,
    required this.faceRect,
    required this.noseBridgeStart,
    required this.noseBridgeEnd,
    required this.upperLipCenter,
  });

  final double imageWidth;
  final double imageHeight;
  final Rect? faceRect;
  final Offset? noseBridgeStart;
  final Offset? noseBridgeEnd;
  final Offset? upperLipCenter;

  bool get hasFace => faceRect != null && !faceRect!.isEmpty;
}

class _PoseLandmarks {
  const _PoseLandmarks({
    required this.faceRect,
    this.noseBridgeStart,
    this.noseBridgeEnd,
    this.upperLipCenter,
  });

  final Rect? faceRect;
  final Offset? noseBridgeStart;
  final Offset? noseBridgeEnd;
  final Offset? upperLipCenter;
}

class _PosePoint {
  const _PosePoint({required this.position, required this.confidence});

  final Offset position;
  final double confidence;
}
