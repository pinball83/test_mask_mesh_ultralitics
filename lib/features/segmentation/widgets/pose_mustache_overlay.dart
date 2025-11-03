import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class PoseMustacheOverlay extends StatefulWidget {
  const PoseMustacheOverlay({
    super.key,
    required this.detections,
    this.assetPath = 'assets/images/mustash.png',
    this.noseConfidenceThreshold = 0.15,
    this.flipHorizontal = false,
    this.flipVertical = false,
    this.debug = true,
  });

  final List<YOLOResult> detections;
  final String assetPath;
  final double noseConfidenceThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final bool debug;

  @override
  State<PoseMustacheOverlay> createState() => _PoseMustacheOverlayState();
}

class _PoseMustacheOverlayState extends State<PoseMustacheOverlay> {
  ui.Image? _mustacheImage;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  @override
  void initState() {
    super.initState();
    _resolveImage();
  }

  @override
  void didUpdateWidget(covariant PoseMustacheOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.assetPath != widget.assetPath) {
      _resolveImage();
    }
  }

  @override
  void dispose() {
    _disposeImageStream();
    super.dispose();
  }

  void _resolveImage() {
    _disposeImageStream();
    final provider = AssetImage(widget.assetPath);
    final stream = provider.resolve(ImageConfiguration.empty);
    _imageStream = stream;
    _imageStreamListener = ImageStreamListener((imageInfo, _) {
      setState(() {
        _mustacheImage = imageInfo.image;
      });
    });
    stream.addListener(_imageStreamListener!);
  }

  void _disposeImageStream() {
    final stream = _imageStream;
    final listener = _imageStreamListener;
    if (stream != null && listener != null) {
      stream.removeListener(listener);
    }
    _imageStream = null;
    _imageStreamListener = null;
  }

  @override
  Widget build(BuildContext context) {
    if (_mustacheImage == null || widget.detections.isEmpty) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _PoseMustachePainter(
          detections: widget.detections,
          mustacheImage: _mustacheImage!,
          noseConfidenceThreshold: widget.noseConfidenceThreshold,
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
          debug: widget.debug,
        ),
      ),
    );
  }
}

class _PoseMustachePainter extends CustomPainter {
  const _PoseMustachePainter({
    required this.detections,
    required this.mustacheImage,
    required this.noseConfidenceThreshold,
    required this.flipHorizontal,
    required this.flipVertical,
    required this.debug,
  });

  final List<YOLOResult> detections;
  final ui.Image mustacheImage;
  final double noseConfidenceThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final bool debug;

  static const int _noseIndex = 0;
  static const int _leftEyeIndex = 1;
  static const int _rightEyeIndex = 2;
  static const int _leftEarIndex = 3;
  static const int _rightEarIndex = 4;
  static const double _minMustacheWidth = 24;
  static const double _maxMustacheWidthFactor = 0.6;
  static const double _fallbackWidthFactor = 0.36;
  static const double _eyeWidthMultiplier = 1.08;
  static const double _earWidthMultiplier = 0.95;
  // No fixed vertical offset; placement is derived from geometry.
  static const double _eyeConfidenceThreshold = 0.25;
  static const double _earConfidenceThreshold = 0.25;

  @override
  void paint(Canvas canvas, Size size) {
    final viewTransform = _ViewTransform.fromDetections(detections, size);
    if (viewTransform == null) return;

    final detection = _pickPrimaryDetection(detections, size);
    if (detection == null) return;
    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.isEmpty) return;
    final confidences = detection.keypointConfidences;
    // Treat keypoints as normalized (0..1) in source image space.
    // Many pose models emit normalized coordinates; forcing this removes
    // ambiguity from device-specific pixel scales.
    final normalized = true;

    final nose = _mapKeypoint(
      keypoints,
      confidences,
      index: _noseIndex,
      threshold: noseConfidenceThreshold,
      viewTransform: viewTransform,
      detection: detection,
      normalized: normalized,
      viewSize: size,
    );
    final bb = detection.boundingBox;

    final leftEar = _mapKeypoint(
      keypoints,
      confidences,
      index: _leftEarIndex,
      threshold: _earConfidenceThreshold,
      viewTransform: viewTransform,
      detection: detection,
      normalized: normalized,
      viewSize: size,
    );
    final rightEar = _mapKeypoint(
      keypoints,
      confidences,
      index: _rightEarIndex,
      threshold: _earConfidenceThreshold,
      viewTransform: viewTransform,
      detection: detection,
      normalized: normalized,
      viewSize: size,
    );

    final leftEye = _mapKeypoint(
      keypoints,
      confidences,
      index: _leftEyeIndex,
      threshold: _eyeConfidenceThreshold,
      viewTransform: viewTransform,
      detection: detection,
      normalized: normalized,
      viewSize: size,
    );
    final rightEye = _mapKeypoint(
      keypoints,
      confidences,
      index: _rightEyeIndex,
      threshold: _eyeConfidenceThreshold,
      viewTransform: viewTransform,
      detection: detection,
      normalized: normalized,
      viewSize: size,
    );

    final width = _resolveMustacheWidth(
      detection: detection,
      viewSize: size,
      leftEar: leftEar,
      rightEar: rightEar,
      leftEye: leftEye,
      rightEye: rightEye,
    );
      final height =
          width *
          (mustacheImage.height.toDouble() / mustacheImage.width.toDouble());

      // Determine head rotation from anchors and place the mustache so that
      // its top edge touches the nose (data-driven; no magic factors).
    final rotation = _resolveRotation(
      leftEar ?? leftEye,
      rightEar ?? rightEye,
    );
    // Anchor: prefer mapped nose; otherwise fall back to face box center.
    final anchor = nose ?? Offset(bb.left + bb.width / 2, bb.top + bb.height * 0.62);
    final offsetX = -sin(rotation) * (height / 2);
    final offsetY =  cos(rotation) * (height / 2);
    final center = Offset(anchor.dx + offsetX, anchor.dy + offsetY);

    canvas.save();
    canvas.translate(center.dx, center.dy);
    if (rotation != 0) canvas.rotate(rotation);

      final rect = Rect.fromCenter(
        center: Offset.zero,
        width: width,
        height: height,
      );

      paintImage(
        canvas: canvas,
        rect: rect,
        image: mustacheImage,
        fit: BoxFit.contain,
        filterQuality: FilterQuality.high,
      );

    // Draw a visible bounding box around the mustache image
    final bbPaint = Paint()
      ..color = Colors.limeAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawRect(rect, bbPaint);
    canvas.restore();

    if (debug) {
      // Visualize candidate detections with mapped nose points
      final nosePaint = Paint()
        ..style = PaintingStyle.fill
        ..color = Colors.redAccent;
      final crossPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0
        ..color = Colors.redAccent.withValues(alpha: 0.9);
      final boxPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5
        ..color = Colors.orangeAccent.withValues(alpha: 0.9);
      final rawPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = Colors.blueAccent;

      for (final d in detections) {
        final kps = d.keypoints;
        if (kps == null || kps.isEmpty) continue;
        final n = _mapKeypoint(
          kps,
          d.keypointConfidences,
          index: _noseIndex,
          threshold: 0.01,
          viewTransform: viewTransform,
          detection: d,
          normalized: _areKeypointsNormalized(kps),
          viewSize: size,
        );
        if (n != null) {
          canvas.drawCircle(n, 5, nosePaint);
          canvas.drawLine(Offset(n.dx - 8, n.dy), Offset(n.dx + 8, n.dy), crossPaint);
          canvas.drawLine(Offset(n.dx, n.dy - 8), Offset(n.dx, n.dy + 8), crossPaint);
        }
        // Draw raw model-reported nose (no mapping) in blue to inspect source coords
        final raw = kps[_noseIndex];
        canvas.drawCircle(Offset(raw.x.toDouble(), raw.y.toDouble()), 3, rawPaint);
        // Also print numbers in the corner for quick read
        final tp = TextPainter(
          text: TextSpan(
            text:
                'raw nose: (${raw.x.toStringAsFixed(1)}, ${raw.y.toStringAsFixed(1)})\n' +
                'bb: L${d.boundingBox.left.toStringAsFixed(1)} T${d.boundingBox.top.toStringAsFixed(1)} W${d.boundingBox.width.toStringAsFixed(1)} H${d.boundingBox.height.toStringAsFixed(1)}',
            style: const TextStyle(color: Colors.white, fontSize: 11),
          ),
          textDirection: TextDirection.ltr,
          maxLines: 3,
        )..layout(maxWidth: size.width - 12);
        tp.paint(canvas, const Offset(6, 6));
        final bb = d.boundingBox;
        if (!bb.isEmpty) {
          canvas.drawRect(bb, boxPaint);
        }
      }
    }
  }

  double _resolveMustacheWidth({
    required YOLOResult detection,
    required Size viewSize,
    Offset? leftEar,
    Offset? rightEar,
    Offset? leftEye,
    Offset? rightEye,
  }) {
    final earSpan = leftEar != null && rightEar != null
        ? (leftEar - rightEar).distance
        : 0;
    final eyeSpan = leftEye != null && rightEye != null
        ? (leftEye - rightEye).distance
        : 0;

    double width;
    if (earSpan > 0) {
      width = earSpan * _earWidthMultiplier;
    } else if (eyeSpan > 0) {
      width = eyeSpan * _eyeWidthMultiplier;
    } else {
      width = detection.boundingBox.width * _fallbackWidthFactor;
    }

    if (!width.isFinite || width <= 0) {
      width = viewSize.width * _fallbackWidthFactor;
    }

    return width.clamp(
      _minMustacheWidth,
      viewSize.width * _maxMustacheWidthFactor,
    );
  }

  double _resolveRotation(Offset? anchorLeft, Offset? anchorRight) {
    if (anchorLeft == null || anchorRight == null) {
      return 0;
    }
    final dx = anchorRight.dx - anchorLeft.dx;
    final dy = anchorRight.dy - anchorLeft.dy;
    if (dx == 0 && dy == 0) {
      return 0;
    }
    return atan2(dy, dx);
  }

  bool _areKeypointsNormalized(List<Point> keypoints) => true;

  Offset? _mapKeypoint(
    List<Point> keypoints,
    List<double>? confidences, {
    required int index,
    required double threshold,
    required _ViewTransform viewTransform,
    required YOLOResult detection,
    required bool normalized,
    required Size viewSize,
  }) {
    if (index < 0 || index >= keypoints.length) return null;
    if (confidences != null &&
        index < confidences.length &&
        confidences[index] < threshold) {
      return null;
    }

    final p = keypoints[index];
    final bb = detection.boundingBox;

    // Deterministic rule: if 0..1 -> relative to bb; else -> pixels scaled to view
    Offset mapped;
    final looksNormalized = p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
    if (looksNormalized && !bb.isEmpty) {
      mapped = Offset(
        bb.left + p.x * bb.width,
        bb.top + p.y * bb.height,
      );
    } else {
      // Raw keypoints are already in view coordinates; use as-is.
      mapped = Offset(p.x.toDouble(), p.y.toDouble());
    }

    if (!mapped.dx.isFinite || !mapped.dy.isFinite) return null;
    return _applyFlip(mapped, viewSize);
  }

  YOLOResult? _pickPrimaryDetection(List<YOLOResult> list, Size viewSize) {
    YOLOResult? best;
    var bestScore = double.negativeInfinity;
    final minArea = (viewSize.width * viewSize.height) * 0.002; // 0.2% of view
    for (final d in list) {
      final kps = d.keypoints;
      if (kps == null || kps.isEmpty) continue;
      final box = d.boundingBox;
      final area = (box.width <= 0 || box.height <= 0)
          ? 0.0
          : (box.width * box.height);
      if (area < minArea) continue; // skip tiny spurious poses

      // Prefer larger boxes and boxes closer to view center.
      final cx = box.left + box.width / 2;
      final cy = box.top + box.height / 2;
      final dx = cx - viewSize.width / 2.0;
      final dy = cy - viewSize.height / 2.0;
      final dist2 = dx * dx + dy * dy;
      final scoreArea = area;
      final scoreCenter = -dist2; // closer to center => larger score
      final score = scoreArea * 0.5 + scoreCenter * 2.0;
      if (score > bestScore) {
        bestScore = score;
        best = d;
      }
    }
    return best;
  }

  Offset _applyFlip(Offset point, Size size) {
    final dx = flipHorizontal ? size.width - point.dx : point.dx;
    final dy = flipVertical ? size.height - point.dy : point.dy;
    return Offset(dx, dy);
  }

  @override
  bool shouldRepaint(covariant _PoseMustachePainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.mustacheImage != mustacheImage ||
        oldDelegate.noseConfidenceThreshold != noseConfidenceThreshold ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical;
  }
}

class _ViewTransform {
  const _ViewTransform({
    required this.scale,
    required this.dx,
    required this.dy,
    required this.scaledWidth,
    required this.scaledHeight,
  });

  final double scale; // scale applied to source pixels to reach view
  final double dx; // horizontal letterbox offset in view
  final double dy; // vertical letterbox offset in view
  final double scaledWidth; // width of the scaled source within view
  final double scaledHeight; // height of the scaled source within view

  // Heuristic mapping: if keypoint looks normalized (<= 1.5),
  // spread across the scaled content; otherwise treat as pixel coords
  // in the ORIGINAL source image space and scale them.
  Offset map(Point point) {
    final isNormalized =
        point.x.abs() <= 1.5 &&
        point.y.abs() <= 1.5 &&
        point.x >= 0 &&
        point.y >= 0;
    if (isNormalized) {
      return Offset(dx + point.x * scaledWidth, dy + point.y * scaledHeight);
    }
    // Treat as pixel coordinates in original image space (e.g., 640x640)
    // and map to view via uniform `scale` + letterbox offsets.
    return Offset(dx + point.x * scale, dy + point.y * scale);
  }

  // Derive scale and offsets using YOLOView's center-crop math inferred from a detection.
  static _ViewTransform? fromDetections(
    List<YOLOResult> detections,
    Size viewSize,
  ) {
    // Estimate original image width/height from any detection
    double? iw;
    double? ih;
    for (final d in detections) {
      final nb = d.normalizedBox;
      final bb = d.boundingBox;
      final nw = nb.width;
      final nh = nb.height;
      if (nw > 0 && nh > 0 && bb.width > 0 && bb.height > 0) {
        iw = bb.width / nw;
        ih = bb.height / nh;
        break;
      }
    }
    if (iw == null || ih == null) {
      // Fallback: assume source already matches the view.
      final vw = viewSize.width;
      final vh = viewSize.height;
      if (vw <= 0 || vh <= 0) return null;
      return _ViewTransform(
        scale: 1.0,
        dx: 0.0,
        dy: 0.0,
        scaledWidth: vw,
        scaledHeight: vh,
      );
    }

    final vw = viewSize.width;
    final vh = viewSize.height;
    if (vw <= 0 || vh <= 0) return null;

    final scale = max(vw / iw, vh / ih);
    final scaledW = iw * scale;
    final scaledH = ih * scale;
    final dx = (vw - scaledW) / 2.0;
    final dy = (vh - scaledH) / 2.0;

    return _ViewTransform(
      scale: scale,
      dx: dx,
      dy: dy,
      scaledWidth: scaledW,
      scaledHeight: scaledH,
    );
  }

  // Legacy estimator removed; mapping now derived per-detection to match YOLOView's transform.
}
