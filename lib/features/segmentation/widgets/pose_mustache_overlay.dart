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
  });

  final List<YOLOResult> detections;
  final String assetPath;
  final double noseConfidenceThreshold;
  final bool flipHorizontal;
  final bool flipVertical;

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
  });

  final List<YOLOResult> detections;
  final ui.Image mustacheImage;
  final double noseConfidenceThreshold;
  final bool flipHorizontal;
  final bool flipVertical;

  static const int _noseIndex = 0;
  static const int _leftEyeIndex = 1;
  static const int _rightEyeIndex = 2;
  static const int _leftEarIndex = 3;
  static const int _rightEarIndex = 4;
  static const double _minMustacheWidth = 24;
  static const double _maxMustacheWidthFactor = 0.6;
  static const double _fallbackWidthFactor = 0.45;
  static const double _eyeWidthMultiplier = 1.35;
  static const double _earWidthMultiplier = 1.1;
  static const double _verticalOffsetFactor = 0.22;
  static const double _eyeConfidenceThreshold = 0.25;
  static const double _earConfidenceThreshold = 0.25;

  @override
  void paint(Canvas canvas, Size size) {
    final transform = _ViewTransform.fromDetections(detections, size);
    if (transform == null) return;

    for (final detection in detections) {
      final keypoints = detection.keypoints;
      if (keypoints == null || keypoints.isEmpty) continue;

      final confidences = detection.keypointConfidences;
      final nose = _keypointForIndex(
        keypoints,
        confidences,
        _noseIndex,
        threshold: noseConfidenceThreshold,
        transform: transform,
        viewSize: size,
      );
      if (nose == null) continue;

      final leftEar = _keypointForIndex(
        keypoints,
        confidences,
        _leftEarIndex,
        threshold: _earConfidenceThreshold,
        transform: transform,
        viewSize: size,
      );
      final rightEar = _keypointForIndex(
        keypoints,
        confidences,
        _rightEarIndex,
        threshold: _earConfidenceThreshold,
        transform: transform,
        viewSize: size,
      );

      final leftEye = _keypointForIndex(
        keypoints,
        confidences,
        _leftEyeIndex,
        threshold: _eyeConfidenceThreshold,
        transform: transform,
        viewSize: size,
      );
      final rightEye = _keypointForIndex(
        keypoints,
        confidences,
        _rightEyeIndex,
        threshold: _eyeConfidenceThreshold,
        transform: transform,
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

      final verticalOffset =
          height * _verticalOffsetFactor * (flipVertical ? -1 : 1);

      final center = Offset(nose.dx, nose.dy + verticalOffset);

      canvas.save();
      canvas.translate(center.dx, center.dy);

      final rotation = _resolveRotation(
        leftEar ?? leftEye,
        rightEar ?? rightEye,
      );
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
        image: mustacheImage,
        fit: BoxFit.contain,
        filterQuality: FilterQuality.high,
      );
      canvas.restore();
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

  Offset? _keypointForIndex(
    List<Point> keypoints,
    List<double>? confidences,
    int index, {
    required double threshold,
    required _ViewTransform transform,
    required Size viewSize,
  }) {
    if (index < 0 || index >= keypoints.length) return null;

    if (confidences != null &&
        index < confidences.length &&
        confidences[index] < threshold) {
      return null;
    }

    final mapped = transform.map(keypoints[index]);
    if (!mapped.dx.isFinite || !mapped.dy.isFinite) {
      return null;
    }
    return _applyFlip(mapped, viewSize);
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
  });

  final double scale;
  final double dx;
  final double dy;

  // Map normalized point (0..1) to view coordinates using uniform scale and offsets.
  Offset map(Point point) {
    return Offset(
      dx + point.x * scale,
      dy + point.y * scale,
    );
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
    if (iw == null || ih == null) return null;

    final vw = viewSize.width;
    final vh = viewSize.height;
    if (vw <= 0 || vh <= 0) return null;

    final scale = max(vw / iw, vh / ih);
    final scaledW = iw * scale;
    final scaledH = ih * scale;
    final dx = (vw - scaledW) / 2.0;
    final dy = (vh - scaledH) / 2.0;

    return _ViewTransform(scale: scale, dx: dx, dy: dy);
  }

  // Legacy estimator removed; mapping now derived per-detection to match YOLOView's transform.
}
