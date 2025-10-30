import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class PoseMustacheOverlay extends StatefulWidget {
  const PoseMustacheOverlay({
    super.key,
    required this.detections,
    this.assetPath = 'assets/images/mustash.png',
    this.noseConfidenceThreshold = 0.35,
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
  static const double _minMustacheWidth = 24;
  static const double _maxMustacheWidthFactor = 0.6;
  static const double _fallbackWidthFactor = 0.35;
  static const double _eyeWidthMultiplier = 1.35;
  static const double _verticalOffsetFactor = 0.35;

  @override
  void paint(Canvas canvas, Size size) {
    final transform = _ViewTransform.fromDetections(detections, size);
    if (transform == null) return;

    for (final detection in detections) {
      final keypoints = detection.keypoints;
      if (keypoints == null || keypoints.isEmpty) continue;

      final confidences = detection.keypointConfidences;
      final noseConfidence =
          confidences != null && confidences.length > _noseIndex
          ? confidences[_noseIndex]
          : 1.0;
      if (noseConfidence < noseConfidenceThreshold) continue;

      if (keypoints.length <= _noseIndex) continue;
      final nosePoint = _applyFlip(transform.map(keypoints[_noseIndex]), size);
      if (!nosePoint.dx.isFinite || !nosePoint.dy.isFinite) continue;

      Offset? leftEye;
      Offset? rightEye;
      if (keypoints.length > _leftEyeIndex &&
          keypoints.length > _rightEyeIndex) {
        final leftEyeConfidence =
            confidences != null && confidences.length > _leftEyeIndex
            ? confidences[_leftEyeIndex]
            : 1.0;
        final rightEyeConfidence =
            confidences != null && confidences.length > _rightEyeIndex
            ? confidences[_rightEyeIndex]
            : 1.0;

        if (leftEyeConfidence > 0.2 && rightEyeConfidence > 0.2) {
          leftEye = _applyFlip(transform.map(keypoints[_leftEyeIndex]), size);
          rightEye = _applyFlip(transform.map(keypoints[_rightEyeIndex]), size);
        }
      }

      final width = _resolveMustacheWidth(
        detection: detection,
        transform: transform,
        viewSize: size,
        leftEye: leftEye,
        rightEye: rightEye,
      );
      final height =
          width *
          (mustacheImage.height.toDouble() / mustacheImage.width.toDouble());

      final verticalOffset =
          height * _verticalOffsetFactor * (flipVertical ? -1 : 1);

      final center = Offset(nosePoint.dx, nosePoint.dy + verticalOffset);

      canvas.save();
      canvas.translate(center.dx, center.dy);

      final rotation = _resolveRotation(leftEye, rightEye);
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
    required _ViewTransform transform,
    required Size viewSize,
    Offset? leftEye,
    Offset? rightEye,
  }) {
    double width;
    if (leftEye != null && rightEye != null) {
      width = (leftEye - rightEye).distance * _eyeWidthMultiplier;
    } else {
      width =
          detection.boundingBox.width * transform.scale * _fallbackWidthFactor;
    }

    if (!width.isFinite || width <= 0) {
      width = viewSize.width * _fallbackWidthFactor;
    }

    width = width.clamp(
      _minMustacheWidth,
      viewSize.width * _maxMustacheWidthFactor,
    );
    return width;
  }

  double _resolveRotation(Offset? leftEye, Offset? rightEye) {
    if (leftEye == null || rightEye == null) {
      return 0;
    }
    final dx = rightEye.dx - leftEye.dx;
    final dy = rightEye.dy - leftEye.dy;
    if (dx == 0 && dy == 0) {
      return 0;
    }
    return atan2(dy, dx);
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

  Offset map(Point point) {
    return Offset(dx + point.x * scale, dy + point.y * scale);
  }

  static _ViewTransform? fromDetections(
    List<YOLOResult> detections,
    Size viewSize,
  ) {
    final sourceSize = _estimateSourceSize(detections);
    if (sourceSize == null || sourceSize.width <= 0 || sourceSize.height <= 0) {
      return null;
    }

    final scale = max(
      viewSize.width / sourceSize.width,
      viewSize.height / sourceSize.height,
    );
    final scaledWidth = sourceSize.width * scale;
    final scaledHeight = sourceSize.height * scale;
    final dx = (viewSize.width - scaledWidth) / 2;
    final dy = (viewSize.height - scaledHeight) / 2;

    return _ViewTransform(scale: scale, dx: dx, dy: dy);
  }

  static Size? _estimateSourceSize(List<YOLOResult> detections) {
    double? width;
    double? height;

    for (final detection in detections) {
      final normalized = detection.normalizedBox;
      final bounds = detection.boundingBox;
      final normalizedWidth = normalized.width;
      final normalizedHeight = normalized.height;

      if (width == null &&
          normalizedWidth > 0 &&
          bounds.width > 0 &&
          normalizedWidth.isFinite) {
        final candidate = bounds.width / normalizedWidth;
        if (candidate.isFinite && candidate > 0) {
          width = candidate;
        }
      }

      if (height == null &&
          normalizedHeight > 0 &&
          bounds.height > 0 &&
          normalizedHeight.isFinite) {
        final candidate = bounds.height / normalizedHeight;
        if (candidate.isFinite && candidate > 0) {
          height = candidate;
        }
      }

      if (width != null && height != null) {
        break;
      }
    }

    if (width != null && height != null) {
      return Size(width, height);
    }

    return null;
  }
}
