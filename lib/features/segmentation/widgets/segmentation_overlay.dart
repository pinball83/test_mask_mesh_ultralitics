import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class SegmentationOverlay extends StatefulWidget {
  const SegmentationOverlay({
    super.key,
    required this.detections,
    required this.maskThreshold,
    required this.flipHorizontal,
    required this.flipVertical,
    this.backgroundAsset = 'assets/images/bg_image.jpg',
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final String backgroundAsset;

  @override
  State<SegmentationOverlay> createState() => _SegmentationOverlayState();
}

class _SegmentationOverlayState extends State<SegmentationOverlay> {
  ui.Image? _backgroundImage;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  @override
  void initState() {
    super.initState();
    _resolveBackgroundImage();
  }

  @override
  void didUpdateWidget(covariant SegmentationOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.backgroundAsset != oldWidget.backgroundAsset) {
      _resolveBackgroundImage();
    }
  }

  @override
  void dispose() {
    _disposeImageStream();
    super.dispose();
  }

  void _resolveBackgroundImage() {
    _disposeImageStream();
    final imageProvider = AssetImage(widget.backgroundAsset);
    final stream = imageProvider.resolve(ImageConfiguration.empty);
    _imageStream = stream;
    _imageStreamListener = ImageStreamListener((imageInfo, _) {
      setState(() {
        _backgroundImage = imageInfo.image;
      });
    });
    stream.addListener(_imageStreamListener!);
  }

  void _disposeImageStream() {
    if (_imageStream != null && _imageStreamListener != null) {
      _imageStream!.removeListener(_imageStreamListener!);
    }
    _imageStream = null;
    _imageStreamListener = null;
  }

  @override
  Widget build(BuildContext context) {
    if (widget.detections.isEmpty) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _SegmentationMaskPainter(
          detections: widget.detections,
          maskThreshold: widget.maskThreshold,
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
          backgroundImage: _backgroundImage,
        ),
      ),
    );
  }
}

class _SegmentationMaskPainter extends CustomPainter {
  const _SegmentationMaskPainter({
    required this.detections,
    required this.maskThreshold,
    required this.flipHorizontal,
    required this.flipVertical,
    this.backgroundImage,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final ui.Image? backgroundImage;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.clipRect(Offset.zero & size);

    final sourceSize = _estimateSourceSize();
    final sourceWidth = sourceSize?.width ?? size.width;
    final sourceHeight = sourceSize?.height ?? size.height;

    if (sourceWidth <= 0 || sourceHeight <= 0) {
      canvas.restore();
      return;
    }

    final scaleX = size.width / sourceWidth;
    final scaleY = size.height / sourceHeight;
    final scale = scaleX > scaleY ? scaleX : scaleY;
    final scaledWidth = sourceWidth * scale;
    final scaledHeight = sourceHeight * scale;
    final dx = (size.width - scaledWidth) / 2;
    final dy = (size.height - scaledHeight) / 2;

    final backgroundPaint = Paint()
      ..color = Colors.black.withValues(alpha: 0.5);
    canvas.saveLayer(Offset.zero & size, Paint());
    if (backgroundImage != null) {
      paintImage(
        canvas: canvas,
        rect: Offset.zero & size,
        image: backgroundImage!,
        fit: BoxFit.cover,
      );
    } else {
      canvas.drawRect(Offset.zero & size, backgroundPaint);
    }
    final clearPaint = Paint()..blendMode = BlendMode.clear;

    for (final detection in detections) {
      final mask = detection.mask;
      final bounds = detection.boundingBox;
      if (mask == null || mask.isEmpty || bounds.isEmpty) continue;

      final maskHeight = mask.length;
      final maskWidth = mask.first.length;
      if (maskWidth == 0 || maskHeight == 0) continue;

      final cellWidth = scaledWidth / maskWidth;
      final cellHeight = scaledHeight / maskHeight;

      var startX = 0;
      var endX = maskWidth;
      var startY = 0;
      var endY = maskHeight;

      final normalizedBounds = detection.normalizedBox;
      if (!normalizedBounds.isEmpty) {
        startX = _clampIndex(
          (normalizedBounds.left.clamp(0.0, 1.0) * maskWidth).floor(),
          0,
          maskWidth,
        );
        endX = _clampIndex(
          (normalizedBounds.right.clamp(0.0, 1.0) * maskWidth).ceil(),
          startX + 1,
          maskWidth,
        );
        startY = _clampIndex(
          (normalizedBounds.top.clamp(0.0, 1.0) * maskHeight).floor(),
          0,
          maskHeight,
        );
        endY = _clampIndex(
          (normalizedBounds.bottom.clamp(0.0, 1.0) * maskHeight).ceil(),
          startY + 1,
          maskHeight,
        );
      }

      for (var y = startY; y < endY; y++) {
        final row = mask[y];
        final mappedY = flipVertical ? (maskHeight - 1 - y) : y;
        final top = dy + mappedY * cellHeight;

        for (var x = startX; x < endX; x++) {
          final value = row[x];
          if (value < maskThreshold) continue;

          final mappedX = flipHorizontal ? (maskWidth - 1 - x) : x;
          final left = dx + mappedX * cellWidth;

          canvas.drawRect(
            Rect.fromLTWH(left, top, cellWidth, cellHeight),
            clearPaint,
          );
        }
      }
    }

    canvas.restore();
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _SegmentationMaskPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.maskThreshold != maskThreshold ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical;
  }

  Size? _estimateSourceSize() {
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

  int _clampIndex(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
  }
}
