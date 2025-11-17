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
    this.maskSmoothing = 0.0,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final String? backgroundAsset;
  final double maskSmoothing;

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
    final asset = widget.backgroundAsset;
    if (asset == null) {
      setState(() => _backgroundImage = null);
      return;
    }
    final imageProvider = AssetImage(asset);
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
          maskSmoothing: widget.maskSmoothing,
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
    this.maskSmoothing = 2.0,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final ui.Image? backgroundImage;
  final double maskSmoothing;

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
    final dx = (size.width - scaledWidth) / 2.0;
    final dy = (size.height - scaledHeight) / 2.0;

    // Auto-detect horizontal/vertical mirroring to match YOLO outputs.
    final (bool autoFlipH, bool autoFlipV) = _detectAutoMirror(
      size: size,
      srcW: sourceWidth,
      srcH: sourceHeight,
      scale: scale,
      dx: dx,
      dy: dy,
    );
    final effectiveFlipH = flipHorizontal ^ autoFlipH;
    final effectiveFlipV = flipVertical ^ autoFlipV;

    _logTransformOnce(
      view: size,
      sourceW: sourceWidth,
      sourceH: sourceHeight,
      scale: scale,
      dx: dx,
      dy: dy,
      effFlipH: effectiveFlipH,
      effFlipV: effectiveFlipV,
    );

    // Draw the replacement background on an offscreen layer, then punch out
    // the subject area to reveal the camera feed underneath.
    final backgroundPaint = Paint()..color = Colors.black; // fully opaque fallback
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

    // Use dstOut to erase the background where the subject mask is present.
    final erasePaint = Paint()
      ..blendMode = BlendMode.dstOut
      ..isAntiAlias = true;
    if (maskSmoothing > 0) {
      erasePaint.maskFilter = ui.MaskFilter.blur(
        ui.BlurStyle.normal,
        maskSmoothing,
      );
    }

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
        final mappedY = effectiveFlipV ? (maskHeight - 1 - y) : y;
        final top = dy + mappedY * cellHeight;

        for (var x = startX; x < endX; x++) {
          final value = row[x];
          if (value < maskThreshold) continue;

          final mappedX = effectiveFlipH ? (maskWidth - 1 - x) : x;
          final left = dx + mappedX * cellWidth;

          canvas.drawRect(
            Rect.fromLTWH(left, top, cellWidth, cellHeight),
            erasePaint,
          );
        }
      }
    }

    // Composite the punched-out background over the camera feed.
    canvas.restore();
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _SegmentationMaskPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.maskThreshold != maskThreshold ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical ||
        oldDelegate.backgroundImage != backgroundImage ||
        oldDelegate.maskSmoothing != maskSmoothing;
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

  (bool, bool) _detectAutoMirror({
    required Size size,
    required double srcW,
    required double srcH,
    required double scale,
    required double dx,
    required double dy,
  }) {
    // Prefer the primary segmentation detection (largest masked area)
    YOLOResult? ref = _pickPrimaryMask(detections);
    ref ??= () {
      for (final d in detections) {
        if (!d.normalizedBox.isEmpty && !d.boundingBox.isEmpty) {
          return d;
        }
      }
      return null;
    }();
    if (ref == null) return (false, false);

    Rect _mapNormToView(Rect n) {
      final l = dx + (n.left.clamp(0.0, 1.0) * srcW) * scale;
      final t = dy + (n.top.clamp(0.0, 1.0) * srcH) * scale;
      final r = dx + (n.right.clamp(0.0, 1.0) * srcW) * scale;
      final b = dy + (n.bottom.clamp(0.0, 1.0) * srcH) * scale;
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

    // Choose the transform with minimal error; add a small bias to reduce flicker.
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

  // Lightweight throttled logger for transform parameters – useful to compare
  // with pose keypoint mapping.
  static DateTime? _lastTransformLog;
  void _logTransformOnce({
    required Size view,
    required double sourceW,
    required double sourceH,
    required double scale,
    required double dx,
    required double dy,
    required bool effFlipH,
    required bool effFlipV,
  }) {
    final now = DateTime.now();
    if (_lastTransformLog != null &&
        now.difference(_lastTransformLog!).inMilliseconds < 1000) {
      return;
    }
    _lastTransformLog = now;
    debugPrint(
      'SEGMENTATION DEBUG — view=${view.width.toStringAsFixed(0)}x${view.height.toStringAsFixed(0)} '
      'src=${sourceW.toStringAsFixed(1)}x${sourceH.toStringAsFixed(1)} '
      'scale=${scale.toStringAsFixed(4)} dx=${dx.toStringAsFixed(1)} dy=${dy.toStringAsFixed(1)} '
      'flip(h:$flipHorizontal v:$flipVertical) eff(h:$effFlipH v:$effFlipV)',
    );
  }
}
