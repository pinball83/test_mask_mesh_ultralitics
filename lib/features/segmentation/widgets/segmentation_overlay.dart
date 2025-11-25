import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

import '../utils/detection_view_geometry.dart';

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
  static const double _targetBackgroundWidth = 720.0;

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
      final original = imageInfo.image;
      _downscaleIfNeeded(original).then((scaled) {
        if (!mounted) return;
        setState(() {
          _backgroundImage = scaled;
        });
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

  Future<ui.Image> _downscaleIfNeeded(ui.Image image) async {
    if (image.width <= _targetBackgroundWidth) return image;
    final targetHeight = (image.height * (_targetBackgroundWidth / image.width))
        .round();
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final targetSize = Size(_targetBackgroundWidth, targetHeight.toDouble());
    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      Rect.fromLTWH(0, 0, targetSize.width, targetSize.height),
      Paint()..filterQuality = FilterQuality.low,
    );
    final picture = recorder.endRecording();
    return picture.toImage(targetSize.width.round(), targetSize.height.round());
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

  static final Paint _backgroundPaint = Paint()..color = Colors.black;
  static final Paint _erasePaintBase = Paint()
    ..blendMode = BlendMode.dstOut
    ..isAntiAlias = true;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.clipRect(Offset.zero & size);

    final geometry = DetectionViewGeometry.fromDetections(detections, size);
    if (geometry == null) {
      canvas.restore();
      return;
    }

    final sourceWidth = geometry.sourceWidth;
    final sourceHeight = geometry.sourceHeight;
    final scale = geometry.scale;
    final scaledWidth = sourceWidth * scale;
    final scaledHeight = sourceHeight * scale;
    final dx = geometry.dx;
    final dy = geometry.dy;

    // Mask tensors are already mirrored for the front camera by the platform
    // (see Segmenter.generateCombinedMaskImage), so applying an additional
    // horizontal flip here would invert the mask. Keep vertical correction for
    // the upside-down source tensor, but disable extra horizontal mirroring.
    const maskSourceIsMirrored = false;
    final effectiveFlipH = maskSourceIsMirrored ? false : flipHorizontal;
    const maskSourceIsUpsideDown = true;
    final effectiveFlipV = maskSourceIsUpsideDown ^ flipVertical;

    _logTransformOnce(
      view: size,
      sourceW: sourceWidth,
      sourceH: sourceHeight,
      scale: scale,
      dx: dx,
      dy: dy,
    );

    // Draw the replacement background on an offscreen layer, then punch out
    // the subject area to reveal the camera feed underneath.
    canvas.saveLayer(Offset.zero & size, Paint());
    if (backgroundImage != null) {
      paintImage(
        canvas: canvas,
        rect: Offset.zero & size,
        image: backgroundImage!,
        fit: BoxFit.cover,
      );
    } else {
      canvas.drawRect(Offset.zero & size, _backgroundPaint);
    }

    // Use dstOut to erase the background where the subject mask is present.
    final erasePaint = _erasePaintBase;
    erasePaint.maskFilter = maskSmoothing > 0
        ? ui.MaskFilter.blur(ui.BlurStyle.normal, maskSmoothing)
        : null;

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

      _drawAggregatedMaskRows(
        canvas: canvas,
        mask: mask,
        maskWidth: maskWidth,
        maskHeight: maskHeight,
        startX: startX,
        endX: endX,
        startY: startY,
        endY: endY,
        cellWidth: cellWidth,
        cellHeight: cellHeight,
        dx: dx,
        dy: dy,
        threshold: maskThreshold,
        flipH: effectiveFlipH,
        flipV: effectiveFlipV,
        paint: erasePaint,
      );
    }

    // Composite the punched-out background over the camera feed.
    canvas.restore();
    canvas.restore();
  }

  void _drawAggregatedMaskRows({
    required Canvas canvas,
    required List<List<double>> mask,
    required int maskWidth,
    required int maskHeight,
    required int startX,
    required int endX,
    required int startY,
    required int endY,
    required double cellWidth,
    required double cellHeight,
    required double dx,
    required double dy,
    required double threshold,
    required bool flipH,
    required bool flipV,
    required Paint paint,
  }) {
    for (var y = startY; y < endY; y++) {
      final row = mask[y];
      final mappedY = flipV ? (maskHeight - 1 - y) : y;
      final top = dy + mappedY * cellHeight;

      var x = startX;
      while (x < endX) {
        if (row[x] < threshold) {
          x++;
          continue;
        }

        // Aggregate contiguous "on" cells to reduce draw calls
        var runEnd = x + 1;
        while (runEnd < endX && row[runEnd] >= threshold) {
          runEnd++;
        }

        final drawStart = flipH ? (maskWidth - runEnd) : x;
        final drawWidth = (runEnd - x) * cellWidth;
        final left = dx + drawStart * cellWidth;

        canvas.drawRect(Rect.fromLTWH(left, top, drawWidth, cellHeight), paint);

        x = runEnd;
      }
    }
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

  int _clampIndex(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
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
      'flip(h:$flipHorizontal v:$flipVertical)',
    );
  }
}
