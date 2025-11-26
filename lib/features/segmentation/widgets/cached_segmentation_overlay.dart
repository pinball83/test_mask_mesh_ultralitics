import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import '../utils/detection_view_geometry.dart';

class CachedSegmentationOverlay extends StatefulWidget {
  const CachedSegmentationOverlay({
    super.key,
    required this.detections,
    required this.maskThreshold,
    required this.flipHorizontal,
    required this.flipVertical,
    this.backgroundAsset = 'assets/images/bg_image.jpg',
    this.maskSmoothing = 0.0,
    this.backgroundOpacity = 1.0,
    this.maskSourceIsUpsideDown = false,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final String? backgroundAsset;
  final double maskSmoothing;
  final double backgroundOpacity; // 0..1
  final bool maskSourceIsUpsideDown;

  @override
  State<CachedSegmentationOverlay> createState() =>
      _CachedSegmentationOverlayState();
}

class _CachedSegmentationOverlayState extends State<CachedSegmentationOverlay> {
  ui.Image? _backgroundImage;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  // Caching for performance
  final Map<String, ui.Picture> _pictureCache = {};
  final Map<String, ui.Image> _maskCache = {};
  String? _lastCacheKey;
  ui.Picture? _cachedPicture;

  static const double _targetBackgroundWidth = 720.0;

  @override
  void initState() {
    super.initState();
    _resolveBackgroundImage();
  }

  @override
  void didUpdateWidget(covariant CachedSegmentationOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    final bgChanged = widget.backgroundAsset != oldWidget.backgroundAsset;
    if (bgChanged) {
      _resolveBackgroundImage();
    }
    final flagsChanged =
        widget.flipHorizontal != oldWidget.flipHorizontal ||
        widget.flipVertical != oldWidget.flipVertical ||
        widget.maskSmoothing != oldWidget.maskSmoothing ||
        widget.backgroundOpacity != oldWidget.backgroundOpacity ||
        widget.maskSourceIsUpsideDown != oldWidget.maskSourceIsUpsideDown;
    if (bgChanged || flagsChanged) {
      _clearCache();
      setState(() {});
    }
  }

  @override
  void dispose() {
    _disposeImageStream();
    _clearCache();
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

  String _generateCacheKey() {
    final detectionsHash = widget.detections.map((d) => d.hashCode).join(',');
    return '${widget.maskThreshold}_${widget.flipHorizontal}_${widget.flipVertical}_${widget.maskSmoothing}_${widget.backgroundOpacity.toStringAsFixed(2)}_${widget.maskSourceIsUpsideDown}_$detectionsHash';
  }

  void _clearCache() {
    for (final picture in _pictureCache.values) {
      picture.dispose();
    }
    _pictureCache.clear();
    _maskCache.clear();
    _cachedPicture?.dispose();
    _cachedPicture = null;
  }

  @override
  Widget build(BuildContext context) {
    if (widget.detections.isEmpty) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _CachedSegmentationMaskPainter(
          detections: widget.detections,
          maskThreshold: widget.maskThreshold,
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
          backgroundImage: _backgroundImage,
          maskSmoothing: widget.maskSmoothing,
          cacheKey: _generateCacheKey(),
          pictureCache: _pictureCache,
          maskCache: _maskCache,
        ),
      ),
    );
  }
}

class _CachedSegmentationMaskPainter extends CustomPainter {
  const _CachedSegmentationMaskPainter({
    required this.detections,
    required this.maskThreshold,
    required this.flipHorizontal,
    required this.flipVertical,
    this.backgroundImage,
    this.maskSmoothing = 0.8,
    this.backgroundOpacity = 1.0,
    this.maskSourceIsUpsideDown = false,
    required this.cacheKey,
    required this.pictureCache,
    required this.maskCache,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final ui.Image? backgroundImage;
  final double maskSmoothing;
  final double backgroundOpacity;
  final bool maskSourceIsUpsideDown;
  final String cacheKey;
  final Map<String, ui.Picture> pictureCache;
  final Map<String, ui.Image> maskCache;

  static final Paint _backgroundPaint = Paint()..color = Colors.black;
  static final Paint _erasePaintBase = Paint()
    ..blendMode = BlendMode.dstOut
    ..isAntiAlias = true;

  @override
  void paint(Canvas canvas, Size size) {
    final layerBounds = Offset.zero & size;
    canvas.save();
    canvas.clipRect(layerBounds);

    final geometry = DetectionViewGeometry.fromDetections(detections, size);
    if (geometry == null) {
      canvas.restore();
      return;
    }

    // Check cache first
    if (pictureCache.containsKey(cacheKey)) {
      final cachedPicture = pictureCache[cacheKey]!;
      canvas.drawPicture(cachedPicture);
      canvas.restore();
      return;
    }

    // Create new picture and cache it
    final recorder = ui.PictureRecorder();
    final pictureCanvas = Canvas(recorder);

    // Use a layer so dstOut properly cuts transparency in the result
    // final layerBounds = Offset.zero & size;
    pictureCanvas.saveLayer(layerBounds, Paint());
    _paintSegmentation(pictureCanvas, size, geometry);
    pictureCanvas.restore();

    final picture = recorder.endRecording();
    pictureCache[cacheKey] = picture;

    // Draw the new picture
    canvas.drawPicture(picture);
    canvas.restore();
  }

  void _paintSegmentation(
    Canvas canvas,
    Size size,
    DetectionViewGeometry geometry,
  ) {
    final sourceWidth = geometry.sourceWidth;
    final sourceHeight = geometry.sourceHeight;
    final scale = geometry.scale;
    final scaledWidth = sourceWidth * scale;
    final scaledHeight = sourceHeight * scale;
    final dx = geometry.dx;
    final dy = geometry.dy;

    const maskSourceIsMirrored = false;
    final effectiveFlipH = maskSourceIsMirrored ? false : flipHorizontal;
    final effectiveFlipV = maskSourceIsUpsideDown ^ flipVertical;

    // Draw background
    if (backgroundImage != null) {
      paintImage(
        canvas: canvas,
        rect: Offset.zero & size,
        image: backgroundImage!,
        fit: BoxFit.cover,
        colorFilter: ColorFilter.mode(
          Colors.white.withOpacity(backgroundOpacity.clamp(0.0, 1.0)),
          BlendMode.modulate,
        ),
      );
    } else {
      final paint = Paint()
        ..color = _backgroundPaint.color.withOpacity(
          backgroundOpacity.clamp(0.0, 1.0),
        );
      canvas.drawRect(Offset.zero & size, paint);
    }

    // Prepare mask painting
    final erasePaint = _erasePaintBase;
    erasePaint.maskFilter = maskSmoothing > 0
        ? ui.MaskFilter.blur(ui.BlurStyle.normal, maskSmoothing)
        : null;

    // Batch mask drawing for better performance
    final maskPaths = <Path>[];

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

      final maskPath = _createMaskPath(
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
      );

      if (maskPath != null) {
        maskPaths.add(maskPath);
      }
    }

    // Draw all masks in one operation
    for (final path in maskPaths) {
      canvas.drawPath(path, erasePaint);
    }
  }

  Path? _createMaskPath({
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
  }) {
    final path = Path();
    bool hasContent = false;

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

        var runEnd = x + 1;
        while (runEnd < endX && row[runEnd] >= threshold) {
          runEnd++;
        }

        final drawStart = flipH ? (maskWidth - runEnd) : x;
        final drawWidth = (runEnd - x) * cellWidth;
        final left = dx + drawStart * cellWidth;

        path.addRect(Rect.fromLTWH(left, top, drawWidth, cellHeight));
        hasContent = true;

        x = runEnd;
      }
    }

    return hasContent ? path : null;
  }

  @override
  bool shouldRepaint(covariant _CachedSegmentationMaskPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.maskThreshold != maskThreshold ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical ||
        oldDelegate.backgroundImage != backgroundImage ||
        oldDelegate.maskSmoothing != maskSmoothing ||
        oldDelegate.cacheKey != cacheKey;
  }

  int _clampIndex(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
  }
}
