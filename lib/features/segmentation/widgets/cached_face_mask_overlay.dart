import 'dart:math';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class CachedFaceMaskOverlay extends StatefulWidget {
  const CachedFaceMaskOverlay({
    super.key,
    required this.poseDetections,
    required this.maskAsset,
    required this.flipHorizontal,
    required this.flipVertical,
  });

  final List<YOLOResult> poseDetections;
  final String? maskAsset;
  final bool flipHorizontal;
  final bool flipVertical;

  @override
  State<CachedFaceMaskOverlay> createState() => _CachedFaceMaskOverlayState();
}

class _CachedFaceMaskOverlayState extends State<CachedFaceMaskOverlay> {
  ui.Image? _maskImage;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;
  
  // Performance caching
  final Map<String, ui.Picture> _pictureCache = {};
  String? _lastCacheKey;
  ui.Picture? _cachedPicture;

  @override
  void initState() {
    super.initState();
    _resolveMaskImage();
  }

  @override
  void didUpdateWidget(covariant CachedFaceMaskOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.maskAsset != oldWidget.maskAsset) {
      _resolveMaskImage();
    }
  }

  @override
  void dispose() {
    _disposeImageStream();
    _clearCache();
    super.dispose();
  }

  void _resolveMaskImage() {
    _disposeImageStream();
    final asset = widget.maskAsset;
    if (asset == null) {
      setState(() => _maskImage = null);
      return;
    }
    final imageProvider = AssetImage(asset);
    final stream = imageProvider.resolve(ImageConfiguration.empty);
    _imageStream = stream;
    _imageStreamListener = ImageStreamListener((imageInfo, _) {
      setState(() {
        _maskImage = imageInfo.image;
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

  void _clearCache() {
    for (final picture in _pictureCache.values) {
      picture.dispose();
    }
    _pictureCache.clear();
    _cachedPicture?.dispose();
    _cachedPicture = null;
  }

  String _generateCacheKey() {
    if (widget.poseDetections.isEmpty || _maskImage == null) return '';
    
    final detection = widget.poseDetections.first;
    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.length < 3) return '';
    
    // Create cache key based on keypoint positions and mask asset
    final nosePos = '${keypoints[0].x.toStringAsFixed(2)}_${keypoints[0].y.toStringAsFixed(2)}';
    final leftEyePos = '${keypoints[1].x.toStringAsFixed(2)}_${keypoints[1].y.toStringAsFixed(2)}';
    final rightEyePos = '${keypoints[2].x.toStringAsFixed(2)}_${keypoints[2].y.toStringAsFixed(2)}';
    
    return '${widget.maskAsset}_${widget.flipHorizontal}_${widget.flipVertical}_${nosePos}_${leftEyePos}_${rightEyePos}';
  }

  @override
  Widget build(BuildContext context) {
    if (widget.poseDetections.isEmpty || _maskImage == null) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _CachedFaceMaskPainter(
          poseDetections: widget.poseDetections,
          maskImage: _maskImage!,
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
          cacheKey: _generateCacheKey(),
          pictureCache: _pictureCache,
        ),
      ),
    );
  }
}

class _CachedFaceMaskPainter extends CustomPainter {
  const _CachedFaceMaskPainter({
    required this.poseDetections,
    required this.maskImage,
    required this.flipHorizontal,
    required this.flipVertical,
    required this.cacheKey,
    required this.pictureCache,
  });

  final List<YOLOResult> poseDetections;
  final ui.Image maskImage;
  final bool flipHorizontal;
  final bool flipVertical;
  final String cacheKey;
  final Map<String, ui.Picture> pictureCache;

  @override
  void paint(Canvas canvas, Size size) {
    if (poseDetections.isEmpty || cacheKey.isEmpty) {
      return;
    }

    // Check cache first
    if (pictureCache.containsKey(cacheKey)) {
      final cachedPicture = pictureCache[cacheKey]!;
      canvas.drawPicture(cachedPicture);
      return;
    }

    // Create new picture and cache it
    final recorder = ui.PictureRecorder();
    final pictureCanvas = Canvas(recorder);
    
    _paintMask(pictureCanvas, size);
    
    final picture = recorder.endRecording();
    pictureCache[cacheKey] = picture;
    
    // Limit cache size to prevent memory leaks
    if (pictureCache.length > 50) {
      final oldestKey = pictureCache.keys.first;
      pictureCache[oldestKey]?.dispose();
      pictureCache.remove(oldestKey);
    }
    
    // Draw the new picture
    canvas.drawPicture(picture);
  }

  void _paintMask(Canvas canvas, Size size) {
    const poseSourceIsUpsideDown = true;
    const poseSourceIsMirrored = false;
    final effectiveFlipH = poseSourceIsMirrored
        ? !flipHorizontal
        : flipHorizontal;
    final effectiveFlipV = poseSourceIsUpsideDown ^ flipVertical;

    // Use the first detected person
    final detection = poseDetections.first;
    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.isEmpty) return;

    // Map keypoints to screen coordinates
    final nose = _mapPosePoint(
      detection,
      0,
      size,
      effectiveFlipH,
      effectiveFlipV,
    );
    final leftEye = _mapPosePoint(
      detection,
      1,
      size,
      effectiveFlipH,
      effectiveFlipV,
    );
    final rightEye = _mapPosePoint(
      detection,
      2,
      size,
      effectiveFlipH,
      effectiveFlipV,
    );

    if (nose == null || leftEye == null || rightEye == null) {
      return;
    }

    // Calculate transformation parameters
    final eyeDist = sqrt(
      pow(rightEye.dx - leftEye.dx, 2) + pow(rightEye.dy - leftEye.dy, 2),
    );
    final angle = atan2(rightEye.dy - leftEye.dy, rightEye.dx - leftEye.dx);
    const maskRotationOffset = pi;
    final appliedAngle = angle + maskRotationOffset;

    final scale = eyeDist * 4.5;
    final maskW = maskImage.width.toDouble();
    final maskH = maskImage.height.toDouble();
    final scaleFactor = scale / maskW;
    final eyeMid = Offset(
      (leftEye.dx + rightEye.dx) / 2,
      (leftEye.dy + rightEye.dy) / 2,
    );
    final anchorPoint = Offset(
      nose.dx - (maskW * scaleFactor * 0.05),
      eyeMid.dy + (maskH * scaleFactor * 0.10),
    );

    // Apply transformations
    canvas.save();
    canvas.translate(anchorPoint.dx, anchorPoint.dy);
    canvas.rotate(appliedAngle);
    canvas.scale(scaleFactor);

    // Draw mask with optimized quality
    canvas.drawImage(
      maskImage,
      Offset(-maskW / 2, -maskH / 2),
      Paint()
        ..filterQuality = FilterQuality.medium
        ..isAntiAlias = true,
    );

    canvas.restore();
  }

  Offset? _mapPosePoint(
    YOLOResult detection,
    int index,
    Size viewSize,
    bool flipH,
    bool flipV,
  ) {
    final keypoints = detection.keypoints;
    if (keypoints == null || index < 0 || index >= keypoints.length) {
      return null;
    }

    final point = keypoints[index];
    if (!point.x.isFinite || !point.y.isFinite) return null;

    double? imageW = detection.imageSize?.width;
    double? imageH = detection.imageSize?.height;

    if (imageW == null || imageH == null) {
      final box = detection.boundingBox;
      final nBox = detection.normalizedBox;
      if (box.width <= 0 || nBox.width <= 0) return null;

      imageW = box.width / nBox.width;
      imageH = box.height / nBox.height;
    }

    var kpX = point.x;
    var kpY = point.y;
    if (kpX.abs() <= 1.2 && kpY.abs() <= 1.2) {
      kpX = kpX * imageW;
      kpY = kpY * imageH;
    }

    final scale = max(viewSize.width / imageW, viewSize.height / imageH);
    final scaledW = imageW * scale;
    final scaledH = imageH * scale;
    final dx = (viewSize.width - scaledW) / 2.0;
    final dy = (viewSize.height - scaledH) / 2.0;

    double screenX = (kpX * scale) + dx;
    double screenY = (kpY * scale) + dy;

    if (flipH) {
      screenX = viewSize.width - screenX;
    }

    if (flipV) {
      screenY = viewSize.height - screenY;
    }

    return Offset(screenX, screenY);
  }

  @override
  bool shouldRepaint(covariant _CachedFaceMaskPainter oldDelegate) {
    return oldDelegate.poseDetections != poseDetections ||
        oldDelegate.maskImage != maskImage ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical ||
        oldDelegate.cacheKey != cacheKey;
  }
}