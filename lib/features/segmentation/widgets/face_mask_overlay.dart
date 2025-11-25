import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class FaceMaskOverlay extends StatefulWidget {
  const FaceMaskOverlay({
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
  State<FaceMaskOverlay> createState() => _FaceMaskOverlayState();
}

class _FaceMaskOverlayState extends State<FaceMaskOverlay> {
  ui.Image? _maskImage;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  @override
  void initState() {
    super.initState();
    _resolveMaskImage();
  }

  @override
  void didUpdateWidget(covariant FaceMaskOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.maskAsset != oldWidget.maskAsset) {
      debugPrint('Mask asset changed: ${widget.maskAsset}');
      _resolveMaskImage();
    }
    if (widget.poseDetections != oldWidget.poseDetections) {
      // debugPrint('Pose detections updated in overlay: ${widget.poseDetections.length}');
    }
  }

  @override
  void dispose() {
    _disposeImageStream();
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

  @override
  Widget build(BuildContext context) {
    if (widget.poseDetections.isEmpty || _maskImage == null) {
      return const SizedBox.shrink();
    }

    return IgnorePointer(
      child: CustomPaint(
        painter: _FaceMaskPainter(
          poseDetections: widget.poseDetections,
          maskImage: _maskImage!,
          flipHorizontal: widget.flipHorizontal,
          flipVertical: widget.flipVertical,
        ),
      ),
    );
  }
}

class _FaceMaskPainter extends CustomPainter {
  const _FaceMaskPainter({
    required this.poseDetections,
    required this.maskImage,
    required this.flipHorizontal,
    required this.flipVertical,
  });

  final List<YOLOResult> poseDetections;
  final ui.Image maskImage;
  final bool flipHorizontal;
  final bool flipVertical;

  @override
  void paint(Canvas canvas, Size size) {
    if (poseDetections.isEmpty) {
      debugPrint('Painter: No pose detections');
      return;
    }

    // Pose outputs arrive upside down from the single-image predictor; undo
    // that with a vertical flip unless the caller explicitly disables it.
    const poseSourceIsUpsideDown = true;
    const poseSourceIsMirrored = false;
    final effectiveFlipH = poseSourceIsMirrored
        ? !flipHorizontal
        : flipHorizontal;
    final effectiveFlipV = poseSourceIsUpsideDown ^ flipVertical;
    debugPrint(
      'Painter: Painting mask for ${poseDetections.length} detections. MaskImage: ${maskImage.width}x${maskImage.height}',
    );

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
      debugPrint(
        'Missing landmarks: nose=$nose, leftEye=$leftEye, rightEye=$rightEye',
      );
      return;
    }

    debugPrint('Drawing mask at nose: $nose');

    // Distance between eyes for scaling
    final eyeDist = sqrt(
      pow(rightEye.dx - leftEye.dx, 2) + pow(rightEye.dy - leftEye.dy, 2),
    );

    // Rotation angle
    final angle = atan2(rightEye.dy - leftEye.dy, rightEye.dx - leftEye.dx);

    // Scale factor (adjust multiplier as needed for mask size relative to face)
    final scale = eyeDist * 4.0;
    final maskW = maskImage.width.toDouble();
    final maskH = maskImage.height.toDouble();
    final scaleFactor = scale / maskW;

    canvas.save();

    // Translate to nose position (or slightly above for eyes)
    // Adjust Y offset based on mask type if needed, here we center on nose/eyes
    canvas.translate(nose.dx, nose.dy);
    canvas.drawCircle(
      Offset.zero,
      6,
      Paint()
        ..color = Colors.redAccent
        ..style = PaintingStyle.fill,
    );

    // Rotate
    canvas.rotate(angle);

    // Scale (keep texture orientation; flips are handled in coordinate mapping)
    canvas.scale(scaleFactor);

    // Draw centered
    canvas.drawImage(
      maskImage,
      Offset(-maskW / 2, -maskH / 2),
      Paint()..filterQuality = FilterQuality.medium,
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

    // Keypoints arrive normalized (0..1). Convert to image pixel space when
    // values look normalized; otherwise assume they are already in pixels.
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
  bool shouldRepaint(covariant _FaceMaskPainter oldDelegate) {
    return oldDelegate.poseDetections != poseDetections ||
        oldDelegate.maskImage != maskImage ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical;
  }
}
