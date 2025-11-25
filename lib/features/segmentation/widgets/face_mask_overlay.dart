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
      _resolveMaskImage();
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
    if (poseDetections.isEmpty) return;

    // Use the first detected person
    final detection = poseDetections.first;
    final keypoints = detection.keypoints;
    if (keypoints == null || keypoints.isEmpty) return;

    // Map keypoints to screen coordinates
    final nose = _mapPosePoint(detection, 0, size);
    final leftEye = _mapPosePoint(detection, 1, size);
    final rightEye = _mapPosePoint(detection, 2, size);

    if (nose == null || leftEye == null || rightEye == null) return;

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

    // Rotate
    canvas.rotate(angle);

    // Scale
    canvas.scale(scaleFactor);

    // Draw centered
    canvas.drawImage(
      maskImage,
      Offset(-maskW / 2, -maskH / 2),
      Paint()..filterQuality = FilterQuality.medium,
    );

    canvas.restore();
  }

  Offset? _mapPosePoint(YOLOResult detection, int index, Size viewSize) {
    final keypoints = detection.keypoints;
    if (keypoints == null || index < 0 || index >= keypoints.length) {
      return null;
    }

    final point = keypoints[index];
    if (!point.x.isFinite || !point.y.isFinite) return null;

    if (detection.imageSize != null) {
      final imageSize = detection.imageSize!;

      // Calculate Aspect Fill scale and offset
      final double scaleX = viewSize.width / imageSize.width;
      final double scaleY = viewSize.height / imageSize.height;
      final double scale = max(scaleX, scaleY);

      final double scaledW = imageSize.width * scale;
      final double scaledH = imageSize.height * scale;
      final double dx = (viewSize.width - scaledW) / 2.0;
      final double dy = (viewSize.height - scaledH) / 2.0;

      double screenX = (point.x * scale) + dx;
      double screenY = (point.y * scale) + dy;

      if (flipHorizontal) {
        screenX = viewSize.width - screenX;
      }

      // Vertical flip is handled by the coordinate system or caller
      if (flipVertical) {
        screenY = viewSize.height - screenY;
      }

      return Offset(screenX, screenY);
    }

    // Fallback if imageSize is missing (shouldn't happen with correct YOLO setup)
    return null;
  }

  @override
  bool shouldRepaint(covariant _FaceMaskPainter oldDelegate) {
    return oldDelegate.poseDetections != poseDetections ||
        oldDelegate.maskImage != maskImage ||
        oldDelegate.flipHorizontal != flipHorizontal ||
        oldDelegate.flipVertical != flipVertical;
  }
}
