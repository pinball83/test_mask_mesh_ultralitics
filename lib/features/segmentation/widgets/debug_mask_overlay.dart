import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class DebugMaskOverlay extends StatelessWidget {
  const DebugMaskOverlay({
    super.key,
    required this.detections,
    this.maskThreshold = 0.5,
    this.flipHorizontal = false,
    this.flipVertical = false,
    this.maskSourceIsUpsideDown = false,
    this.color = const Color(0xFF00FFFF),
    this.opacity = 0.25,
  });

  final List<YOLOResult> detections;
  final double maskThreshold;
  final bool flipHorizontal;
  final bool flipVertical;
  final bool maskSourceIsUpsideDown;
  final Color color;
  final double opacity;

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) return const SizedBox.shrink();
    return IgnorePointer(
      child: CustomPaint(
        painter: _DebugMaskPainter(
          detections: detections,
          threshold: maskThreshold,
          flipH: flipHorizontal,
          flipV: maskSourceIsUpsideDown ^ flipVertical,
          color: color.withOpacity(opacity.clamp(0.0, 1.0)),
        ),
      ),
    );
  }
}

class _DebugMaskPainter extends CustomPainter {
  const _DebugMaskPainter({
    required this.detections,
    required this.threshold,
    required this.flipH,
    required this.flipV,
    required this.color,
  });

  final List<YOLOResult> detections;
  final double threshold;
  final bool flipH;
  final bool flipV;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill
      ..isAntiAlias = false;

    for (final d in detections) {
      final mask = d.mask;
      final bounds = d.boundingBox;
      if (mask == null || mask.isEmpty || bounds.isEmpty) continue;

      // Infer source image size from normalized box and absolute box
      final n = d.normalizedBox;
      if (n.isEmpty) continue;

      final imageW = bounds.width / n.width;
      final imageH = bounds.height / n.height;
      if (imageW <= 0 || imageH <= 0) continue;

      final scale = _max(size.width / imageW, size.height / imageH);
      final scaledW = imageW * scale;
      final scaledH = imageH * scale;
      final dx = (size.width - scaledW) / 2.0;
      final dy = (size.height - scaledH) / 2.0;

      final mh = mask.length;
      final mw = mask.first.length;
      if (mw == 0 || mh == 0) continue;

      final cellW = scaledW / mw;
      final cellH = scaledH / mh;

      // Subset by normalized box for speed
      var sx = (n.left.clamp(0.0, 1.0) * mw).floor();
      var ex = (n.right.clamp(0.0, 1.0) * mw).ceil();
      var sy = (n.top.clamp(0.0, 1.0) * mh).floor();
      var ey = (n.bottom.clamp(0.0, 1.0) * mh).ceil();
      if (sx < 0) sx = 0;
      if (sy < 0) sy = 0;
      if (ex > mw) ex = mw;
      if (ey > mh) ey = mh;

      for (var y = sy; y < ey; y++) {
        final row = mask[y];
        final mappedY = flipV ? (mh - 1 - y) : y;
        final top = dy + mappedY * cellH;
        var x = sx;
        while (x < ex) {
          if (row[x] < threshold) {
            x++;
            continue;
          }
          var runEnd = x + 1;
          while (runEnd < ex && row[runEnd] >= threshold) {
            runEnd++;
          }
          final drawStart = flipH ? (mw - runEnd) : x;
          final left = dx + drawStart * cellW;
          final w = (runEnd - x) * cellW;
          canvas.drawRect(Rect.fromLTWH(left, top, w, cellH), paint);
          x = runEnd;
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant _DebugMaskPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.threshold != threshold ||
        oldDelegate.flipH != flipH ||
        oldDelegate.flipV != flipV ||
        oldDelegate.color != color;
  }

  double _max(double a, double b) => a > b ? a : b;
}

