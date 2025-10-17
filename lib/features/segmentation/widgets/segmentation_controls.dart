import 'package:flutter/material.dart';

import '../controller/segmentation_controller.dart';

class SegmentationControls extends StatelessWidget {
  const SegmentationControls({super.key, required this.controller});

  final SegmentationController controller;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
      decoration: const BoxDecoration(
        color: Color(0xFF101010),
        boxShadow: [
          BoxShadow(
            color: Colors.black45,
            blurRadius: 12,
            offset: Offset(0, -4),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: SwitchListTile.adaptive(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Show segmentation mask'),
                  value: controller.showMasks,
                  onChanged: (_) => controller.toggleMasks(),
                ),
              ),
              IconButton(
                tooltip: controller.flipMaskHorizontal
                    ? 'Disable horizontal mirror'
                    : 'Mirror mask horizontally',
                isSelected: controller.flipMaskHorizontal,
                onPressed: controller.toggleMaskHorizontalFlip,
                icon: const Icon(Icons.swap_horiz),
                selectedIcon: const Icon(Icons.swap_horiz),
              ),
              IconButton(
                tooltip: controller.flipMaskVertical
                    ? 'Disable vertical mirror'
                    : 'Mirror mask vertically',
                isSelected: controller.flipMaskVertical,
                onPressed: controller.toggleMaskVerticalFlip,
                icon: const Icon(Icons.swap_vert),
                selectedIcon: const Icon(Icons.swap_vert),
              ),
              IconButton(
                tooltip: 'Flip camera',
                onPressed: controller.flipCamera,
                icon: const Icon(Icons.flip_camera_android_outlined),
              ),
              IconButton(
                tooltip: 'Zoom out',
                onPressed: controller.zoomOut,
                icon: const Icon(Icons.zoom_out),
              ),
              IconButton(
                tooltip: 'Zoom in',
                onPressed: controller.zoomIn,
                icon: const Icon(Icons.zoom_in),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Mask threshold (${controller.maskThreshold.toStringAsFixed(2)})',
            style: theme.textTheme.labelLarge,
          ),
          Slider(
            value: controller.maskThreshold.clamp(0.1, 0.95),
            min: 0.1,
            max: 0.95,
            onChanged: controller.updateMaskThreshold,
          ),
          const SizedBox(height: 12),
          Text(
            'Confidence threshold (${controller.confidenceThreshold.toStringAsFixed(2)})',
            style: theme.textTheme.labelLarge,
          ),
          Slider(
            value: controller.confidenceThreshold.clamp(0.1, 0.95),
            min: 0.1,
            max: 0.95,
            onChanged: (value) => controller.updateConfidence(value),
          ),
        ],
      ),
    );
  }
}
