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
              Text('Overlay mode', style: theme.textTheme.labelLarge),
              const SizedBox(width: 12),
              Expanded(
                child: DropdownButton<SegmentationOverlayMode>(
                  value: controller.overlayMode,
                  isExpanded: true,
                  onChanged: (mode) {
                    if (mode != null) controller.setOverlayMode(mode);
                  },
                  items: const [
                    DropdownMenuItem(
                      value: SegmentationOverlayMode.backgroundReplacement,
                      child: Text('Background replacement'),
                    ),
                    DropdownMenuItem(
                      value: SegmentationOverlayMode.maskOnly,
                      child: Text('Mask only'),
                    ),
                    DropdownMenuItem(
                      value: SegmentationOverlayMode.combined,
                      child: Text('Combined'),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
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
