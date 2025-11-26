import 'package:flutter/material.dart';
import 'package:test_mask_mesh_ultralitics/features/segmentation/view/optimized_video_segmentation_screen.dart';

import '../controller/segmentation_controller.dart';
import '../widgets/segmentation_camera_view.dart';
import '../widgets/segmentation_controls.dart';

class SegmentationScreen extends StatefulWidget {
  const SegmentationScreen({super.key});

  @override
  State<SegmentationScreen> createState() => _SegmentationScreenState();
}

class _SegmentationScreenState extends State<SegmentationScreen> {
  late final SegmentationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = SegmentationController();
    _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Camera Segmentation'),
        backgroundColor: Colors.black,
        actions: [
          IconButton(
            icon: const Icon(Icons.video_library),
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (context) => OptimizedVideoSegmentationScreen(),
                ),
              );
            },
          ),
        ],
      ),
      body: SafeArea(
        child: AnimatedBuilder(
          animation: _controller,
          builder: (context, _) {
            if (_controller.isUnsupportedPlatform) {
              return _UnsupportedPlatform(status: _controller.statusMessage);
            }

            return Column(
              children: [
                Expanded(
                  child: SegmentationCameraView(controller: _controller),
                ),
                SegmentationControls(controller: _controller),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _UnsupportedPlatform extends StatelessWidget {
  const _UnsupportedPlatform({required this.status});

  final String status;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.devices_other, size: 64, color: Colors.white70),
            const SizedBox(height: 16),
            Text(
              status,
              style: Theme.of(context).textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
