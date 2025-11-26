import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';
import '../controller/optimized_video_segmentation_controller.dart';
import '../widgets/cached_segmentation_overlay.dart';
import '../widgets/cached_face_mask_overlay.dart';
import '../widgets/debug_mask_overlay.dart';

class OptimizedVideoSegmentationScreen extends StatefulWidget {
  const OptimizedVideoSegmentationScreen({super.key});

  @override
  State<OptimizedVideoSegmentationScreen> createState() =>
      _OptimizedVideoSegmentationScreenState();
}

class _OptimizedVideoSegmentationScreenState
    extends State<OptimizedVideoSegmentationScreen> {
  late OptimizedVideoSegmentationController _controller;

  // UI State
  bool _isLoading = true;
  String? _errorMessage;
  String _statusMessage = '';
  Timer? _fpsUpdateTimer;
  double _currentFps = 0.0;
  final int _targetFps = 30;

  @override
  void initState() {
    super.initState();
    _controller = OptimizedVideoSegmentationController();
    _controller.addListener(_onControllerChanged);
    _initializeVideo();
    _startFpsMonitoring();
  }

  @override
  void dispose() {
    _fpsUpdateTimer?.cancel();
    _controller.removeListener(_onControllerChanged);
    _controller.dispose();
    super.dispose();
  }

  void _onControllerChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  void _startFpsMonitoring() {
    _fpsUpdateTimer = Timer.periodic(Duration(milliseconds: 500), (_) {
      if (mounted) {
        setState(() {
          _currentFps = _controller.fpsCounter.getAverageFps();
        });
      }
    });
  }

  Future<void> _initializeVideo() async {
    try {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });

      final tempDir = await getTemporaryDirectory();
      final videoFile = File('${tempDir.path}/sample_video.mp4');
      // Prepare video and frames in controller, then start playback (video + processing)
      await _controller.initializeFromFile(videoFile, targetFps: _targetFps);
      _controller.startPlayback();

      setState(() {
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Failed to initialize video: $e';
      });
    }
  }

  // Frame extraction moved into controller

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Initializing video segmentation...'),
            ],
          ),
        ),
      );
    }

    if (_errorMessage != null) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(_errorMessage!),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _initializeVideo,
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Optimized Video Segmentation'),
        actions: [_buildPerformanceIndicator(), _buildControls()],
      ),
      body: Stack(
        children: [
          // Video player widget (replace with your actual video player)
          _buildVideoPlayer(),

          // Segmentation overlay
          Positioned.fill(
            child: ListenableBuilder(
              listenable: _controller,
              builder: (context, child) {
                return CachedSegmentationOverlay(
                  detections: _controller.currentDetections,
                  maskThreshold: 0.5,
                  flipHorizontal: false,
                  flipVertical: false,
                  backgroundAsset: 'assets/images/bg_image.jpg',
                  maskSmoothing: 1.0,
                  backgroundOpacity: 0.85,
                  maskSourceIsUpsideDown: false,
                );
              },
            ),
          ),

          // Debug: raw mask overlay (tinted)
          // Positioned.fill(
          //   child: ListenableBuilder(
          //     listenable: _controller,
          //     builder: (context, child) {
          //       return DebugMaskOverlay(
          //         detections: _controller.currentDetections,
          //         maskThreshold: 0.5,
          //         flipHorizontal: false,
          //         flipVertical: false,
          //         maskSourceIsUpsideDown: false,
          //         color: Colors.cyan,
          //         opacity: 0.20,
          //       );
          //     },
          //   ),
          // ),

          // Face mask overlay
          Positioned.fill(
            child: ListenableBuilder(
              listenable: _controller,
              builder: (context, child) {
                return CachedFaceMaskOverlay(
                  poseDetections: _controller.currentPoseDetections,
                  maskAsset: 'assets/images/masks/m_cat.png',
                  flipHorizontal: false,
                  flipVertical: false,
                  opacity: 0.7,
                  poseSourceIsUpsideDown: false,
                  maskRotationOffset: 3.141592653589793,
                );
              },
            ),
          ),

          // Performance overlay
          Positioned(top: 16, left: 16, child: _buildPerformanceOverlay()),
        ],
      ),
      bottomNavigationBar: _buildPlaybackControls(),
    );
  }

  Widget _buildVideoPlayer() {
    final vc = _controller.videoController;
    if (vc == null || !vc.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Center(
      child: AspectRatio(
        aspectRatio: vc.value.aspectRatio,
        child: VideoPlayer(vc),
      ),
    );
  }

  Widget _buildPerformanceIndicator() {
    return Padding(
      padding: const EdgeInsets.only(right: 16.0),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: _currentFps >= 25 ? Colors.green : Colors.orange,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          '${_currentFps.toStringAsFixed(1)} FPS',
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  Widget _buildControls() {
    return PopupMenuButton<String>(
      icon: const Icon(Icons.more_vert),
      onSelected: _handleMenuSelection,
      itemBuilder: (context) => [
        const PopupMenuItem(value: 'quality', child: Text('Adjust Quality')),
        const PopupMenuItem(value: 'buffer', child: Text('Buffer Settings')),
        const PopupMenuItem(value: 'debug', child: Text('Debug Info')),
      ],
    );
  }

  Widget _buildPerformanceOverlay() {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.7),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'FPS: ${_currentFps.toStringAsFixed(1)}',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
          Text(
            'Frame: ${_controller.currentFrameIndex}',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
          Text(
            'Status: ${_controller.isPlaying ? 'Playing' : 'Paused'}',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildPlaybackControls() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          IconButton(
            onPressed: _controller.isPlaying
                ? null
                : () => _controller.seekToFrame(0),
            icon: const Icon(Icons.first_page),
          ),
          IconButton(
            onPressed: _controller.isPlaying
                ? () => _controller.pausePlayback()
                : () => _controller.startPlayback(),
            icon: Icon(_controller.isPlaying ? Icons.pause : Icons.play_arrow),
            iconSize: 48,
          ),
          IconButton(
            onPressed: _controller.isPlaying
                ? null
                : () => _controller.seekToFrame(999999),
            icon: const Icon(Icons.last_page),
          ),
        ],
      ),
    );
  }

  void _handleMenuSelection(String value) {
    switch (value) {
      case 'quality':
        _showQualityDialog();
        break;
      case 'buffer':
        _showBufferDialog();
        break;
      case 'debug':
        _showDebugDialog();
        break;
    }
  }

  void _showQualityDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Quality Settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Frame Duration:'),
            Slider(
              value: _controller.frameDuration.inMilliseconds.toDouble(),
              min: 16,
              max: 100,
              divisions: 5,
              label: '${_controller.frameDuration.inMilliseconds}ms',
              onChanged: (value) {
                _controller.setFrameDuration(
                  Duration(milliseconds: value.round()),
                );
              },
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _showBufferDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Buffer Settings'),
        content: const Text(
          'Buffer management is automatically optimized for performance.\n\n'
          '• Frame queue: 3 frames\n'
          '• Pre-processing: 3 frames ahead\n'
          '• Cache limit: 10 frames',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _showDebugDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Debug Information'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Current FPS: ${_currentFps.toStringAsFixed(2)}'),
            Text('Frame Index: ${_controller.currentFrameIndex}'),
            Text('Is Playing: ${_controller.isPlaying}'),
            Text('Detections: ${_controller.currentDetections.length}'),
            Text(
              'Pose Detections: ${_controller.currentPoseDetections.length}',
            ),
            const SizedBox(height: 16),
            const Text(
              'Performance Tips:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const Text('• FPS > 25: Good performance'),
            const Text('• FPS 15-25: Acceptable'),
            const Text('• FPS < 15: Needs optimization'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}
