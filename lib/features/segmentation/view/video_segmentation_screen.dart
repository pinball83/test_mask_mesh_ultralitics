import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:ffmpeg_kit_min_gpl/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_min_gpl/return_code.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:video_player/video_player.dart';

import '../controller/segmentation_controller.dart';
import '../widgets/segmentation_overlay.dart';

class VideoSegmentationScreen extends StatefulWidget {
  const VideoSegmentationScreen({super.key, required this.controller});

  final SegmentationController controller;

  @override
  State<VideoSegmentationScreen> createState() =>
      _VideoSegmentationScreenState();
}

class _VideoSegmentationScreenState extends State<VideoSegmentationScreen> {
  VideoPlayerController? _videoController;
  List<String> _framePaths = [];
  int _currentFrameIndex = 0;
  Timer? _playbackTimer;
  bool _isProcessing = false;
  String? _statusMessage;
  List<YOLOResult> _currentDetections = [];
  ui.Image? _currentFrameImage;
  bool _isInferring = false;
  String? _selectedBackground = 'assets/images/bg_image.jpg';
  String? _selectedMask;

  // FPS control
  static const int _targetFps = 30;
  static const Duration _frameDuration = Duration(
    milliseconds: 1000 ~/ _targetFps,
  );

  // Inference optimization: run inference every N frames
  static const int _inferenceInterval = 2;
  int _framesSinceLastInference = 0;

  YOLO? _yolo;

  @override
  void initState() {
    super.initState();
    _initializeYolo();
    _initializeVideo();
  }

  Future<void> _initializeYolo() async {
    final modelPath = widget.controller.modelPath;
    if (modelPath != null) {
      _yolo = YOLO(modelPath: modelPath, task: YOLOTask.segment, useGpu: true);
      await _yolo!.loadModel();
    }
  }

  Future<void> _initializeVideo() async {
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Preparing video...';
    });

    try {
      // 1. Copy asset to temp file
      final tempDir = await getTemporaryDirectory();
      final videoFile = File('${tempDir.path}/sample_video.mp4');

      // Check if asset exists, if not use a placeholder or error
      try {
        final byteData = await rootBundle.load(
          'assets/videos/sample_video.mp4',
        );
        await videoFile.writeAsBytes(byteData.buffer.asUint8List());
      } catch (e) {
        setState(() {
          _isProcessing = false;
          _statusMessage =
              'Video asset not found. Please add assets/videos/sample_video.mp4';
        });
        return;
      }

      // 2. Initialize video player for reference (optional, mostly for audio if needed)
      _videoController = VideoPlayerController.file(videoFile);
      await _videoController!.initialize();

      // 3. Extract frames using FFmpeg
      setState(() => _statusMessage = 'Extracting frames...');
      final framesDir = Directory('${tempDir.path}/frames');
      if (await framesDir.exists()) {
        await framesDir.delete(recursive: true);
      }
      await framesDir.create();

      // Extract at 30fps, scale to 640 width (maintain aspect ratio) for performance
      final command =
          '-i ${videoFile.path} -vf "fps=$_targetFps,scale=640:-1" ${framesDir.path}/frame_%04d.jpg';
      final session = await FFmpegKit.execute(command);
      final returnCode = await session.getReturnCode();

      if (ReturnCode.isSuccess(returnCode)) {
        final files = await framesDir.list().toList();
        _framePaths =
            files
                .whereType<File>()
                .map((f) => f.path)
                .where((p) => p.endsWith('.jpg'))
                .toList()
              ..sort();

        setState(() {
          _isProcessing = false;
          _statusMessage = null;
        });

        _startPlayback();
      } else {
        setState(() {
          _isProcessing = false;
          _statusMessage = 'Failed to extract frames';
        });
      }
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _statusMessage = 'Error: $e';
      });
    }
  }

  void _startPlayback() {
    _playbackTimer?.cancel();
    _currentFrameIndex = 0;
    _framesSinceLastInference = 0;
    _playbackTimer = Timer.periodic(_frameDuration, (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      _processNextFrame();
    });
  }

  Future<void> _processNextFrame() async {
    if (_framePaths.isEmpty || _yolo == null) return;

    final framePath = _framePaths[_currentFrameIndex];
    final frameFile = File(framePath);
    final frameBytes = await frameFile.readAsBytes();

    // Decode image for display
    final codec = await ui.instantiateImageCodec(frameBytes);
    final frameInfo = await codec.getNextFrame();
    final image = frameInfo.image;

    // Update displayed frame immediately
    if (mounted) {
      setState(() {
        _currentFrameImage = image;
      });
    }

    // Run inference asynchronously, but only if:
    // 1. No inference is currently running
    // 2. It's time for inference (based on interval)
    _framesSinceLastInference++;
    if (!_isInferring && _framesSinceLastInference >= _inferenceInterval) {
      _framesSinceLastInference = 0;
      _isInferring = true;

      // Run inference without blocking (fire and forget)
      _runInferenceAsync(frameBytes);
    }

    // Advance to next frame
    if (mounted) {
      setState(() {
        _currentFrameIndex = (_currentFrameIndex + 1) % _framePaths.length;
      });
    }
  }

  Future<void> _runInferenceAsync(Uint8List frameBytes) async {
    try {
      final result = await _yolo!.predict(
        frameBytes,
        confidenceThreshold: 0.75,
      );
      final detections = (result['detections'] as List)
          .map((d) => YOLOResult.fromMap(d))
          .toList();

      if (mounted) {
        setState(() {
          _currentDetections = detections;
        });
      }
    } catch (e) {
      debugPrint('Inference error: $e');
    } finally {
      _isInferring = false;
    }
  }

  void _showBackgroundSelector() async {
    final backgrounds = [
      'assets/images/bg_image.jpg',
      'assets/images/backgrounds/b_blur_on.png',
      'assets/images/backgrounds/b_bookshelf.jpeg',
      'assets/images/backgrounds/b_bubbles.jpeg',
      'assets/images/backgrounds/b_cafe.jpeg',
      'assets/images/backgrounds/b_cosy_street.jpeg',
      'assets/images/backgrounds/b_fantasy.jpeg',
      'assets/images/backgrounds/b_forest.jpeg',
      'assets/images/backgrounds/b_harry_p.jpeg',
      'assets/images/backgrounds/b_loft.jpeg',
      'assets/images/backgrounds/b_mountains.jpeg',
      'assets/images/backgrounds/b_neon.jpeg',
      'assets/images/backgrounds/b_ocean.jpeg',
      'assets/images/backgrounds/b_pedestrian_passage.jpeg',
      'assets/images/backgrounds/b_rug.jpeg',
      'assets/images/backgrounds/b_stars.jpeg',
      'assets/images/backgrounds/b_valentine.jpeg',
      'assets/images/backgrounds/b_vysotka.jpeg',
    ];

    showModalBottomSheet(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) => Container(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Select Background',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              SizedBox(
                height: 100,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: backgrounds.length,
                  itemBuilder: (context, index) {
                    final bg = backgrounds[index];
                    final isSelected = _selectedBackground == bg;
                    return GestureDetector(
                      onTap: () {
                        // Update main widget state for preview
                        setState(() {
                          _selectedBackground = bg;
                        });
                        // Update modal state for selection border
                        setModalState(() {});
                      },
                      child: Container(
                        width: 100,
                        margin: const EdgeInsets.only(right: 8),
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: isSelected ? Colors.blue : Colors.grey,
                            width: isSelected ? 3 : 1,
                          ),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.asset(bg, fit: BoxFit.cover),
                        ),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  void _showMaskSelector() async {
    final masks = [
      null, // No mask option
      'assets/images/masks/m_butterfly_hearts.png',
      'assets/images/masks/m_cat.png',
      'assets/images/masks/m_death_w_coat.png',
      'assets/images/masks/m_diablo.png',
      'assets/images/masks/m_harry_p.png',
      'assets/images/masks/m_nymb.png',
      'assets/images/masks/m_peaky_blinders.png',
      'assets/images/masks/m_pirate.png',
      'assets/images/masks/m_santa.png',
    ];

    showModalBottomSheet(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) => Container(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Select Mask',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              SizedBox(
                height: 100,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: masks.length,
                  itemBuilder: (context, index) {
                    final mask = masks[index];
                    final isSelected = _selectedMask == mask;
                    return GestureDetector(
                      onTap: () {
                        // Update main widget state for preview
                        setState(() {
                          _selectedMask = mask;
                        });
                        // Update modal state for selection border
                        setModalState(() {});
                      },
                      child: Container(
                        width: 100,
                        margin: const EdgeInsets.only(right: 8),
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: isSelected ? Colors.blue : Colors.grey,
                            width: isSelected ? 3 : 1,
                          ),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: mask == null
                              ? const Center(
                                  child: Text(
                                    'None',
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                )
                              : Image.asset(mask, fit: BoxFit.contain),
                        ),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _playbackTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Video Segmentation')),
      body: _isProcessing
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(_statusMessage ?? 'Processing...'),
                ],
              ),
            )
          : _statusMessage != null && _framePaths.isEmpty
          ? Center(child: Text(_statusMessage!))
          : Stack(
              fit: StackFit.expand,
              children: [
                if (_currentFrameImage != null)
                  CustomPaint(painter: _ImagePainter(_currentFrameImage!)),
                // Overlay
                SegmentationOverlay(
                  detections: _currentDetections,
                  maskThreshold: 0.5,
                  flipHorizontal: false,
                  flipVertical: true,
                  backgroundAsset: _selectedBackground,
                ),
                // Controls
                Positioned(
                  bottom: 32,
                  left: 0,
                  right: 0,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Background selector button
                      IconButton.filled(
                        onPressed: _showBackgroundSelector,
                        icon: const Icon(Icons.image),
                        tooltip: 'Select Background',
                      ),
                      const SizedBox(width: 16),
                      // Play/Pause button
                      IconButton.filled(
                        onPressed: () {
                          if (_playbackTimer?.isActive ?? false) {
                            _playbackTimer?.cancel();
                          } else {
                            _startPlayback();
                          }
                          setState(() {});
                        },
                        icon: Icon(
                          (_playbackTimer?.isActive ?? false)
                              ? Icons.pause
                              : Icons.play_arrow,
                        ),
                        tooltip: 'Play/Pause',
                      ),
                      const SizedBox(width: 16),
                      // Mask selector button
                      IconButton.filled(
                        onPressed: _showMaskSelector,
                        icon: const Icon(Icons.face),
                        tooltip: 'Select Mask',
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }
}

class _ImagePainter extends CustomPainter {
  final ui.Image image;

  _ImagePainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    paintImage(
      canvas: canvas,
      rect: Offset.zero & size,
      image: image,
      fit: BoxFit.contain,
    );
  }

  @override
  bool shouldRepaint(covariant _ImagePainter oldDelegate) {
    return oldDelegate.image != image;
  }
}
