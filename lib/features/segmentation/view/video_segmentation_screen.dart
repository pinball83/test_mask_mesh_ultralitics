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
import '../widgets/face_mask_overlay.dart';
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
  List<Uint8List> _frameBytes = [];
  int _currentFrameIndex = 0;
  bool _isInferring = false;
  bool _isProcessing = false;
  bool _segModelReady = false;
  bool _poseModelReady = false;
  String? _statusMessage;

  List<YOLOResult> _currentDetections = [];
  List<YOLOResult> _currentPoseDetections = [];
  String? _selectedBackground = 'assets/images/bg_image.jpg';
  String? _selectedMask;

  // FPS control
  static const int _targetFps = 24;
  static const int _defaultInferenceStride = 4; // run inference every N frames
  static const Duration _frameDuration = Duration(
    milliseconds: 1000 ~/ _targetFps,
  );

  int get _inferenceStride => Platform.isIOS ? 3 : _defaultInferenceStride;

  bool _isPlaying = false;

  YOLO? _yolo;
  YOLO? _poseYolo;

  @override
  void initState() {
    super.initState();
    _initializeYolo();
    _initializeVideo();
  }

  Future<void> _initializeYolo() async {
    final modelPath = widget.controller.modelPath;
    if (modelPath != null) {
      final seg = YOLO(
        modelPath: modelPath,
        task: YOLOTask.segment,
        useGpu: true,
        useMultiInstance: true,
      );
      try {
        final loaded = await seg.loadModel();
        if (loaded) {
          _yolo = seg;
          _segModelReady = true;
        }
      } catch (e) {
        debugPrint('Error loading Segmentation Model: $e');
      }
    }

    final poseModelPath = widget.controller.poseModelPath;
    debugPrint('Initializing Pose Model: $poseModelPath');
    if (poseModelPath != null) {
      try {
        final pose = YOLO(
          modelPath: poseModelPath,
          task: YOLOTask.pose,
          useGpu: true,
          useMultiInstance: true,
        );
        final loaded = await pose.loadModel();
        if (loaded) {
          _poseYolo = pose;
          _poseModelReady = true;
          debugPrint('Pose Model loaded successfully');
        }
      } catch (e) {
        debugPrint('Error loading Pose Model: $e');
      }
    } else {
      debugPrint('Pose Model path is null');
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

      // 2. Initialize video player for rendering (hardware-decoded texture)
      _videoController = VideoPlayerController.file(videoFile);
      await _videoController!.initialize();
      await _videoController!.setLooping(true);
      await _videoController!.play();

      // 3. Extract frames using FFmpeg
      final framesDir = Directory('${tempDir.path}/frames');
      if (!await framesDir.exists()) {
        await framesDir.create();
      }

      final existingFiles = await framesDir.list().toList();
      final hasFrames = existingFiles
          .where((f) => f.path.endsWith('.jpg'))
          .isNotEmpty;

      if (!hasFrames) {
        setState(() => _statusMessage = 'Extracting frames...');
        // Extract at 30fps, scale to 640 width (maintain aspect ratio) for performance
        final scaledWidth = Platform.isIOS ? 480 : 360;
        final command =
            '-i ${videoFile.path} -vf "fps=$_targetFps,scale=$scaledWidth:-1" ${framesDir.path}/frame_%04d.jpg';
        final session = await FFmpegKit.execute(command);
        final returnCode = await session.getReturnCode();

        if (!ReturnCode.isSuccess(returnCode)) {
          setState(() {
            _isProcessing = false;
            _statusMessage = 'Failed to extract frames';
          });
          return;
        }
      }

      final files = await framesDir.list().toList();
      _framePaths =
          files
              .whereType<File>()
              .map((f) => f.path)
              .where((p) => p.endsWith('.jpg'))
              .toList()
            ..sort();

      // 4. Load frames for inference (bytes only to save memory)
      setState(() => _statusMessage = 'Loading frames...');
      _frameBytes = [];
      for (final path in _framePaths) {
        final file = File(path);
        final bytes = await file.readAsBytes();
        _frameBytes.add(bytes);
      }

      setState(() {
        _isProcessing = false;
        _statusMessage = null;
      });

      _startPlayback();
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _statusMessage = 'Error: $e';
      });
    }
  }

  void _startPlayback() {
    _isPlaying = true;
    _currentFrameIndex = 0;
    _processFrameLoop();
  }

  Future<void> _processFrameLoop() async {
    while (mounted && _isPlaying) {
      final frameStart = DateTime.now();

      final shouldInfer =
          _frameBytes.isNotEmpty &&
          (_segModelReady || _poseModelReady) &&
          (_currentFrameIndex % _inferenceStride == 0) &&
          !_isInferring;
      if (shouldInfer) {
        final frameBytes = _frameBytes[_currentFrameIndex];
        unawaited(_startInference(frameBytes));
      }

      final elapsed = DateTime.now().difference(frameStart);
      final delay = _frameDuration - elapsed;
      if (delay > Duration.zero) {
        await Future.delayed(delay);
      }

      if (!mounted || !_isPlaying) break;
      _currentFrameIndex = (_currentFrameIndex + 1) % _framePaths.length;
    }
  }

  Future<void> _startInference(Uint8List frameBytes) async {
    if (_isInferring) return;
    _isInferring = true;
    try {
      await _runInference(frameBytes);
    } finally {
      _isInferring = false;
    }
  }

  Future<void> _runInference(Uint8List frameBytes) async {
    try {
      final futures = <Future<dynamic>>[];

      // 1. Segmentation
      if (_segModelReady && _yolo != null) {
        futures.add(_yolo!.predict(frameBytes, confidenceThreshold: 0.75));
      } else {
        futures.add(Future.value(null));
      }

      // 2. Pose Detection
      if (_poseModelReady && _poseYolo != null) {
        futures.add(_poseYolo!.predict(frameBytes, confidenceThreshold: 0.5));
      } else {
        futures.add(Future.value(null));
      }

      final results = await Future.wait(futures);
      final segResult = results[0];
      final poseResult = results[1];

      List<YOLOResult> detections = [];
      if (segResult is Map && segResult['detections'] is List) {
        detections = (segResult['detections'] as List)
            .whereType<Map<dynamic, dynamic>>()
            .map((d) => YOLOResult.fromMap(Map<String, dynamic>.from(d)))
            .toList();
      }

      List<YOLOResult> poseDetections = [];
      if (poseResult is Map) {
        poseDetections = _parsePoseDetections(
          Map<String, dynamic>.from(poseResult),
        );
      }

      if (mounted) {
        setState(() {
          _currentDetections = detections;
          _currentPoseDetections = poseDetections;
        });
      }
    } catch (e) {
      debugPrint('Inference error: $e');
    } finally {
      // _isInferring = false; // No longer needed
    }
  }

  List<YOLOResult> _parsePoseDetections(Map<String, dynamic> poseResult) {
    ui.Size? poseImageSize;
    final poseSize = poseResult['imageSize'];
    if (poseSize is Map) {
      final w = poseSize['width'];
      final h = poseSize['height'];
      if (w is num && h is num) {
        poseImageSize = ui.Size(w.toDouble(), h.toDouble());
      }
    }

    final parsed = <YOLOResult>[];
    final rawDetections = poseResult['detections'];
    if (rawDetections is List) {
      for (final detection in rawDetections.whereType<Map>()) {
        final map = Map<String, dynamic>.from(detection);
        if (poseImageSize != null) {
          map['imageSize'] = {
            'width': poseImageSize.width,
            'height': poseImageSize.height,
          };
        }
        parsed.add(YOLOResult.fromMap(map));
      }
    }
    if (parsed.isNotEmpty) return parsed;

    final boxes = poseResult['boxes'];
    final keypoints = poseResult['keypoints'];
    if (boxes is! List || keypoints is! List) return parsed;

    final itemCount = boxes.length < keypoints.length
        ? boxes.length
        : keypoints.length;

    for (var i = 0; i < itemCount; i++) {
      final box = boxes[i];
      final kp = keypoints[i];
      if (box is! Map) continue;
      final typedBox = Map<String, dynamic>.from(box);

      final detectionMap = <String, dynamic>{
        'classIndex': typedBox['classIndex'] ?? 0,
        'className': typedBox['className'] ?? typedBox['class'] ?? '',
        'confidence': (typedBox['confidence'] as num?)?.toDouble() ?? 0.0,
        'boundingBox': {
          'left': (typedBox['x1'] as num?)?.toDouble() ?? 0.0,
          'top': (typedBox['y1'] as num?)?.toDouble() ?? 0.0,
          'right': (typedBox['x2'] as num?)?.toDouble() ?? 0.0,
          'bottom': (typedBox['y2'] as num?)?.toDouble() ?? 0.0,
        },
        'normalizedBox': {
          'left': (typedBox['x1_norm'] as num?)?.toDouble() ?? 0.0,
          'top': (typedBox['y1_norm'] as num?)?.toDouble() ?? 0.0,
          'right': (typedBox['x2_norm'] as num?)?.toDouble() ?? 0.0,
          'bottom': (typedBox['y2_norm'] as num?)?.toDouble() ?? 0.0,
        },
      };

      if (poseImageSize != null) {
        detectionMap['imageSize'] = {
          'width': poseImageSize.width,
          'height': poseImageSize.height,
        };
      }

      if (kp is Map && kp['coordinates'] is List) {
        final coords = (kp['coordinates'] as List).whereType<Map>().expand((
          coord,
        ) {
          final coordMap = Map<String, dynamic>.from(coord);
          final x = (coordMap['x'] as num?)?.toDouble() ?? 0.0;
          final y = (coordMap['y'] as num?)?.toDouble() ?? 0.0;
          final conf = (coordMap['confidence'] as num?)?.toDouble() ?? 0.0;
          return [x, y, conf];
        }).toList();
        if (coords.isNotEmpty) {
          detectionMap['keypoints'] = coords;
        }
      }

      parsed.add(YOLOResult.fromMap(detectionMap));
    }

    return parsed;
  }

  void _showBackgroundSelector() async {
    final backgrounds = [
      null, // No background option
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
                          child: bg == null
                              ? const Center(
                                  child: Text(
                                    'None',
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                )
                              : Image.asset(bg, fit: BoxFit.cover),
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
                        debugPrint('Selected mask: $mask');
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
    _isPlaying = false;
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
                if (_videoController?.value.isInitialized ?? false)
                  Center(
                    child: AspectRatio(
                      aspectRatio: _videoController!.value.aspectRatio,
                      child: VideoPlayer(_videoController!),
                    ),
                  ),
                // Overlay
                if (_selectedBackground != null)
                  SegmentationOverlay(
                    detections: _currentDetections,
                    maskThreshold: Platform.isIOS ? 0.4 : 0.5,
                    flipHorizontal: false,
                    flipVertical: true,
                    backgroundAsset: _selectedBackground,
                  ),
                // Face Mask Overlay
                FaceMaskOverlay(
                  poseDetections: _currentPoseDetections,
                  maskAsset: _selectedMask,
                  flipHorizontal: false,
                  flipVertical: true,
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
                          if (_isPlaying) {
                            _isPlaying = false;
                          } else {
                            _startPlayback();
                          }
                          setState(() {});
                        },
                        icon: Icon(_isPlaying ? Icons.pause : Icons.play_arrow),
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
