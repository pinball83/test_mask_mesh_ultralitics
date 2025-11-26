import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:ffmpeg_kit_min_gpl/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_min_gpl/return_code.dart';
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import '../services/frame_processing_service.dart';

class OptimizedVideoSegmentationController extends ChangeNotifier {
  final FrameProcessingService _processingService = FrameProcessingService();

  // Video playback state
  bool _isPlaying = false;
  int _currentFrameIndex = 0;
  Timer? _playbackTimer;
  VideoPlayerController? _videoController;
  bool _videoInitialized = false;

  // Frame data
  List<Uint8List> _frameBytes = [];
  List<String> _framePaths = [];
  Duration _frameDuration = Duration(
    milliseconds: 33,
  ); // ~30 FPS (adjusted on init)

  // Processing results with buffering
  final Map<int, ProcessedFrame> _processedFrames = {};
  final Map<int, SegmentationResult> _segmentationResults = {};
  final Map<int, PoseResult> _poseResults = {};

  // Current display state
  List<YOLOResult> _currentDetections = [];
  List<YOLOResult> _currentPoseDetections = [];

  // Stream subscriptions
  StreamSubscription<SegmentationResult>? _segmentationSubscription;
  StreamSubscription<PoseResult>? _poseSubscription;

  // Performance optimization
  static const int maxBufferedFrames = 6;
  static const int processingAheadFrames = 1;

  // Getters
  bool get isPlaying => _isPlaying;
  int get currentFrameIndex => _currentFrameIndex;
  List<YOLOResult> get currentDetections => _currentDetections;
  List<YOLOResult> get currentPoseDetections => _currentPoseDetections;
  Duration get frameDuration => _frameDuration;
  FpsCounter get fpsCounter => _processingService.fpsCounter;
  VideoPlayerController? get videoController => _videoController;
  bool get isVideoInitialized => _videoInitialized;

  Future<void> initialize(List<String> framePaths) async {
    _framePaths = framePaths;
    await _loadFrames();
    await _processingService.initialize();
    _setupStreamListeners();
  }

  Future<List<String>> _extractFramesIfNeeded({
    required File videoFile,
    required int targetFps,
  }) async {
    final tempDir = await getTemporaryDirectory();
    final framesDir = Directory('${tempDir.path}/frames');
    if (!await framesDir.exists()) {
      await framesDir.create();
    }

    final existingFiles = await framesDir.list().toList();
    final hasFrames = existingFiles.whereType<File>().any(
      (f) => f.path.endsWith('.jpg'),
    );

    if (!hasFrames) {
      // Extract at target fps; scale down for performance (Android smaller)
      final scaledWidth = Platform.isIOS ? 480 : 256;
      final command =
          '-i ${videoFile.path} -vf "fps=$targetFps,scale=$scaledWidth:-1" ${framesDir.path}/frame_%04d.jpg';
      final session = await FFmpegKit.execute(command);
      final returnCode = await session.getReturnCode();
      if (!ReturnCode.isSuccess(returnCode)) {
        throw Exception('Failed to extract frames');
      }
    }

    final files = await framesDir.list().toList();
    final paths =
        files
            .whereType<File>()
            .map((f) => f.path)
            .where((p) => p.endsWith('.jpg'))
            .toList()
          ..sort();
    return paths;
  }

  Future<void> initializeFromFile(
    File videoFile, {
    required int targetFps,
  }) async {
    // 0) Validate the video file exists
    if (!await videoFile.exists()) {
      throw Exception('Video file not found: ${videoFile.path}');
    }

    // 1) Extract frames and initialize processing
    final paths = await _extractFramesIfNeeded(
      videoFile: videoFile,
      targetFps: targetFps,
    );
    await initialize(paths);

    // 2) Initialize video player
    _videoController = VideoPlayerController.file(videoFile);
    await _videoController!.initialize();
    await _videoController!.setLooping(true);
    // Align frame duration to requested fps
    _frameDuration = Duration(milliseconds: (1000 / targetFps).round());
    _videoInitialized = true;
    notifyListeners();
  }

  Future<void> _loadFrames() async {
    _frameBytes.clear();
    for (final path in _framePaths) {
      final bytes = await _loadFrameBytes(path);
      _frameBytes.add(bytes);
    }
  }

  Future<Uint8List> _loadFrameBytes(String path) async {
    // Load frame bytes from file
    final file = File(path);
    return await file.readAsBytes();
  }

  void _setupStreamListeners() {
    _segmentationSubscription = _processingService.segmentationStream.listen((
      result,
    ) {
      _segmentationResults[result.frameIndex] = result;
      if (result.frameIndex == _currentFrameIndex) {
        final newDetections = result.detections;
        if (!listEquals(newDetections, _currentDetections)) {
          _currentDetections = newDetections;
          notifyListeners();
        }
      }
      _tryCombineResults(result.frameIndex);
    });

    _poseSubscription = _processingService.poseStream.listen((result) {
      _poseResults[result.frameIndex] = result;
      if (result.frameIndex == _currentFrameIndex) {
        final newPose = result.detections;
        if (!listEquals(newPose, _currentPoseDetections)) {
          _currentPoseDetections = newPose;
          notifyListeners();
        }
      }
      _tryCombineResults(result.frameIndex);
    });
  }

  void _tryCombineResults(int frameIndex) {
    final segResult = _segmentationResults[frameIndex];
    final poseResult = _poseResults[frameIndex];

    if (segResult != null && poseResult != null) {
      final processedFrame = ProcessedFrame(
        frameIndex: frameIndex,
        segmentationDetections: segResult.detections,
        poseDetections: poseResult.detections,
        timestamp: DateTime.now(),
      );

      _processedFrames[frameIndex] = processedFrame;

      // Update current display if this is the current frame
      if (frameIndex == _currentFrameIndex) {
        _updateDisplayFrame(processedFrame);
      }

      // Clean old results to prevent memory leaks
      _cleanOldFrames(frameIndex);
    }
  }

  void _updateDisplayFrame(ProcessedFrame frame) {
    _currentDetections = frame.segmentationDetections;
    _currentPoseDetections = frame.poseDetections;
    notifyListeners();
  }

  void _cleanOldFrames(int currentFrameIndex) {
    final cutoffFrame = currentFrameIndex - maxBufferedFrames;

    _segmentationResults.removeWhere((key, _) => key < cutoffFrame);
    _poseResults.removeWhere((key, _) => key < cutoffFrame);
    _processedFrames.removeWhere((key, _) => key < cutoffFrame);
  }

  void startPlayback() {
    if (_isPlaying) return;

    _isPlaying = true;
    _playbackTimer = Timer.periodic(_frameDuration, _onPlaybackTick);
    unawaited(_videoController?.play());

    // Pre-process a few frames ahead
    _schedulePreprocessing();
    // Process current frame immediately for fast first paint
    _processCurrentFrame();
  }

  void _onPlaybackTick(Timer timer) {
    if (!_isPlaying || _frameBytes.isEmpty) return;

    // Update current frame
    _currentFrameIndex = (_currentFrameIndex + 1) % _framePaths.length;

    // Check if we have processed results for this frame
    final processedFrame = _processedFrames[_currentFrameIndex];
    if (processedFrame != null) {
      _updateDisplayFrame(processedFrame);
    } else {
      // Use last known results if current frame not processed yet
      // This prevents flickering
    }

    // Schedule processing for upcoming frames
    _schedulePreprocessing();
  }

  void _schedulePreprocessing() {
    for (int i = 0; i < processingAheadFrames; i++) {
      final futureFrameIndex =
          (_currentFrameIndex + i + 1) % _framePaths.length;

      if (!_processedFrames.containsKey(futureFrameIndex) &&
          futureFrameIndex < _frameBytes.length) {
        final frameData = FrameData(
          bytes: _frameBytes[futureFrameIndex],
          index: futureFrameIndex,
          timestamp: DateTime.now(),
        );

        _processingService.processFrame(frameData);
      }
    }
  }

  void _processCurrentFrame() {
    if (_frameBytes.isEmpty || _currentFrameIndex >= _frameBytes.length) return;
    final frameData = FrameData(
      bytes: _frameBytes[_currentFrameIndex],
      index: _currentFrameIndex,
      timestamp: DateTime.now(),
    );
    _processingService.processFrame(frameData);
  }

  void pausePlayback() {
    _isPlaying = false;
    _playbackTimer?.cancel();
    _playbackTimer = null;
    unawaited(_videoController?.pause());
  }

  void seekToFrame(int frameIndex) {
    if (frameIndex < 0 || frameIndex >= _framePaths.length) return;

    _currentFrameIndex = frameIndex;

    // Try to display processed frame if available
    final processedFrame = _processedFrames[frameIndex];
    if (processedFrame != null) {
      _updateDisplayFrame(processedFrame);
    }

    // Pre-process around the new position
    _schedulePreprocessing();
  }

  void setFrameDuration(Duration duration) {
    _frameDuration = duration;

    // Restart playback timer with new duration if playing
    if (_isPlaying) {
      _playbackTimer?.cancel();
      _playbackTimer = Timer.periodic(_frameDuration, _onPlaybackTick);
    }
  }

  @override
  void dispose() {
    _playbackTimer?.cancel();
    _segmentationSubscription?.cancel();
    _poseSubscription?.cancel();
    _processingService.dispose();
    unawaited(_videoController?.dispose());
    super.dispose();
  }
}
