import 'dart:async';
import 'dart:collection';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

class FrameProcessingService {
  static final FrameProcessingService _instance = FrameProcessingService._internal();
  factory FrameProcessingService() => _instance;
  FrameProcessingService._internal();

  // Isolates for heavy computations
  Isolate? _segmentationIsolate;
  Isolate? _poseIsolate;
  
  // Frame queues
  final Queue<FrameData> _segmentationQueue = Queue<FrameData>();
  final Queue<FrameData> _poseQueue = Queue<FrameData>();
  final Queue<ProcessedFrame> _renderQueue = Queue<ProcessedFrame>();
  
  // Stream controllers for async communication
  final _segmentationController = StreamController<SegmentationResult>.broadcast();
  final _poseController = StreamController<PoseResult>.broadcast();
  final _renderController = StreamController<ProcessedFrame>.broadcast();
  
  // Performance monitoring
  final _fpsCounter = FpsCounter();
  Timer? _performanceTimer;
  
  // Configuration
  static const int maxQueueSize = 3;
  static const int targetFps = 30;
  static const Duration frameInterval = Duration(milliseconds: 1000 ~/ targetFps);

  Stream<SegmentationResult> get segmentationStream => _segmentationController.stream;
  Stream<PoseResult> get poseStream => _poseController.stream;
  Stream<ProcessedFrame> get renderStream => _renderController.stream;
  FpsCounter get fpsCounter => _fpsCounter;

  Future<void> initialize() async {
    await _initializeIsolates();
    _startPerformanceMonitoring();
  }

  Future<void> _initializeIsolates() async {
    // Create isolates for parallel processing
    final receivePort = ReceivePort();
    
    _segmentationIsolate = await Isolate.spawn(
      _segmentationWorker,
      receivePort.sendPort,
    );
    
    _poseIsolate = await Isolate.spawn(
      _poseWorker,
      receivePort.sendPort,
    );
    
    receivePort.listen((message) {
      _handleWorkerMessage(message);
    });
  }

  void processFrame(FrameData frameData) {
    _fpsCounter.recordFrame();
    
    // Add to processing queues if not full
    if (_segmentationQueue.length < maxQueueSize) {
      _segmentationQueue.add(frameData);
    }
    
    if (_poseQueue.length < maxQueueSize) {
      _poseQueue.add(frameData.copy());
    }
    
    // Trigger processing
    _unawaited(_processSegmentationQueue());
    _unawaited(_processPoseQueue());
  }

  Future<void> _processSegmentationQueue() async {
    if (_segmentationQueue.isEmpty) return;
    
    final frameData = _segmentationQueue.removeFirst();
    
    try {
      final result = await _computeSegmentation(frameData);
      _segmentationController.add(result);
    } catch (e) {
      debugPrint('Segmentation error: $e');
    }
  }

  Future<void> _processPoseQueue() async {
    if (_poseQueue.isEmpty) return;
    
    final frameData = _poseQueue.removeFirst();
    
    try {
      final result = await _computePose(frameData);
      _poseController.add(result);
    } catch (e) {
      debugPrint('Pose detection error: $e');
    }
  }

  Future<SegmentationResult> _computeSegmentation(FrameData frameData) async {
    // Run segmentation in compute isolate
    return await compute(_runSegmentationIsolate, frameData);
  }

  Future<PoseResult> _computePose(FrameData frameData) async {
    // Run pose detection in compute isolate
    return await compute(_runPoseIsolate, frameData);
  }

  void _handleWorkerMessage(dynamic message) {
    // Handle messages from worker isolates
    if (message is Map<String, dynamic>) {
      switch (message['type']) {
        case 'segmentation_result':
          _segmentationController.add(SegmentationResult.fromMap(message));
          break;
        case 'pose_result':
          _poseController.add(PoseResult.fromMap(message));
          break;
      }
    }
  }

  void _startPerformanceMonitoring() {
    _performanceTimer = Timer.periodic(Duration(seconds: 1), (_) {
      final fps = _fpsCounter.getAverageFps();
      debugPrint('Current FPS: ${fps.toStringAsFixed(1)}');
      debugPrint('Queue sizes - Segmentation: ${_segmentationQueue.length}, Pose: ${_poseQueue.length}, Render: ${_renderQueue.length}');
    });
  }

  void dispose() {
    _segmentationIsolate?.kill(priority: Isolate.immediate);
    _poseIsolate?.kill(priority: Isolate.immediate);
    _performanceTimer?.cancel();
    _segmentationController.close();
    _poseController.close();
    _renderController.close();
  }
}

// Data classes
class FrameData {
  final Uint8List bytes;
  final int index;
  final DateTime timestamp;
  final ui.Size? imageSize;
  
  FrameData({
    required this.bytes,
    required this.index,
    required this.timestamp,
    this.imageSize,
  });
  
  FrameData copy() => FrameData(
    bytes: bytes,
    index: index,
    timestamp: timestamp,
    imageSize: imageSize,
  );
}

class ProcessedFrame {
  final int frameIndex;
  final List<YOLOResult> segmentationDetections;
  final List<YOLOResult> poseDetections;
  final DateTime timestamp;
  
  ProcessedFrame({
    required this.frameIndex,
    required this.segmentationDetections,
    required this.poseDetections,
    required this.timestamp,
  });
}

class SegmentationResult {
  final List<YOLOResult> detections;
  final int frameIndex;
  final DateTime timestamp;
  
  SegmentationResult({
    required this.detections,
    required this.frameIndex,
    required this.timestamp,
  });
  
  factory SegmentationResult.fromMap(Map<String, dynamic> map) {
    return SegmentationResult(
      detections: (map['detections'] as List?)
          ?.map((d) => YOLOResult.fromMap(Map<String, dynamic>.from(d)))
          .toList() ?? [],
      frameIndex: map['frameIndex'] ?? 0,
      timestamp: DateTime.fromMillisecondsSinceEpoch(map['timestamp'] ?? 0),
    );
  }
}

class PoseResult {
  final List<YOLOResult> detections;
  final int frameIndex;
  final DateTime timestamp;
  
  PoseResult({
    required this.detections,
    required this.frameIndex,
    required this.timestamp,
  });
  
  factory PoseResult.fromMap(Map<String, dynamic> map) {
    return PoseResult(
      detections: (map['detections'] as List?)
          ?.map((d) => YOLOResult.fromMap(Map<String, dynamic>.from(d)))
          .toList() ?? [],
      frameIndex: map['frameIndex'] ?? 0,
      timestamp: DateTime.fromMillisecondsSinceEpoch(map['timestamp'] ?? 0),
    );
  }
}

class FpsCounter {
  final List<DateTime> _frameTimes = [];
  static const int maxSamples = 60;
  
  void recordFrame() {
    _frameTimes.add(DateTime.now());
    if (_frameTimes.length > maxSamples) {
      _frameTimes.removeAt(0);
    }
  }
  
  double getAverageFps() {
    if (_frameTimes.length < 2) return 0.0;
    
    final duration = _frameTimes.last.difference(_frameTimes.first).inMilliseconds;
    return duration > 0 ? (_frameTimes.length - 1) * 1000.0 / duration : 0.0;
  }
  
  void reset() {
    _frameTimes.clear();
  }
}

// Isolate entry points
void _segmentationWorker(SendPort sendPort) {
  // Worker isolate for segmentation
}

void _poseWorker(SendPort sendPort) {
  // Worker isolate for pose detection
}

// Compute functions for main thread processing
SegmentationResult _runSegmentationIsolate(FrameData frameData) {
  // Implementation for segmentation computation
  return SegmentationResult(
    detections: [],
    frameIndex: frameData.index,
    timestamp: frameData.timestamp,
  );
}

PoseResult _runPoseIsolate(FrameData frameData) {
  // Implementation for pose computation
  return PoseResult(
    detections: [],
    frameIndex: frameData.index,
    timestamp: frameData.timestamp,
  );
}

void _unawaited(Future<void> future) {
  // Ignore unawaited futures
}