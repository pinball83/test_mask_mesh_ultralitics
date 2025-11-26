import 'dart:async';
import 'dart:collection';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import 'model_loader.dart';

class FrameProcessingService {
  static final FrameProcessingService _instance = FrameProcessingService._internal();
  factory FrameProcessingService() => _instance;
  FrameProcessingService._internal();

  // Isolates for heavy computations
  Isolate? _segmentationIsolate;
  Isolate? _poseIsolate;
  SendPort? _segmentationSendPort;
  SendPort? _poseSendPort;
  bool _segmentationBusy = false;
  bool _poseBusy = false;
  bool _useWorkerIsolates = false; // Disabled due to platform channel limits

  // Main-isolate YOLO runtimes
  YOLO? _segYolo;
  YOLO? _poseYolo;
  
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
  static const int maxQueueSize = 1;
  static const int targetFps = 30;
  static const Duration frameInterval = Duration(milliseconds: 1000 ~/ targetFps);
  // Throttling/strides
  int segmentationStride = 2; // run segmentation every Nth frame
  int poseStride = 6; // run pose every Nth frame

  Stream<SegmentationResult> get segmentationStream => _segmentationController.stream;
  Stream<PoseResult> get poseStream => _poseController.stream;
  Stream<ProcessedFrame> get renderStream => _renderController.stream;
  FpsCounter get fpsCounter => _fpsCounter;

  Future<void> initialize() async {
    await _initializeIsolates();
    _startPerformanceMonitoring();
  }

  Future<void> _initializeIsolates() async {
    // Prepare models first so workers can load them immediately
    final loader = ModelLoader();
    final segPath = await loader.ensureSegmentationModel(
      modelName: ModelLoader.modelNameSegmentation,
      onStatus: (_) {},
    );
    final posePath = await loader.ensureSegmentationModel(
      modelName: ModelLoader.modelNamePose,
      onStatus: (_) {},
    );

    // Create isolates for parallel processing and wire communication
    final receivePort = ReceivePort();

    // Initialize main-isolate YOLO instances
    if (segPath != null) {
      try {
        final yolo = YOLO(
          modelPath: segPath,
          task: YOLOTask.segment,
          useGpu: true,
          useMultiInstance: true,
        );
        final loaded = await yolo.loadModel();
        if (loaded) _segYolo = yolo;
      } catch (_) {}
    }
    if (posePath != null) {
      try {
        final yolo = YOLO(
          modelPath: posePath,
          task: YOLOTask.pose,
          useGpu: true,
          useMultiInstance: true,
        );
        final loaded = await yolo.loadModel();
        if (loaded) _poseYolo = yolo;
      } catch (_) {}
    }

    // Optionally enable worker isolates in future if plugin supports it
    if (_useWorkerIsolates && segPath != null && posePath != null) {
      _segmentationIsolate = await Isolate.spawn(
        _segmentationWorker,
        {
          'mainSendPort': receivePort.sendPort,
          'modelPath': segPath,
          'task': 'segment',
        },
      );
      _poseIsolate = await Isolate.spawn(
        _poseWorker,
        {
          'mainSendPort': receivePort.sendPort,
          'modelPath': posePath,
          'task': 'pose',
        },
      );
      receivePort.listen((message) {
        _handleWorkerMessage(message);
      });
    }
  }

  void processFrame(FrameData frameData) {
    _fpsCounter.recordFrame();
    
    // Apply per-task stride to reduce load
    final idx = frameData.index;
    final allowSeg = segmentationStride <= 1 || (idx % segmentationStride == 0);
    final allowPose = poseStride <= 1 || (idx % poseStride == 0);

    // Keep only the freshest frame in each queue when under pressure
    if (allowSeg) {
      if (_segmentationQueue.length >= maxQueueSize) {
        // Drop stale frames, keep only newest
        _segmentationQueue.clear();
      }
      _segmentationQueue.add(frameData);
    }

    if (allowPose) {
      if (_poseQueue.length >= maxQueueSize) {
        _poseQueue.clear();
      }
      _poseQueue.add(frameData.copy());
    }
    
    // Trigger processing
    _unawaited(_processSegmentationQueue());
    _unawaited(_processPoseQueue());
  }

  Future<void> _processSegmentationQueue() async {
    if (_segmentationBusy || _segmentationQueue.isEmpty) return;

    // Prefer main-isolate YOLO when available
    if (_segYolo != null && !_useWorkerIsolates) {
      final frameData = _segmentationQueue.removeFirst();
      _segmentationBusy = true;
      try {
        final raw = await _segYolo!.predict(frameData.bytes, confidenceThreshold: 0.6);
        List<YOLOResult> detections = [];
        if (raw is Map && raw['detections'] is List) {
          detections = (raw['detections'] as List)
              .whereType<Map>()
              .map((d) => YOLOResult.fromMap(Map<String, dynamic>.from(d.cast<String, dynamic>())))
              .toList();
        }
        _segmentationController.add(SegmentationResult(
          detections: detections,
          frameIndex: frameData.index,
          timestamp: frameData.timestamp,
        ));
      } catch (e) {
        debugPrint('Segmentation error: $e');
      } finally {
        _segmentationBusy = false;
        _unawaited(_processSegmentationQueue());
      }
    } else if (_segmentationSendPort != null) {
      final frameData = _segmentationQueue.removeFirst();
      _segmentationBusy = true;
      _segmentationSendPort!.send({
        'type': 'process',
        'frame': _encodeFrame(frameData),
      });
    } else {
      // Fallback to compute isolate if worker not ready
      final frameData = _segmentationQueue.removeFirst();
      try {
        final result = await _computeSegmentation(frameData);
        _segmentationController.add(result);
      } catch (e) {
        debugPrint('Segmentation error: $e');
      }
    }
  }

  Future<void> _processPoseQueue() async {
    if (_poseBusy || _poseQueue.isEmpty) return;

    if (_poseYolo != null && !_useWorkerIsolates) {
      final frameData = _poseQueue.removeFirst();
      _poseBusy = true;
      try {
        final raw = await _poseYolo!.predict(frameData.bytes, confidenceThreshold: 0.5);
        List<YOLOResult> detections = [];
        if (raw is Map) {
          final list = raw['detections'];
          if (list is List) {
            detections = list
                .whereType<Map>()
                .map((d) => YOLOResult.fromMap(Map<String, dynamic>.from(d.cast<String, dynamic>())))
                .toList();
          } else {
            // Normalize boxes/keypoints format
            detections = _normalizePose(raw);
          }
        }
        _poseController.add(PoseResult(
          detections: detections,
          frameIndex: frameData.index,
          timestamp: frameData.timestamp,
        ));
      } catch (e) {
        debugPrint('Pose detection error: $e');
      } finally {
        _poseBusy = false;
        _unawaited(_processPoseQueue());
      }
    } else if (_poseSendPort != null) {
      final frameData = _poseQueue.removeFirst();
      _poseBusy = true;
      _poseSendPort!.send({
        'type': 'process',
        'frame': _encodeFrame(frameData),
      });
    } else {
      // Fallback to compute isolate if worker not ready
      final frameData = _poseQueue.removeFirst();
      try {
        final result = await _computePose(frameData);
        _poseController.add(result);
      } catch (e) {
        debugPrint('Pose detection error: $e');
      }
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
        case 'worker_ready':
          final worker = message['worker'] as String?;
          final sendPort = message['sendPort'];
          if (sendPort is SendPort && worker != null) {
            if (worker == 'segmentation') {
              _segmentationSendPort = sendPort;
              _unawaited(_processSegmentationQueue());
            } else if (worker == 'pose') {
              _poseSendPort = sendPort;
              _unawaited(_processPoseQueue());
            }
          }
          break;
        case 'segmentation_result':
          _segmentationBusy = false;
          _segmentationController.add(SegmentationResult.fromMap(message));
          // Kick next item in queue, if any
          _unawaited(_processSegmentationQueue());
          break;
        case 'pose_result':
          _poseBusy = false;
          _poseController.add(PoseResult.fromMap(message));
          // Kick next item in queue, if any
          _unawaited(_processPoseQueue());
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
    unawaited(_segYolo?.dispose());
    unawaited(_poseYolo?.dispose());
  }

  Map<String, dynamic> _encodeFrame(FrameData frame) => {
        'bytes': frame.bytes,
        'index': frame.index,
        'timestamp': frame.timestamp.millisecondsSinceEpoch,
        if (frame.imageSize != null)
          'imageSize': {
            'width': frame.imageSize!.width,
            'height': frame.imageSize!.height,
          },
      };
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

// Isolate entry points are implemented at the end of this file

// Compute functions for main thread processing
SegmentationResult _runSegmentationIsolate(FrameData frameData) {
  // Placeholder computation when workers are not ready
  return SegmentationResult(
    detections: const [],
    frameIndex: frameData.index,
    timestamp: frameData.timestamp,
  );
}

PoseResult _runPoseIsolate(FrameData frameData) {
  // Placeholder computation when workers are not ready
  return PoseResult(
    detections: const [],
    frameIndex: frameData.index,
    timestamp: frameData.timestamp,
  );
}

void _unawaited(Future<void> future) {
  // Ignore unawaited futures
}

List<YOLOResult> _normalizePose(Map<String, dynamic> result) {
  final boxes = result['boxes'];
  final keypoints = result['keypoints'];
  if (boxes is! List || keypoints is! List) return const [];
  final count = boxes.length < keypoints.length ? boxes.length : keypoints.length;
  final out = <YOLOResult>[];
  for (var i = 0; i < count; i++) {
    final box = boxes[i];
    final kp = keypoints[i];
    if (box is! Map) continue;
    final b = Map<String, dynamic>.from(box.cast<String, dynamic>());
    final det = <String, dynamic>{
      'classIndex': b['classIndex'] ?? 0,
      'className': b['className'] ?? b['class'] ?? '',
      'confidence': (b['confidence'] as num?)?.toDouble() ?? 0.0,
      'boundingBox': {
        'left': (b['x1'] as num?)?.toDouble() ?? 0.0,
        'top': (b['y1'] as num?)?.toDouble() ?? 0.0,
        'right': (b['x2'] as num?)?.toDouble() ?? 0.0,
        'bottom': (b['y2'] as num?)?.toDouble() ?? 0.0,
      },
      'normalizedBox': {
        'left': (b['x1_norm'] as num?)?.toDouble() ?? 0.0,
        'top': (b['y1_norm'] as num?)?.toDouble() ?? 0.0,
        'right': (b['x2_norm'] as num?)?.toDouble() ?? 0.0,
        'bottom': (b['y2_norm'] as num?)?.toDouble() ?? 0.0,
      },
    };
    if (kp is Map && kp['coordinates'] is List) {
      final coords = (kp['coordinates'] as List)
          .whereType<Map>()
          .expand((coord) {
        final c = Map<String, dynamic>.from(coord.cast<String, dynamic>());
        final x = (c['x'] as num?)?.toDouble() ?? 0.0;
        final y = (c['y'] as num?)?.toDouble() ?? 0.0;
        final conf = (c['confidence'] as num?)?.toDouble() ?? 0.0;
        return [x, y, conf];
      }).toList();
      if (coords.isNotEmpty) det['keypoints'] = coords;
    }
    out.add(YOLOResult.fromMap(det));
  }
  return out;
}

// === Worker isolate implementations ===

void _segmentationWorker(dynamic initMessage) async {
  if (initMessage is! Map) return;
  final map = Map<Object?, Object?>.from(initMessage as Map);
  final mainSendPort = map['mainSendPort'] as SendPort?;
  final modelPath = map['modelPath'] as String?;

  if (mainSendPort == null || modelPath == null) {
    return;
  }

  // Create a dedicated port to receive frames
  final workerReceivePort = ReceivePort();
  mainSendPort.send({
    'type': 'worker_ready',
    'worker': 'segmentation',
    'sendPort': workerReceivePort.sendPort,
  });

  // Initialize YOLO model inside this isolate
  final yolo = YOLO(
    modelPath: modelPath,
    task: YOLOTask.segment,
    useGpu: true,
    useMultiInstance: true,
  );
  try {
    await yolo.loadModel();
  } catch (_) {
    // If model fails to load, still drain messages to avoid deadlock
  }

  await for (final message in workerReceivePort) {
    if (message is! Map) continue;
    final typed = Map<Object?, Object?>.from(message as Map);
    if (typed['type'] != 'process') continue;
    final frame = typed['frame'];
    if (frame is! Map) continue;
    final f = Map<Object?, Object?>.from(frame);
    final bytes = f['bytes'];
    final index = f['index'];
    final ts = f['timestamp'];
    if (bytes is! Uint8List || index is! int) continue;

    Map<String, dynamic>? result;
    try {
      final raw = await yolo.predict(bytes, confidenceThreshold: 0.6);
      if (raw is Map) {
        result = Map<String, dynamic>.from(raw.cast<String, dynamic>());
      }
    } catch (_) {
      // On failure, emit empty detections to keep pipeline moving
      result = {'detections': <dynamic>[]};
    }

    mainSendPort.send({
      'type': 'segmentation_result',
      'detections': (result?['detections'] as List?) ?? const <dynamic>[],
      'frameIndex': index,
      'timestamp': ts is int
          ? ts
          : DateTime.now().millisecondsSinceEpoch,
    });
  }
}

void _poseWorker(dynamic initMessage) async {
  if (initMessage is! Map) return;
  final map = Map<Object?, Object?>.from(initMessage as Map);
  final mainSendPort = map['mainSendPort'] as SendPort?;
  final modelPath = map['modelPath'] as String?;

  if (mainSendPort == null || modelPath == null) {
    return;
  }

  final workerReceivePort = ReceivePort();
  mainSendPort.send({
    'type': 'worker_ready',
    'worker': 'pose',
    'sendPort': workerReceivePort.sendPort,
  });

  final yolo = YOLO(
    modelPath: modelPath,
    task: YOLOTask.pose,
    useGpu: true,
    useMultiInstance: true,
  );
  try {
    await yolo.loadModel();
  } catch (_) {}

  await for (final message in workerReceivePort) {
    if (message is! Map) continue;
    final typed = Map<Object?, Object?>.from(message as Map);
    if (typed['type'] != 'process') continue;
    final frame = typed['frame'];
    if (frame is! Map) continue;
    final f = Map<Object?, Object?>.from(frame);
    final bytes = f['bytes'];
    final index = f['index'];
    final ts = f['timestamp'];
    final imgSize = f['imageSize'];
    if (bytes is! Uint8List || index is! int) continue;

    Map<String, dynamic>? result;
    try {
      final raw = await yolo.predict(bytes, confidenceThreshold: 0.5);
      if (raw is Map) {
        result = Map<String, dynamic>.from(raw.cast<String, dynamic>());
      }
    } catch (_) {
      result = {'detections': <dynamic>[]};
    }

    // Normalize to a 'detections' list if needed
    List<dynamic> detections = (result?['detections'] as List?) ?? const [];
    if (detections.isEmpty && result != null) {
      final boxes = result['boxes'];
      final keypoints = result['keypoints'];
      if (boxes is List && keypoints is List) {
        final count = boxes.length < keypoints.length ? boxes.length : keypoints.length;
        final normalized = <Map<String, dynamic>>[];
        for (var i = 0; i < count; i++) {
          final box = boxes[i];
          final kp = keypoints[i];
          if (box is! Map) continue;
          final b = Map<String, dynamic>.from(box.cast<String, dynamic>());
          final det = <String, dynamic>{
            'classIndex': b['classIndex'] ?? 0,
            'className': b['className'] ?? b['class'] ?? '',
            'confidence': (b['confidence'] as num?)?.toDouble() ?? 0.0,
            'boundingBox': {
              'left': (b['x1'] as num?)?.toDouble() ?? 0.0,
              'top': (b['y1'] as num?)?.toDouble() ?? 0.0,
              'right': (b['x2'] as num?)?.toDouble() ?? 0.0,
              'bottom': (b['y2'] as num?)?.toDouble() ?? 0.0,
            },
            'normalizedBox': {
              'left': (b['x1_norm'] as num?)?.toDouble() ?? 0.0,
              'top': (b['y1_norm'] as num?)?.toDouble() ?? 0.0,
              'right': (b['x2_norm'] as num?)?.toDouble() ?? 0.0,
              'bottom': (b['y2_norm'] as num?)?.toDouble() ?? 0.0,
            },
          };
          if (kp is Map && kp['coordinates'] is List) {
            final coords = (kp['coordinates'] as List)
                .whereType<Map>()
                .expand((coord) {
              final c = Map<String, dynamic>.from(coord.cast<String, dynamic>());
              final x = (c['x'] as num?)?.toDouble() ?? 0.0;
              final y = (c['y'] as num?)?.toDouble() ?? 0.0;
              final conf = (c['confidence'] as num?)?.toDouble() ?? 0.0;
              return [x, y, conf];
            }).toList();
            if (coords.isNotEmpty) det['keypoints'] = coords;
          }
          normalized.add(det);
        }
        detections = normalized;
      }
    }

    final out = <String, dynamic>{
      'type': 'pose_result',
      'detections': detections,
      'frameIndex': index,
      'timestamp': ts is int
          ? ts
          : DateTime.now().millisecondsSinceEpoch,
    };

    if (imgSize is Map) {
      out['imageSize'] = imgSize.cast<String, dynamic>();
    }

    mainSendPort.send(out);
  }
}
