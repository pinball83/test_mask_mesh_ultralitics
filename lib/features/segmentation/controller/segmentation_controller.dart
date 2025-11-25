import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:ultralytics_yolo/models/yolo_model_spec.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

import '../services/model_loader.dart';

enum SegmentationOverlayMode {
  backgroundReplacement,
  maskOnly,
  combined,
  simplePose,
}

class SegmentationController extends ChangeNotifier {
  SegmentationController({ModelLoader? modelLoader})
    : _modelLoader = modelLoader ?? ModelLoader();

  final ModelLoader _modelLoader;
  final YOLOViewController yoloController = YOLOViewController();

  bool _isLoading = true;
  bool _isUnsupportedPlatform = false;
  SegmentationOverlayMode _overlayMode = SegmentationOverlayMode.simplePose;
  double _confidenceThreshold = 0.45;
  double _maskThreshold = 0.4;
  double _currentZoomLevel = 1.0;
  double _fps = 0.0;
  String _statusMessage = 'Preparing segmentation model...';
  double _downloadProgress = 0.0;
  String? _modelPath;
  String? _poseModelPath;
  String? _errorMessage;
  List<YOLOResult> _currentDetections = const [];
  List<YOLOResult> _poseDetections = const [];
  List<YOLOModelSpec> _yoloModels = const [];
  bool _flipMaskHorizontal = true;
  bool _flipMaskVertical = true;
  final bool _preferFrontCamera = true;
  bool _defaultCameraApplied = false;
  Timer? _cameraRetryTimer;

  YOLOStreamingConfig get streamingConfig => YOLOStreamingConfig.custom(
    includeDetections: false,
    includeClassifications: false,
    includeProcessingTimeMs: false,
    includeFps: false,
    includeMasks: true,
    includePoses: true,
    includeOBB: false,
    includeOriginalImage: false,
    // Ensure max FPS by setting these to null
    maxFPS: 30,
    throttleInterval: null,
    inferenceFrequency: null,
    skipFrames: 3,
  );

  bool get isLoading => _isLoading;
  bool get isUnsupportedPlatform => _isUnsupportedPlatform;
  SegmentationOverlayMode get overlayMode => _overlayMode;
  double get confidenceThreshold => _confidenceThreshold;
  double get maskThreshold => _maskThreshold;
  double get currentZoomLevel => _currentZoomLevel;
  double get fps => _fps;
  String get statusMessage => _statusMessage;
  double get downloadProgress => _downloadProgress;
  String? get modelPath => _modelPath;
  String? get poseModelPath => _poseModelPath;
  String? get errorMessage => _errorMessage;
  List<YOLOResult> get detections => _currentDetections;
  List<YOLOResult> get poseDetections => _poseDetections;

  // New getters for analysis
  int get segmentationCount =>
      _currentDetections.where((d) => d.mask != null).length;
  int get poseCount => _poseDetections.length;

  List<YOLOModelSpec> get yoloModels => _yoloModels;
  bool get flipMaskHorizontal => _flipMaskHorizontal;
  bool get flipMaskVertical => _flipMaskVertical;

  Future<void> initialize() async {
    if (!Platform.isAndroid && !Platform.isIOS) {
      _isUnsupportedPlatform = true;
      _statusMessage = 'This demo currently supports Android or iOS devices.';
      _isLoading = false;
      notifyListeners();
      return;
    }

    if (_modelPath != null) return;

    _statusMessage = 'Fetching segmentation model...';
    _isLoading = true;
    _errorMessage = null;
    _downloadProgress = 0;
    notifyListeners();

    try {
      const totalStages = 2;
      var currentStage = 0;

      void setStageProgress(double stageProgress) {
        final normalized = ((currentStage + stageProgress) / totalStages).clamp(
          0.0,
          1.0,
        );
        if ((_downloadProgress - normalized).abs() > 0.001) {
          _downloadProgress = normalized;
          notifyListeners();
        }
      }

      final segmentationPath = await _modelLoader.ensureSegmentationModel(
        modelName: ModelLoader.modelNameSegmentation,
        onProgress: (progress) {
          setStageProgress(progress);
        },
        onStatus: (status) {
          _statusMessage = status;
          notifyListeners();
        },
      );

      setStageProgress(1);
      currentStage = 1;

      if (segmentationPath == null) {
        throw Exception('Unable to locate segmentation model');
      }

      _statusMessage = 'Fetching pose model...';
      notifyListeners();

      final posePath = await _modelLoader.ensureSegmentationModel(
        modelName: ModelLoader.modelNamePose,
        onProgress: (progress) {
          setStageProgress(progress);
        },
        onStatus: (status) {
          _statusMessage = status;
          notifyListeners();
        },
      );

      setStageProgress(1);
      currentStage = 2;

      if (posePath == null) {
        throw Exception('Unable to locate pose model');
      }

      _modelPath = segmentationPath;
      _poseModelPath = posePath;
      _applyOverlayModeModels();

      await yoloController.setThresholds(
        confidenceThreshold: _confidenceThreshold,
        iouThreshold: 0.45,
        numItemsThreshold: 1,
      );

      _statusMessage = 'Model ready. Initializing camera...';
      _isLoading = false;
      notifyListeners();
      ensurePreferredCamera();
    } catch (error, stackTrace) {
      _errorMessage = 'Failed to prepare model: $error';
      _statusMessage = 'Tap to retry';
      _isLoading = false;
      debugPrint('SegmentationController.initialize error: $error');
      debugPrint(stackTrace.toString());
      notifyListeners();
    }
  }

  void onResults(List<YOLOResult> results) {
    ensurePreferredCamera();
    _currentDetections = results;
    if (_overlayMode != SegmentationOverlayMode.backgroundReplacement) {
      _poseDetections = results
          .where((result) => result.keypoints != null)
          .toList(growable: false);
    } else {
      _poseDetections = const [];
    }
    notifyListeners();
  }

  void onPerformance(YOLOPerformanceMetrics metrics) {
    if ((_fps - metrics.fps).abs() > 0.05) {
      _fps = metrics.fps;
      notifyListeners();
    }
    ensurePreferredCamera();
  }

  void onZoomChanged(double zoomLevel) {
    if ((_currentZoomLevel - zoomLevel).abs() > 0.01) {
      _currentZoomLevel = zoomLevel;
      notifyListeners();
    }
  }

  Future<void> refreshModel() async {
    _modelPath = null;
    _poseModelPath = null;
    _currentDetections = const [];
    _poseDetections = const [];
    _yoloModels = const [];
    await initialize();
  }

  void setOverlayMode(SegmentationOverlayMode mode) {
    if (_overlayMode == mode) return;
    _overlayMode = mode;
    if (mode == SegmentationOverlayMode.backgroundReplacement) {
      _poseDetections = const [];
    }
    _applyOverlayModeModels();
    notifyListeners();
  }

  Future<void> flipCamera() async {
    await yoloController.switchCamera();
  }

  Future<void> zoomIn() async {
    await yoloController.zoomIn();
  }

  Future<void> zoomOut() async {
    await yoloController.zoomOut();
  }

  Future<void> setZoomLevel(double level) async {
    await yoloController.setZoomLevel(level);
  }

  void updateConfidence(double value) {
    _confidenceThreshold = value;
    unawaited(yoloController.setConfidenceThreshold(value));
    notifyListeners();
  }

  void updateMaskThreshold(double value) {
    _maskThreshold = value;
    notifyListeners();
  }

  void toggleMaskHorizontalFlip() {
    _flipMaskHorizontal = !_flipMaskHorizontal;
    notifyListeners();
  }

  void toggleMaskVerticalFlip() {
    _flipMaskVertical = !_flipMaskVertical;
    notifyListeners();
  }

  void ensurePreferredCamera() {
    if (!_preferFrontCamera || _defaultCameraApplied) {
      return;
    }

    if (yoloController.isInitialized) {
      _defaultCameraApplied = true;
      unawaited(yoloController.switchCamera());
      return;
    }

    _scheduleCameraRetry();
  }

  void _scheduleCameraRetry() {
    if (_cameraRetryTimer != null) return;
    _cameraRetryTimer = Timer(const Duration(milliseconds: 150), () {
      _cameraRetryTimer = null;
      ensurePreferredCamera();
    });
  }

  void _applyOverlayModeModels() {
    final segPath = _modelPath;
    if (segPath == null) return;

    final posePath = _poseModelPath;

    List<YOLOModelSpec> models;

    switch (_overlayMode) {
      case SegmentationOverlayMode.backgroundReplacement:
        models = [
          YOLOModelSpec(
            modelPath: segPath,
            type: ModelLoader.modelNameSegmentation,
            task: YOLOTask.segment,
          ),
        ];
        break;
      case SegmentationOverlayMode.maskOnly:
        if (posePath != null) {
          models = [
            YOLOModelSpec(
              modelPath: posePath,
              type: ModelLoader.modelNamePose,
              task: YOLOTask.pose,
            ),
          ];
        } else {
          models = [
            YOLOModelSpec(
              modelPath: segPath,
              type: ModelLoader.modelNameSegmentation,
              task: YOLOTask.segment,
            ),
          ];
        }
        break;
      case SegmentationOverlayMode.combined:
        models = [
          YOLOModelSpec(
            modelPath: segPath,
            type: ModelLoader.modelNameSegmentation,
            task: YOLOTask.segment,
          ),
        ];
        if (posePath != null) {
          models = [
            ...models,
            YOLOModelSpec(
              modelPath: posePath,
              type: ModelLoader.modelNamePose,
              task: YOLOTask.pose,
            ),
          ];
        }
        break;
      case SegmentationOverlayMode.simplePose:
        if (posePath != null) {
          models = [
            YOLOModelSpec(
              modelPath: posePath,
              type: ModelLoader.modelNamePose,
              task: YOLOTask.pose,
            ),
          ];
        } else {
          // Fallback if pose model missing (shouldn't happen if init succeeds)
          models = [
            YOLOModelSpec(
              modelPath: segPath,
              type: ModelLoader.modelNameSegmentation,
              task: YOLOTask.segment,
            ),
          ];
        }
        break;
    }

    _yoloModels = models;
  }

  @override
  void dispose() {
    unawaited(yoloController.stop());
    _cameraRetryTimer?.cancel();
    super.dispose();
  }
}
