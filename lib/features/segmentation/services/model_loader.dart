import 'dart:io';

import 'package:archive/archive.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';

typedef ProgressCallback = void Function(double progress);
typedef StatusCallback = void Function(String message);

/// Handles locating, downloading, and caching the segmentation model.
class ModelLoader {
  static const String modelNameSegmentation = 'yolo11n-seg';
  static const String modelNamePose = 'yolo11s-pose';
  static const String modelNameRoot = 'yolo11n';

  static const String _downloadBase =
      'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0';

  static final MethodChannel _channel =
      ChannelConfig.createSingleImageChannel();

  Future<String?> ensureSegmentationModel({
    required String modelName,
    ProgressCallback? onProgress,
    StatusCallback? onStatus,
  }) async {
    if (Platform.isAndroid) {
      return _prepareAndroidModel(
        modelName: modelName,
        onProgress: onProgress,
        onStatus: onStatus,
      );
    }
    if (Platform.isIOS) {
      return _prepareIOSModel(
        modelName: modelName,
        onProgress: onProgress,
        onStatus: onStatus,
      );
    }
    onStatus?.call('Unsupported platform for Ultralytics camera runtime.');
    return null;
  }

  Future<String?> _prepareAndroidModel({
    required String modelName,
    ProgressCallback? onProgress,
    StatusCallback? onStatus,
  }) async {
    final assetName = '$modelName.tflite';
    onStatus?.call('Checking device for $modelName...');

    try {
      final result = await _channel.invokeMethod<Map<dynamic, dynamic>>(
        'checkModelExists',
        {'modelPath': assetName},
      );
      if (result != null && result['exists'] == true) {
        if (result['location'] == 'assets') {
          return assetName;
        }
        final path = result['path'] as String?;
        if (path != null) return path;
      }
    } catch (_) {
      // Fall back to manual download.
    }

    final docsDir = await getApplicationDocumentsDirectory();
    final target = File('${docsDir.path}/$assetName');
    if (await target.exists()) {
      onStatus?.call('Found cached model');
      return target.path;
    }

    onStatus?.call('Downloading segmentation model...');
    final bytes = await _downloadFile(
      '$_downloadBase/$assetName',
      onProgress: onProgress,
    );
    if (bytes == null || bytes.isEmpty) {
      onStatus?.call('Download failed.');
      return null;
    }

    await target.writeAsBytes(bytes);
    onStatus?.call('Model ready.');
    return target.path;
  }

  Future<String?> _prepareIOSModel({
    required String modelName,
    ProgressCallback? onProgress,
    StatusCallback? onStatus,
  }) async {
    final bundleName = '$modelName.mlpackage';
    onStatus?.call('Checking bundle for $modelName...');

    try {
      final result = await _channel.invokeMethod<Map<dynamic, dynamic>>(
        'checkModelExists',
        {'modelPath': bundleName},
      );
      if (result != null && result['exists'] == true) {
        return bundleName;
      }
    } catch (_) {
      // Continue to cache/download path.
    }

    final docsDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${docsDir.path}/$bundleName');
    if (await modelDir.exists()) {
      final manifest = File('${modelDir.path}/Manifest.json');
      if (await manifest.exists()) {
        onStatus?.call('Using cached model');
        return modelDir.path;
      }
      await modelDir.delete(recursive: true);
    }

    onStatus?.call('Downloading segmentation model...');
    final bytes = await _downloadFile(
      '$_downloadBase/$bundleName.zip',
      onProgress: onProgress,
    );
    if (bytes == null || bytes.isEmpty) {
      onStatus?.call('Download failed.');
      return null;
    }

    final extracted = await _extractZip(bytes, modelDir, onStatus: onStatus);
    if (extracted == null) {
      onStatus?.call('Extracting model failed.');
      return null;
    }
    onStatus?.call('Model ready.');
    return extracted;
  }

  Future<List<int>?> _downloadFile(
    String url, {
    ProgressCallback? onProgress,
  }) async {
    try {
      final client = http.Client();
      final request = http.Request('GET', Uri.parse(url));
      final response = await client.send(request);
      final expected = response.contentLength ?? 0;
      final bytes = <int>[];
      var received = 0;

      await for (final chunk in response.stream) {
        bytes.addAll(chunk);
        received += chunk.length;
        if (expected > 0) {
          onProgress?.call(received / expected);
        }
      }
      onProgress?.call(1);
      client.close();
      return bytes;
    } catch (_) {
      return null;
    }
  }

  Future<String?> _extractZip(
    List<int> bytes,
    Directory target, {
    StatusCallback? onStatus,
  }) async {
    try {
      onStatus?.call('Extracting model package...');
      final archive = ZipDecoder().decodeBytes(bytes);
      if (await target.exists()) {
        await target.delete(recursive: true);
      }
      await target.create(recursive: true);

      String? rootPrefix;
      if (archive.isNotEmpty) {
        final firstEntry = archive.first.name;
        if (firstEntry.contains('/') &&
            firstEntry.split('/').first.endsWith('.mlpackage')) {
          rootPrefix = '${firstEntry.split('/').first}/';
        }
      }

      for (final file in archive) {
        var filename = file.name;
        if (rootPrefix != null) {
          if (filename == rootPrefix.replaceAll('/', '')) {
            continue;
          }
          if (filename.startsWith(rootPrefix)) {
            filename = filename.substring(rootPrefix.length);
          }
        }

        if (filename.isEmpty) continue;

        if (file.isFile) {
          final outFile = File('${target.path}/$filename');
          await outFile.parent.create(recursive: true);
          await outFile.writeAsBytes(file.content as List<int>);
        }
      }
      return target.path;
    } catch (_) {
      if (await target.exists()) {
        await target.delete(recursive: true);
      }
      return null;
    }
  }
}
