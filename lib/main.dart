import 'package:flutter/material.dart';
import 'features/segmentation/view/segmentation_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SegmentationApp());
}

class SegmentationApp extends StatelessWidget {
  const SegmentationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ultralytics Background Segmentation',
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          brightness: Brightness.dark,
          seedColor: Colors.tealAccent,
        ),
        useMaterial3: true,
      ),
      debugShowCheckedModeBanner: false,
      home: const SegmentationScreen(),
    );
  }
}
