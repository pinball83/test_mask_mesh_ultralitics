import 'package:flutter_test/flutter_test.dart';

import 'package:test_mask_mesh_ultralitics/main.dart';

void main() {
  testWidgets('Shows unsupported platform notice on non-mobile hosts',
      (tester) async {
    await tester.pumpWidget(const SegmentationApp());
    await tester.pumpAndSettle();

    expect(
      find.textContaining('supports Android or iOS devices'),
      findsOneWidget,
    );
  });
}
