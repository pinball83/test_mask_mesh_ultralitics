# Repository Guidelines

This guide summarizes expectations for contributors working inside `test_mask_mesh_ultralitics`, a Flutter playground for experimenting with Ultralytics-driven background segmentation and face mask binding. Keep it close as you explore the codebase, and update it whenever workflows evolve.

## Project Structure & Module Organization
Application code lives in `lib/`, with `lib/main.dart` as the entry point for configuring widgets, routes, and dependency injection. Keep feature-specific widgets and services in dedicated subdirectories under `lib/` to avoid bloating the root.
Tests belong in `test/`; mirror the `lib/` structure when adding new suites so related files are easy to locate. Platform scaffolding stays in `android/` and `ios/`, which should only change when platform integrations or build settings require it.

## Build, Test, and Development Commands
Run `flutter pub get` after dependency updates to hydrate `.dart_tool/`. `flutter analyze` performs static analysis using the project lint rules. Execute `flutter test` for widget and unit suites; add `--coverage` when reviewing overall hit rates. Use `flutter run -d <device>` to iterate locally on a specific emulator or physical device.

## Coding Style & Naming Conventions
The repository inherits `flutter_lints` via `analysis_options.yaml`; resolve analyzer findings before pushing. Use Dart’s standard two-space indentation and prefer `dart format .` (or the IDE equivalent) to keep diffs clean. Adopt `UpperCamelCase` for classes, `lowerCamelCase` for methods and fields, and `snake_case` for file names such as `face_mask_panel.dart`. Keep widgets small and composable—extract pure helpers when build methods exceed ~100 lines.

## Testing Guidelines
Expand on the starter `test/widget_test.dart` by co-locating new specs alongside their feature counterparts (e.g., `test/widgets/face_mask_panel_test.dart`). Favor descriptive test names that read like behavior statements (`should_render_mask_controls`). High-risk changes should include golden tests or integration coverage when feasible. Before opening a pull request, confirm `flutter analyze` and `flutter test --coverage` both pass locally.

## Commit & Pull Request Guidelines
Commit frequently following Conventional Commits (`feat:`, `fix:`, `refactor:`) so changelog automation stays viable. Each pull request should summarize the user-facing impact, list validation steps (including analyzer/tests), and reference tracking issues. When UI shifts are involved, attach emulator screenshots or short screen recordings. Draft and review branch histories remain linear—rebase instead of merge to keep history readable.
