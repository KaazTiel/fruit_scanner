import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: YOLOExample(),
      debugShowCheckedModeBanner: false,
    );
  }
}

// enum FilterType { good, bad }

class YOLOExample extends StatefulWidget {
  const YOLOExample({super.key});

  @override
  State<YOLOExample> createState() => _YOLOExampleState();
}

class _YOLOExampleState extends State<YOLOExample> {
  final YOLOViewController controller = YOLOViewController();

  List<YOLOResult> allResults = [];
  List<YOLOResult> filteredResults = [];

  double fps = 0;
  int? _lastFrameTimestamp;

  double _confidenceThreshold = 0.5;
  // FilterType _filter = FilterType.bad;

  bool _modelResponded = false;
  bool _modelHasResults = false;

  // final Set<String> badClasses = {
  //   "bad apple",
  //   "bad banana",
  //   "bad orange",
  //   "bad pomegranate",
  // };

  late Future<String> _modelFuture;

  @override
  void initState() {
    super.initState();
    _modelFuture = Future.value('best_nano_float32.tflite');

    // Inicializa o threshold no controller
    _setConfidenceThreshold(_confidenceThreshold);
  }

  Future<void> _setConfidenceThreshold(double value) async {
    await controller.setConfidenceThreshold(value);
    setState(() {
      _confidenceThreshold = value;
    });
    // _applyFilters();
  }

  // void _applyFilters() {
  //   filteredResults = allResults.where((r) {
  //     if (r.confidence < _confidenceThreshold) return false;
  //     switch (_filter) {
  //       case FilterType.good:
  //         return !badClasses.contains(r.className);
  //       case FilterType.bad:
  //         return badClasses.contains(r.className);
  //     }
  //   }).toList();
  //   setState(() {});
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Detector de Frutas"),
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Center(child: Text('FPS: ${fps.toStringAsFixed(1)}')),
          ),
          // PopupMenuButton<FilterType>(
          //   icon: const Icon(Icons.filter_list),
          //   onSelected: (FilterType selected) {
          //     setState(() => _filter = selected);
          //     _applyFilters();
          //   },
          //   itemBuilder: (context) => const [
          //     PopupMenuItem(value: FilterType.good, child: Text('Mostrar Boas')),
          //     PopupMenuItem(value: FilterType.bad, child: Text('Mostrar Ruins')),
          //   ],
          // ),
        ],
      ),
      body: FutureBuilder<String>(
        future: _modelFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError || !snapshot.hasData) {
            return const Center(child: Text('Falha ao carregar modelo'));
          }

          return Stack(
            children: [
              YOLOView(
                modelPath: snapshot.data!,
                task: YOLOTask.detect,
                controller: controller,
                onResult: (res) {
                  final now = DateTime.now().millisecondsSinceEpoch;
                  if (_lastFrameTimestamp != null) {
                    final delta = now - _lastFrameTimestamp!;
                    if (delta > 0) {
                      setState(() {
                        fps = 1000 / delta;
                      });
                    }
                  }
                  _lastFrameTimestamp = now;

                  if (!_modelResponded) {
                    setState(() {
                      _modelResponded = true;
                    });
                  }

                  allResults = res;
                  // _applyFilters();

                  setState(() {
                    _modelHasResults = res.isNotEmpty;
                  });

                  for (final r in filteredResults.take(3)) {
                    debugPrint(
                      '${r.className} ${(r.confidence * 100).toStringAsFixed(1)}% em ${r.boundingBox}',
                    );
                  }
                },
              ),

              // Feedback visual do modelo
              Align(
                alignment: Alignment.topCenter,
                child: Padding(
                  padding: const EdgeInsets.only(top: 20),
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 500),
                    child: !_modelResponded
                        ? const Chip(
                            key: ValueKey("loading"),
                            backgroundColor: Colors.orange,
                            label: Text(
                              "Modelo carregando...",
                              style: TextStyle(color: Colors.white),
                            ),
                          )
                        : Chip(
                            key: ValueKey("loaded"),
                            backgroundColor:
                                _modelHasResults ? Colors.green : Colors.blueGrey,
                            label: Text(
                              _modelHasResults ? "Modelo ativo!" : "Modelo sem detecções",
                              style: const TextStyle(color: Colors.white),
                            ),
                          ),
                  ),
                ),
              ),

              // Slider de confiança controlando controller
              Positioned(
                bottom: 50,
                left: 20,
                right: 20,
                child: Column(
                  children: [
                    Text("Confiança mínima: ${(_confidenceThreshold * 100).toInt()}%"),
                    Slider(
                      value: _confidenceThreshold,
                      min: 0,
                      max: 1,
                      divisions: 100,
                      label: "${(_confidenceThreshold * 100).toInt()}%",
                      onChanged: (v) {
                        _setConfidenceThreshold(v);
                      },
                    ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
