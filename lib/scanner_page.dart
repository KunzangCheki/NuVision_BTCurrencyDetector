import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ScannerPage extends StatefulWidget {
  @override
  _ScannerPageState createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> {
  CameraController? _controller;
  late List<CameraDescription> _cameras;
  String _predictionResult = '';
  final FlutterTts flutterTts = FlutterTts();

  @override
  void initState() {
    super.initState();
    _initCamera();
    Future.delayed(Duration(seconds: 1), _speakInstruction);
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras[0], ResolutionPreset.medium);
    await _controller!.initialize();
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _speakInstruction() async {
    await flutterTts.setLanguage("en-US");
    await flutterTts.setSpeechRate(0.45);
    await flutterTts.speak("Double tap to classify the bank note.");
  }

  Future<void> _speakText(String text) async {
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.speak(text);
  }

  Future<void> _captureAndPredict() async {
    if (!_controller!.value.isInitialized) return;

    try {
      final image = await _controller!.takePicture();
      final bytes = await image.readAsBytes();
      final base64Image = base64Encode(bytes);

      final uri = Uri.parse("http://10.9.89.110:5000/predict");

      final response = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "data": ["data:image/jpeg;base64,$base64Image"]
        }),
      );

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        final result = decoded['result'].toString();
        setState(() => _predictionResult = "Detected: $result");
        await _speakText("Detected $result");
      } else {
        setState(() => _predictionResult = "Detection failed.");
        await _speakText("Detection failed.");
      }
    } catch (e) {
      setState(() => _predictionResult = "Error occurred.");
      await _speakText("Error during detection.");
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    flutterTts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onDoubleTap: _captureAndPredict,
      onVerticalDragUpdate: (details) {
        if (details.primaryDelta! > 20) {
          Navigator.of(context).pop(); // Downward swipe → back
        }
      },
      onHorizontalDragUpdate: (details) {
        if (details.primaryDelta! > 20) {
          Navigator.pushReplacementNamed(context, '/summation'); // → right
        }
      },
      child: Scaffold(
        backgroundColor: Colors.white,
        body: SafeArea(
          child: Column(
            children: [
              AppBar(
                title: Text("Classify the Bank Notes"),
                backgroundColor: const Color.fromRGBO(255, 152, 0, 1),
                automaticallyImplyLeading: false,
              ),
              _controller == null || !_controller!.value.isInitialized
                  ? Expanded(child: Center(child: CircularProgressIndicator()))
                  : Expanded(
                      child: Container(
                        margin: EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          border: Border.all(
                              color: const Color.fromRGBO(121, 85, 72, 1),
                              width: 3),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: CameraPreview(_controller!),
                        ),
                      ),
                    ),
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Text(
                  _predictionResult,
                  style: TextStyle(fontSize: 18, color: Colors.black),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
