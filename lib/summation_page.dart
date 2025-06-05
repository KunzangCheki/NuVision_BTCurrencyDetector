import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:vibration/vibration.dart';
import 'dart:convert';
import 'dart:async';

class SummationPage extends StatefulWidget {
  @override
  _SummationPageState createState() => _SummationPageState();
}

class _SummationPageState extends State<SummationPage> {
  CameraController? _controller;
  late List<CameraDescription> _cameras;
  String _predictionResult = '';
  final FlutterTts flutterTts = FlutterTts();
  double _totalAmount = 0;
  Timer? _resetTimer;

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
    await flutterTts.speak("Double tap to scan and add bank notes.");
  }

  Future<void> _speakText(String text) async {
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.speak(text);
  }

  void _startResetTimer() {
    _resetTimer?.cancel();
    _resetTimer = Timer(Duration(seconds: 10), () {
      setState(() {
        _totalAmount = 0;
        _predictionResult = "Summation reset after 10 seconds.";
      });
      _speakText("Summation stopped and reset after 10 seconds of inactivity.");
    });
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
        final match = RegExp(r'\d+').firstMatch(result);

        if (match != null) {
          final amount = double.parse(match.group(0)!);
          setState(() {
            _totalAmount += amount;
            _predictionResult = "Detected: $result\nTotal: Nu $_totalAmount";
          });

          if (await Vibration.hasVibrator() ?? false) {
            Vibration.vibrate(duration: 300);
          }

          await _speakText("Detected $result. Total is $_totalAmount");
          _startResetTimer();
        } else {
          setState(() => _predictionResult = "Could not extract value.");
          await _speakText("Could not extract value.");
        }
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
    _resetTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onDoubleTap: _captureAndPredict,
      onVerticalDragUpdate: (details) {
        if (details.primaryDelta! > 20) {
          Navigator.of(context).pop(); // swipe down
        }
      },
      onHorizontalDragUpdate: (details) {
        if (details.primaryDelta! < -20) {
          Navigator.pushReplacementNamed(context, '/scanner'); // → left
        }
      },
      child: Scaffold(
        backgroundColor: Colors.white,
        body: SafeArea(
          child: Column(
            children: [
              AppBar(
                title: Text("Sum the Bank Notes"),
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
