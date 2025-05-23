import 'package:currency_detector/home_page.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'scanner_page.dart'; // Classification-only page
import 'summation_page.dart'; // Summation-only page

List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(CurrencyDetectorApp());
}

class CurrencyDetectorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NuVision',
      initialRoute: '/',
      routes: {
        '/': (context) => WelcomePage(),
        '/scanner': (context) => ScannerPage(), 
        '/summation': (context) => SummationPage(),
      },
      debugShowCheckedModeBanner: false,
    );
  }
}