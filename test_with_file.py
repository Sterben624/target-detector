"""
Test detector with actual audio file (bypassing microphone)
"""
import numpy as np
import librosa
from shahed_detector.shahed_detector_module import ShahedDetector

def test_audio_file(file_path, mode='quad_plus_shahed', use_bandpass=False, model_path=None):
    """Test detector on a clean audio file"""

    # Доступні режими
    MODES = {
        'quad_plus_shahed': {
            'fullband': 'shahed_detector/models/quad_plus_shahed_fullband.pth',
            'bandpass': 'shahed_detector/models/quad_plus_shahed_bandpass.pth',
        },
        'quad_plus_pure_shahed': {
            'fullband': 'shahed_detector/models/quad_plus_pure_shahed_fullband.pth',
            'bandpass': 'shahed_detector/models/quad_plus_pure_shahed_bandpass.pth',
        }
    }

    # Load detector
    if model_path is None:
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(MODES.keys())}")

        band_type = 'bandpass' if use_bandpass else 'fullband'
        model_path = MODES[mode][band_type]

    detector = ShahedDetector(
        model_path=model_path,
        use_bandpass=use_bandpass,
        sample_rate=16000,
        n_mfcc=40,
        n_mels=128,
        duration=1.0
    )

    # Load audio file
    print(f"\nLoading: {file_path}")
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    print(f"Loaded: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f} sec)")

    # Process in 1-second chunks
    chunk_samples = int(sr * 1.0)

    results = []
    for i in range(0, len(audio), chunk_samples // 2):  # 50% overlap
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            break

        result = detector.predict(chunk, original_sr=sr)
        results.append(result)

        prob = result['probability']
        conf = result['confidence']
        pred = result['predicted_class']
        cls = result['class_name']

        timestamp = i / sr
        print(f"[{timestamp:5.1f}s] {cls:12s} | Prob: {prob:.3f}, Conf: {conf:.3f} | "
              f"[No:{result['probabilities']['без_шахеда']:.3f} | "
              f"Yes:{result['probabilities']['з_шахедом']:.3f}]")

    # Summary
    shahed_count = sum(1 for r in results if r['predicted_class'] == 1)
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(results)}")
    print(f"Shahed detected: {shahed_count} ({shahed_count/len(results)*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="ResNet6-CBAM Shahed Detector - File Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  quad_plus_shahed         - Quadcopter + Shahed (default)
  quad_plus_pure_shahed    - Quadcopter + Shahed + Pure Shahed

Examples:
  python test_with_file.py my_target_only.wav
  python test_with_file.py my_target_only.wav --mode quad_plus_shahed --bandpass
  python test_with_file.py my_target_only.wav --mode quad_plus_pure_shahed
        """
    )
    parser.add_argument('file', type=str, help='Audio file path (.wav)')
    parser.add_argument('--mode', type=str, default='quad_plus_shahed',
                       choices=['quad_plus_shahed', 'quad_plus_pure_shahed'],
                       help='Detection mode (default: quad_plus_shahed)')
    parser.add_argument('--bandpass', action='store_true',
                       help='Use bandpass filter 700-850 Hz')
    parser.add_argument('--model', type=str, default=None,
                       help='Custom model path (overrides --mode)')

    args = parser.parse_args()

    print("="*60)
    print("ResNet6-CBAM Shahed Detector - File Test")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Filter: {'Bandpass (700-850 Hz)' if args.bandpass else 'Full Spectrum'}")

    test_audio_file(args.file, mode=args.mode, use_bandpass=args.bandpass, model_path=args.model)
