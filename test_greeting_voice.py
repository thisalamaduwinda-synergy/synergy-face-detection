#!/usr/bin/env python3
"""
test_greeting_voice.py
──────────────────────────────────────────────────────────────
Quick test script to verify greeting service functionality.
Tests both:
  1. Greeting service initialization
  2. Text-to-speech generation
  3. Audio playback
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def load_config() -> Dict[str, Any]:
    """Load config.yaml"""
    import yaml
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available"""
    deps = {}
    
    try:
        import edge_tts
        deps["edge_tts"] = True
        print("✓ edge-tts installed")
    except ImportError:
        deps["edge_tts"] = False
        print("✗ edge-tts NOT installed (required for neural voice)")
    
    try:
        import pyttsx3
        deps["pyttsx3"] = True
        print("✓ pyttsx3 installed")
    except ImportError:
        deps["pyttsx3"] = False
        print("✗ pyttsx3 NOT installed (fallback TTS)")
    
    try:
        import pygame
        deps["pygame"] = True
        print("✓ pygame installed")
    except ImportError:
        deps["pygame"] = False
        print("✗ pygame NOT installed (required for local audio playback)")
    
    try:
        import yaml
        deps["pyyaml"] = True
        print("✓ PyYAML installed")
    except ImportError:
        deps["pyyaml"] = False
        print("✗ PyYAML NOT installed")
    
    try:
        import numpy
        deps["numpy"] = True
        print("✓ numpy installed")
    except ImportError:
        deps["numpy"] = False
        print("✗ numpy NOT installed")
    
    return deps

def test_edge_tts() -> bool:
    """Test edge-tts neural voice generation"""
    print("\n" + "="*60)
    print("TEST 1: edge-tts Neural Voice Generation")
    print("="*60)
    
    try:
        import edge_tts
        import asyncio
        import tempfile
        
        async def generate_speech():
            text = "Hi, John Smith"
            voice = "en-US-AriaNeural"
            
            print(f"Generating speech: '{text}'")
            print(f"Voice: {voice}")
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            
            communicate = edge_tts.Communicate(text, voice=voice)
            await communicate.save(tmp_path)
            
            import os
            file_size = os.path.getsize(tmp_path)
            print(f"✓ Generated MP3 file: {tmp_path} ({file_size} bytes)")
            
            # Cleanup
            os.unlink(tmp_path)
            return True
        
        asyncio.run(generate_speech())
        print("✓ edge-tts test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ edge-tts test FAILED: {e}")
        return False

def test_pyttsx3() -> bool:
    """Test pyttsx3 fallback TTS"""
    print("\n" + "="*60)
    print("TEST 2: pyttsx3 Fallback TTS")
    print("="*60)
    
    try:
        import pyttsx3
        
        print("Initializing pyttsx3...")
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)
        
        voices = engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name}")
        
        text = "Hi, Jane Doe"
        print(f"\nGenerating speech: '{text}'")
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        
        import os
        if os.path.exists(tmp_path):
            file_size = os.path.getsize(tmp_path)
            print(f"✓ Generated WAV file: {tmp_path} ({file_size} bytes)")
            os.unlink(tmp_path)
        
        print("✓ pyttsx3 test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ pyttsx3 test FAILED: {e}")
        return False

def test_greeting_service_init(cfg: Dict[str, Any]) -> bool:
    """Test GreetingService initialization"""
    print("\n" + "="*60)
    print("TEST 3: GreetingService Initialization")
    print("="*60)
    
    try:
        from modules.greeting_service import GreetingService
        
        print("Creating GreetingService instance...")
        svc = GreetingService(cfg, sdk=None)
        
        print(f"  enabled: {svc.enabled}")
        print(f"  output: {svc.output}")
        print(f"  tts_engine: {svc.tts_engine}")
        print(f"  template: {svc.template}")
        print(f"  vip_template: {svc.vip_template}")
        print(f"  voice_name: {svc.voice_name}")
        print(f"  cooldown_seconds: {svc.cooldown}")
        
        print("\nStarting service...")
        svc.start()
        time.sleep(0.5)
        
        print("\nQueuing test greeting...")
        result = svc.greet(
            employee_id="EMP001",
            employee_name="John Smith",
            is_vip=False
        )
        print(f"Greeting queued: {result}")
        
        print("Waiting for audio processing...")
        time.sleep(3)
        
        svc.stop()
        print("✓ GreetingService test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ GreetingService test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vip_greeting(cfg: Dict[str, Any]) -> bool:
    """Test VIP greeting message"""
    print("\n" + "="*60)
    print("TEST 4: VIP Greeting Message")
    print("="*60)
    
    try:
        from modules.greeting_service import GreetingService
        
        svc = GreetingService(cfg, sdk=None)
        svc.start()
        time.sleep(0.5)
        
        print("Queuing VIP greeting...")
        result = svc.greet(
            employee_id="EMP_VIP_001",
            employee_name="Alice Johnson",
            is_vip=True
        )
        print(f"VIP greeting queued: {result}")
        
        print("Waiting for audio processing...")
        time.sleep(3)
        
        svc.stop()
        print("✓ VIP Greeting test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ VIP Greeting test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_report(deps: Dict[str, bool], results: Dict[str, bool]) -> None:
    """Generate test report"""
    print("\n\n" + "="*60)
    print("GREETING VOICE TEST REPORT")
    print("="*60)
    
    print("\n📦 DEPENDENCIES:")
    critical_deps = ["edge_tts", "pygame", "pyyaml", "numpy"]
    fallback_deps = ["pyttsx3"]
    
    missing_critical = [d for d in critical_deps if not deps.get(d, False)]
    missing_fallback = [d for d in fallback_deps if not deps.get(d, False)]
    
    if missing_critical:
        print(f"  ⚠️  CRITICAL MISSING: {', '.join(missing_critical)}")
    else:
        print("  ✓ All critical dependencies installed")
    
    if missing_fallback:
        print(f"  ℹ️  Fallback missing: {', '.join(missing_fallback)}")
    else:
        print("  ✓ Fallback TTS available")
    
    print("\n🧪 TEST RESULTS:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\n📊 Summary: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count and not missing_critical:
        print("\n🎉 GREETING VOICE FEATURE IS WORKING CORRECTLY!")
    elif passed_count == total_count:
        print("\n⚠️  Tests passed but some dependencies are missing")
        print("   Install with: pip install edge-tts pygame")
    else:
        print("\n❌ Some tests failed - see details above")

def main():
    print("GREETING VOICE VERIFICATION TEST")
    print("="*60)
    
    # Load config
    try:
        cfg = load_config()
        print("✓ Config loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return 1
    
    # Check dependencies
    print("CHECKING DEPENDENCIES:")
    print("-"*60)
    deps = check_dependencies()
    
    results = {}
    
    # Run tests
    if deps.get("edge_tts"):
        results["edge-tts Neural Voice"] = test_edge_tts()
    else:
        print("\nℹ️  Skipping edge-tts test (not installed)")
    
    if deps.get("pyttsx3"):
        results["pyttsx3 Fallback TTS"] = test_pyttsx3()
    else:
        print("\nℹ️  Skipping pyttsx3 test (not installed)")
    
    if deps.get("pyyaml") and deps.get("numpy"):
        results["GreetingService Init"] = test_greeting_service_init(cfg)
    else:
        print("\nℹ️  Skipping GreetingService test (missing dependencies)")
    
    if deps.get("pyyaml") and deps.get("numpy") and (deps.get("edge_tts") or deps.get("pyttsx3")):
        results["VIP Greeting"] = test_vip_greeting(cfg)
    else:
        print("\nℹ️  Skipping VIP greeting test (missing dependencies)")
    
    # Generate report
    generate_report(deps, results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
