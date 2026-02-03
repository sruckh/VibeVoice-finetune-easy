#!/usr/bin/env python3
"""
VibeVoice Dataset Preparation Script

This script prepares your audio data for fine-tuning VibeVoice models.
It supports various input formats and creates a properly formatted JSONL file.

Usage:
    python prepare_dataset.py --audio_dir /path/to/audio --output dataset.jsonl
    python prepare_dataset.py --audio_dir /path/to/audio --transcript_dir /path/to/transcripts --output dataset.jsonl
    python prepare_dataset.py --csv metadata.csv --output dataset.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import re

# Try to import audio processing libraries
try:
    import soundfile as sf
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False
    print("Warning: soundfile not installed. Audio validation will be limited.")

try:
    import librosa
    LIBROSA_SUPPORT = True
except ImportError:
    LIBROSA_SUPPORT = False


def get_audio_files(directory: str, extensions: tuple = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')) -> List[Path]:
    """Recursively find all audio files in directory."""
    audio_dir = Path(directory)
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.rglob(f'*{ext}'))
    return sorted(audio_files)


def validate_audio_file(filepath: Path, target_sr: int = 24000) -> Dict[str, Any]:
    """Validate audio file and return metadata."""
    info = {
        'valid': False,
        'duration': 0,
        'sample_rate': 0,
        'error': None
    }
    
    if not filepath.exists():
        info['error'] = f"File not found: {filepath}"
        return info
    
    try:
        if AUDIO_SUPPORT:
            data, sr = sf.read(str(filepath))
            info['duration'] = len(data) / sr
            info['sample_rate'] = sr
            info['valid'] = True
        elif LIBROSA_SUPPORT:
            data, sr = librosa.load(str(filepath), sr=None)
            info['duration'] = librosa.get_duration(y=data, sr=sr)
            info['sample_rate'] = sr
            info['valid'] = True
        else:
            # Basic validation - just check file size
            size_mb = filepath.stat().st_size / (1024 * 1024)
            info['duration'] = size_mb * 10  # Rough estimate
            info['valid'] = True
    except Exception as e:
        info['error'] = str(e)
    
    return info


def transcribe_audio_whisper(audio_path: Path, model_size: str = "base") -> str:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper
        
        print(f"  Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)
        
        print(f"  Transcribing {audio_path.name}...")
        result = model.transcribe(str(audio_path))
        
        return result["text"].strip()
    except ImportError:
        print("  Error: Whisper not installed. Install with: pip install openai-whisper")
        return ""
    except Exception as e:
        print(f"  Error transcribing: {e}")
        return ""


def find_transcript(audio_path: Path, transcript_dir: Optional[Path] = None) -> Optional[str]:
    """Find transcript for audio file."""
    # Try same directory with .txt extension
    txt_path = audio_path.with_suffix('.txt')
    if txt_path.exists():
        return txt_path.read_text(encoding='utf-8').strip()
    
    # Try with .json extension (common format)
    json_path = audio_path.with_suffix('.json')
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            if 'text' in data:
                return data['text']
            if 'transcript' in data:
                return data['transcript']
        except:
            pass
    
    # Try transcript directory
    if transcript_dir:
        txt_path = transcript_dir / f"{audio_path.stem}.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding='utf-8').strip()
    
    return None


def auto_detect_speakers(text: str) -> List[str]:
    """Automatically detect speaker patterns in text."""
    # Common patterns: "Speaker 0:", "Speaker 1:", "Person A:", etc.
    patterns = [
        r'Speaker \d+',
        r'Person [A-Z]',
        r'[A-Z][a-z]+:',
    ]
    
    speakers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        speakers.update(matches)
    
    return list(speakers)


def create_dataset_entry(
    audio_path: Path,
    text: str,
    voice_prompt: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create a dataset entry in VibeVoice format."""
    entry = {
        "text": text,
        "audio": str(audio_path.absolute())
    }
    
    if voice_prompt:
        entry["voice_prompts"] = voice_prompt
    
    if metadata:
        entry["metadata"] = metadata
    
    return entry


def prepare_from_directory(args) -> List[Dict]:
    """Prepare dataset from audio directory."""
    print(f"Scanning directory: {args.audio_dir}")
    
    audio_files = get_audio_files(args.audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("No audio files found!")
        return []
    
    entries = []
    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else None
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_path.name}")
        
        # Validate audio
        info = validate_audio_file(audio_path)
        if not info['valid']:
            print(f"  ✗ Skipping: {info['error']}")
            continue
        
        print(f"  Duration: {info['duration']:.2f}s, SR: {info['sample_rate']}Hz")
        
        # Get transcript
        text = None
        
        # Try to find existing transcript
        if not args.auto_transcribe:
            text = find_transcript(audio_path, transcript_dir)
            if text:
                print(f"  ✓ Found transcript")
        
        # Auto-transcribe if needed
        if text is None and args.auto_transcribe:
            text = transcribe_audio_whisper(audio_path, args.whisper_model)
            if text:
                print(f"  ✓ Transcribed: {text[:80]}...")
        
        # Use placeholder if no transcript
        if text is None:
            if args.allow_empty:
                text = "Please transcribe this audio."
                print(f"  ⚠ No transcript found, using placeholder")
            else:
                print(f"  ✗ Skipping: No transcript found (use --allow-empty to include anyway)")
                continue
        
        # Format text with speaker label if requested
        if args.speaker_prefix and not text.startswith("Speaker"):
            text = f"{args.speaker_prefix}: {text}"
        
        # Handle voice prompts for multi-speaker
        voice_prompt = None
        if args.voice_prompts_dir:
            vp_dir = Path(args.voice_prompts_dir)
            # Look for matching voice prompt or use random one
            vp_files = get_audio_files(vp_dir)
            if vp_files:
                voice_prompt = str(random.choice(vp_files).absolute())
                print(f"  ✓ Assigned voice prompt")
        
        entry = create_dataset_entry(
            audio_path=audio_path,
            text=text,
            voice_prompt=voice_prompt,
            metadata={
                'duration': info['duration'],
                'original_filename': audio_path.name
            }
        )
        entries.append(entry)
    
    return entries


def prepare_from_csv(args) -> List[Dict]:
    """Prepare dataset from CSV file."""
    import csv
    
    print(f"Reading CSV: {args.csv}")
    
    entries = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Detect columns
        fieldnames = reader.fieldnames or []
        print(f"CSV columns: {fieldnames}")
        
        # Map common column names
        audio_col = args.csv_audio_col or next(
            (c for c in fieldnames if c.lower() in ['audio', 'audio_path', 'file', 'path', 'filename']),
            fieldnames[0] if fieldnames else None
        )
        text_col = args.csv_text_col or next(
            (c for c in fieldnames if c.lower() in ['text', 'transcript', 'transcription', 'sentence', 'prompt']),
            fieldnames[1] if len(fieldnames) > 1 else None
        )
        
        print(f"Using columns: audio='{audio_col}', text='{text_col}'")
        
        for i, row in enumerate(reader, 1):
            audio_path = Path(row[audio_col]) if audio_col else None
            text = row[text_col] if text_col else None
            
            # Resolve relative paths
            if audio_path and not audio_path.is_absolute():
                audio_path = Path(args.audio_dir) / audio_path if args.audio_dir else audio_path
            
            if audio_path and audio_path.exists() and text:
                entry = create_dataset_entry(
                    audio_path=audio_path,
                    text=text,
                    metadata={'csv_row': i}
                )
                entries.append(entry)
            else:
                print(f"  Row {i}: Skipping (missing audio or text)")
    
    print(f"Created {len(entries)} entries from CSV")
    return entries


def split_dataset(entries: List[Dict], train_ratio: float = 0.9) -> tuple:
    """Split dataset into train and validation sets."""
    random.shuffle(entries)
    split_idx = int(len(entries) * train_ratio)
    return entries[:split_idx], entries[split_idx:]


def write_jsonl(entries: List[Dict], filepath: Path):
    """Write entries to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(entries)} entries to {filepath}")


def validate_dataset(entries: List[Dict]) -> Dict[str, Any]:
    """Validate dataset and return statistics."""
    stats = {
        'total_entries': len(entries),
        'with_voice_prompts': sum(1 for e in entries if 'voice_prompts' in e),
        'with_metadata': sum(1 for e in entries if 'metadata' in e),
        'avg_text_length': sum(len(e.get('text', '')) for e in entries) / max(len(entries), 1),
        'audio_extensions': {}
    }
    
    for entry in entries:
        audio_path = entry.get('audio', '')
        ext = Path(audio_path).suffix.lower()
        stats['audio_extensions'][ext] = stats['audio_extensions'].get(ext, 0) + 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for VibeVoice fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - audio files with auto-generated transcripts
  python prepare_dataset.py --audio_dir ./my_audio --auto-transcribe --output dataset.jsonl
  
  # With separate transcript files
  python prepare_dataset.py --audio_dir ./audio --transcript_dir ./transcripts --output dataset.jsonl
  
  # From CSV metadata
  python prepare_dataset.py --csv metadata.csv --audio_col audio_path --text_col transcript --output dataset.jsonl
  
  # Multi-speaker with voice prompts
  python prepare_dataset.py --audio_dir ./audio --voice_prompts_dir ./prompts --output dataset.jsonl
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio_dir', type=str, help='Directory containing audio files')
    input_group.add_argument('--csv', type=str, help='CSV file with metadata')
    
    # Transcript options
    parser.add_argument('--transcript_dir', type=str, help='Directory containing transcript files')
    parser.add_argument('--auto_transcribe', '--auto-transcribe', action='store_true',
                        help='Auto-transcribe audio using Whisper')
    parser.add_argument('--whisper_model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size for transcription (default: base)')
    parser.add_argument('--allow_empty', action='store_true',
                        help='Include entries without transcripts (with placeholder text)')
    
    # CSV options
    parser.add_argument('--csv_audio_col', type=str, help='CSV column name for audio paths')
    parser.add_argument('--csv_text_col', type=str, help='CSV column name for text/transcripts')
    
    # Voice prompt options
    parser.add_argument('--voice_prompts_dir', type=str,
                        help='Directory containing voice prompt audio files')
    
    # Text formatting
    parser.add_argument('--speaker_prefix', type=str, default='Speaker 0',
                        help='Prefix for single-speaker text (default: "Speaker 0")')
    
    # Output options
    parser.add_argument('--output', type=str, default='data/dataset.jsonl',
                        help='Output JSONL file path (default: data/dataset.jsonl)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--no_val_split', action='store_true',
                        help='Do not create validation split')
    
    # Processing options
    parser.add_argument('--validate_audio', action='store_true', default=True,
                        help='Validate audio files (default: True)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VibeVoice Dataset Preparation")
    print("=" * 60)
    
    # Prepare dataset
    if args.audio_dir:
        entries = prepare_from_directory(args)
    elif args.csv:
        entries = prepare_from_csv(args)
    else:
        print("Error: Must specify either --audio_dir or --csv")
        sys.exit(1)
    
    if not entries:
        print("\nNo valid entries found! Please check your input data.")
        sys.exit(1)
    
    # Validate and show statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    stats = validate_dataset(entries)
    print(f"Total entries: {stats['total_entries']}")
    print(f"With voice prompts: {stats['with_voice_prompts']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} characters")
    print(f"Audio formats: {stats['audio_extensions']}")
    
    # Write output files
    print("\n" + "=" * 60)
    print("Writing Output Files")
    print("=" * 60)
    
    output_path = Path(args.output)
    
    if args.no_val_split or args.val_split <= 0:
        write_jsonl(entries, output_path)
    else:
        train_entries, val_entries = split_dataset(entries, 1 - args.val_split)
        
        # Write train set
        train_path = output_path.with_suffix('.train.jsonl')
        write_jsonl(train_entries, train_path)
        
        # Write validation set
        val_path = output_path.with_suffix('.val.jsonl')
        write_jsonl(val_entries, val_path)
        
        # Also write a combined file
        write_jsonl(entries, output_path)
        
        print(f"\nDataset split: {len(train_entries)} train, {len(val_entries)} validation")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext step: Train your model with:")
    print(f"  python train.py --train_dataset {output_path}")


if __name__ == '__main__':
    main()
