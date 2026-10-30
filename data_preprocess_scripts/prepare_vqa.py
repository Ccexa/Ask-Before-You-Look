#!/usr/bin/env python3
"""Combine VQA annotations and collect images for COMPASS PathVQA training."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, nargs='+', required=True,
                        help='Input JSON lists (TumorVQA, TumorVQA+, and/or converted PathVQA).')
    parser.add_argument('--image-dir', type=Path, action='append', required=True,
                        help='Directory to search recursively for images; may be repeated.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Dataset folder containing images/ and <split>_vqa.json.')
    parser.add_argument('--split', default='train', choices=('train', 'val', 'test'))
    parser.add_argument('--shuffle-seed', type=int, default=None,
                        help='Shuffle records reproducibly; default preserves input order.')
    args = parser.parse_args()

    records = []
    for annotation in args.annotations:
        with annotation.open(encoding='utf-8') as stream:
            data = json.load(stream)
        if not isinstance(data, list):
            raise ValueError(f'{annotation}: expected a JSON list')
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f'{annotation}, record {i}: expected an object')
            image, turns = record.get('image'), record.get('conversations')
            if not isinstance(image, str) or not image or Path(image).name != image or image in ('.', '..'):
                raise ValueError(f'{annotation}, record {i}: image must be a filename')
            if not isinstance(turns, list) or len(turns) < 2 or len(turns) % 2:
                raise ValueError(f'{annotation}, record {i}: expected human/gpt conversation pairs')
            for j, turn in enumerate(turns):
                role = 'human' if j % 2 == 0 else 'gpt'
                if not isinstance(turn, dict) or turn.get('from') != role or not isinstance(turn.get('value'), str):
                    raise ValueError(f'{annotation}, record {i}, turn {j}: expected {role} and a string value')
            records.append(record)
    if not records:
        raise ValueError('No annotations found')

    needed = {record['image'] for record in records}
    found = {}
    for directory in args.image_dir:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        for path in directory.rglob('*'):
            if path.is_file() and path.name in needed:
                found.setdefault(path.name, []).append(path)
    missing = sorted(needed - found.keys())
    if missing:
        raise FileNotFoundError(f'{len(missing)} images missing; examples: {missing[:10]}')

    image_dir = args.output_dir / 'images'
    for name, paths in found.items():
        if len(paths) > 1 and any(digest(p) != digest(paths[0]) for p in paths[1:]):
            raise ValueError(f'Different images share filename {name}: {paths}')
        target = image_dir / name
        if target.exists() and digest(target) != digest(paths[0]):
            raise FileExistsError(f'Existing image differs: {target}')

    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(records)
    output = args.output_dir / f'{args.split}_vqa.json'
    encoded = json.dumps(records, ensure_ascii=False, indent=2) + '\n'
    if output.exists() and output.read_text(encoding='utf-8') != encoded:
        raise FileExistsError(f'Refusing to overwrite existing annotations: {output}')
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in found.items():
        target = image_dir / name
        if not target.exists():
            shutil.copy2(paths[0], target)
    if not output.exists():
        output.write_text(encoded, encoding='utf-8')
    print(f'{output}: {len(records)} records; {len(needed)} images in {image_dir}')


if __name__ == '__main__':
    main()
