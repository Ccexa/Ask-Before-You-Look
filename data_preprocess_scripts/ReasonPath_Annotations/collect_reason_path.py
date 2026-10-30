#!/usr/bin/env python3
"""Collect image patches and masks into the PathInstruct dataset layout."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.digest()


def index_files(directories, filenames):
    found = {}
    for directory in directories:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        for path in directory.rglob('*'):
            if path.is_file() and path.name in filenames:
                found.setdefault(path.name, []).append(path)
    missing = sorted(filenames - found.keys())
    if missing:
        raise FileNotFoundError(f'{len(missing)} files missing in {directories}; examples: {missing[:10]}')
    for name, paths in found.items():
        if len(paths) > 1:
            first = sha256(paths[0])
            if any(sha256(path) != first for path in paths[1:]):
                raise ValueError(f'Conflicting files named {name}: {paths}')
    return {name: paths[0] for name, paths in found.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations-dir', type=Path,
                        default=Path(__file__).resolve().parent,
                        help='Folder containing train.json, validation.json and/or test.json.')
    parser.add_argument('--image-dir', type=Path, action='append', required=True,
                        help='Cropped image directory; may be repeated.')
    parser.add_argument('--mask-dir', type=Path, action='append', required=True,
                        help='Cropped segmentation mask directory; may be repeated.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Dataset directory, e.g. DATASET_ROOT/PathInstruct.')
    parser.add_argument('--splits', nargs='+', choices=('train', 'validation', 'test'),
                        default=None, help='Requested splits; by default use available JSONs.')
    args = parser.parse_args()

    splits = args.splits or [s for s in ('train', 'validation', 'test')
                             if (args.annotations_dir / f'{s}.json').is_file()]
    if not splits:
        raise ValueError(f'No split JSONs found in {args.annotations_dir}')
    if len(set(splits)) != len(splits):
        raise ValueError('Duplicate split requested')
    manifests = {}
    names = set()
    for split in splits:
        path = args.annotations_dir / f'{split}.json'
        with path.open(encoding='utf-8') as stream:
            records = json.load(stream)
        if not isinstance(records, list) or not records:
            raise ValueError(f'{path}: expected a nonempty JSON list')
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f'{path}, record {i}: expected an object')
            name = record.get('filename')
            if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
                raise ValueError(f'{path}, record {i}: invalid filename')
            if not all(isinstance(record.get(key), str) for key in ('query', 'explanation')):
                raise ValueError(f'{path}, record {i}: query and explanation must be strings')
            names.add(name)
        manifests[split] = (path, records)

    images = index_files(args.image_dir, names)
    masks = index_files(args.mask_dir, names)
    copies = {}
    for split, (manifest, records) in manifests.items():
        destination = args.output_dir / split
        copies[destination / 'data.json'] = manifest
        for record in records:
            name = record['filename']
            copies[destination / 'images' / name] = images[name]
            copies[destination / 'annotations' / name] = masks[name]
    for target, source in copies.items():
        if target.exists() and (not target.is_file() or sha256(target) != sha256(source)):
            raise FileExistsError(f'Existing file differs: {target}')
    for target, source in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    for split, (_, records) in manifests.items():
        print(f'{split}: {len(records)} records -> {args.output_dir / split}')


if __name__ == '__main__':
    main()
