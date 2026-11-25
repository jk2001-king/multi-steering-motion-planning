"""Read-only offline relocation checks; writes reports, never runs an experiment."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'migration'
OLD = Path(json.loads((META/'relocation.json').read_text())['original_root'])


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-study', type=Path,
                        help='Preserved/trashed original IJCAS study for a full comparison')
    args = parser.parse_args()
    manifest = json.loads((META/'relocation.json').read_text())
    changed = {row['path'] for row in manifest['changed_active_files']}
    changed.update(manifest['offline_readers_with_explicit_path_mapping'])
    retirement_path = META/'publication_cleanup.json'
    retirement = json.loads(retirement_path.read_text()) if retirement_path.exists() else {}
    retired = {row['path']: row for row in retirement.get('retired_files', [])}
    moved = retirement.get('technical_data_moves', {})
    runtime_cleanup_path = META/'runtime_cleanup.json'
    runtime_cleanup = json.loads(runtime_cleanup_path.read_text()) if runtime_cleanup_path.exists() else {}
    archived_runtime = {row['path']:row for row in runtime_cleanup.get('archived_python_sources', [])}
    runtime_archive = Path(runtime_cleanup['archive_root']) if archived_runtime else None
    archived_source_files_verified = 0
    allowed_retirement_prefixes = ('studies/IJCAS/paper/', 'studies/IJCAS/figures/')
    allowed_retirement_files = {'studies/IJCAS/scripts/build_bilingual_manuscript.py',
        'src/patient_initial_stability/test/test_bilingual_manuscript.py',
        'tools/latex_build.py', 'tools/test_portable_paper_paths.py'}
    if any(not path.startswith(allowed_retirement_prefixes) and path not in allowed_retirement_files
           for path in retired):
        raise ValueError('Only explicitly retired publication assets may be excluded; never experiment logs')
    excluded_publication_files = 0
    relocated_technical_files = 0
    relocated_technical_mismatches = []
    failures, broken, outside, syntax = [], [], [], []
    verified = 0
    evidence_files = 0
    evidence_bytes = 0
    start = time.monotonic()
    source_root = args.reference_study or OLD/'studies/IJCAS'
    original_available = source_root.is_dir()
    if original_available:
        checks = [(Path('studies/IJCAS')/source.relative_to(source_root), None)
                  for source in sorted(source_root.rglob('*'))
                  if not source.is_symlink() and source.is_file()
                  and '__pycache__' not in source.parts]
    else:
        recorded = [json.loads(line) for line in (META/'sha256_manifest.jsonl').read_text().splitlines()]
        checks = [(Path(row['path']), row['sha256_original']) for row in recorded]
        assert len(checks) >= 20000, 'Full recorded manifest is required after cleanup'
    temporary_manifest = META/'sha256_manifest.jsonl.tmp'
    with temporary_manifest.open('w') as report:
        for relative, expected in checks:
            source = source_root/relative.relative_to('studies/IJCAS')
            target = ROOT/relative
            preserved = META/'original_sources'/relative
            original = digest(source) if original_available else expected
            if relative.as_posix() in retired and relative.as_posix() not in moved:
                report.write(json.dumps(dict(path=str(relative), sha256_original=original,
                    intentionally_excluded_publication_asset=True))+'\n')
                excluded_publication_files += 1
                continue
            check_path = preserved if relative.as_posix() in changed else target
            if relative.as_posix() in archived_runtime:
                archived_path = runtime_archive/relative
                if (not archived_path.is_file() or digest(archived_path) !=
                        archived_runtime[relative.as_posix()]['sha256_at_archive']):
                    failures.append('Archived runtime copy: '+str(relative))
                if relative.as_posix() not in changed:
                    check_path = archived_path
                archived_source_files_verified += 1
            if relative.as_posix() in moved:
                destination = ROOT/moved[relative.as_posix()]
                copied_sha = digest(destination) if destination.is_file() else None
                if copied_sha != retired[relative.as_posix()]['sha256_at_cleanup']:
                    relocated_technical_mismatches.append(str(relative))
                if relative.as_posix() not in changed:
                    check_path = destination
                relocated_technical_files += 1
            actual = digest(check_path) if check_path.is_file() else None
            if original != actual:
                failures.append(str(relative))
            report.write(json.dumps(dict(path=str(relative), sha256_original=original,
                sha256_preserved=actual, comparison_file=str(check_path.relative_to(ROOT))
                    if check_path.is_relative_to(ROOT) else str(check_path)))+'\n')
            verified += 1
            if relative.parts[:3] == ('studies', 'IJCAS', 'results'):
                evidence_files += 1
                evidence_bytes += check_path.stat().st_size
            if verified % 5000 == 0:
                print(f'Checksummed {verified} IJCAS files; mismatches={len(failures)}', flush=True)
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {'build','install','log','migration','archive'}:
            continue
        if path.is_symlink():
            if not path.exists():broken.append(str(relative))
            destination = path.resolve()
            if destination != ROOT and ROOT not in destination.parents:
                outside.append([str(relative),str(destination)])
        elif path.is_file() and path.suffix == '.py' and 'results' not in relative.parts:
            if '__pycache__' in relative.parts or 'source_provenance' in relative.parts:
                continue
            try:
                ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as error:
                syntax.append([str(relative),str(error)])
    verification = dict(schema=2, workspace_root=str(ROOT),
        original_preserved_in_new_workspace=not bool(excluded_publication_files),
        original_experiment_logs_preserved=not bool(failures),
        publication_files_intentionally_excluded=excluded_publication_files,
        relocated_technical_files_verified=relocated_technical_files,
        archived_runtime_sources_verified=archived_source_files_verified,
        runtime_archive_inside_workspace=bool(runtime_archive and runtime_archive.is_relative_to(ROOT)),
        relocated_technical_copy_mismatches=relocated_technical_mismatches,
        original_study_folder_still_present=(OLD/'studies/IJCAS').is_dir(),
        comparison_source=str(source_root) if original_available else 'recorded SHA-256 manifest',
        ijcas_source_files_verified=verified, original_byte_mismatches=failures,
        raw_result_files_verified=evidence_files, raw_result_bytes_verified=evidence_bytes,
        broken_links=broken, links_into_original_or_other_workspaces=outside,
        python_syntax_errors=syntax, elapsed_wall_s=round(time.monotonic()-start,3),
        verification_scope='file integrity, Python syntax, symlinks; not a live navigation trial',
        remote_runtime_dependencies=['ROS 2 Humble in /opt/ros/humble',
            'Isaac Sim SDK in existing isaac_env', 'NVIDIA Hospital/Grid USD and textures'],
        experiments_run=0, simulator_started=False)
    print(json.dumps(verification,indent=2), flush=True)
    if failures or relocated_technical_mismatches or broken or outside or syntax:
        raise SystemExit(1)
    temporary_manifest.replace(META/'sha256_manifest.jsonl')
    temporary_report = META/'verification.json.tmp'
    temporary_report.write_text(json.dumps(verification,indent=2)+'\n')
    temporary_report.replace(META/'verification.json')


if __name__ == '__main__':
    main()
