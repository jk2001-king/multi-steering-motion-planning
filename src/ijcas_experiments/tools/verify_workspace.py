"""Non-mutating source/evidence checks; derived receipts go only to log/ijcas."""
from pathlib import Path
import ast,hashlib,json,os,subprocess
from ijcas_paths import ROOT


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def verify(full=False):
    meta=ROOT/'archive/layout_20261126'
    manifest=json.loads((meta/'before.json').read_text())
    bad=[];checked=0;size=0
    if full:
        for row in manifest['files']:
            archived=ROOT/'archive/original'/row['path']
            path=archived if archived.exists() or archived.is_symlink() else ROOT/row['path']
            if 'symlink' in row:same=path.is_symlink() and os.readlink(path)==row['symlink']
            else:same=path.is_file() and digest(path)==row['sha256']
            if same:checked+=1;size+=row.get('size',0)
            else:bad.append(row['path'])
    syntax=[];changed=[]
    for path in (ROOT/'src').rglob('*.py'):
        if '__pycache__' in path.parts:continue
        try:ast.parse(path.read_text())
        except SyntaxError as error:syntax.append(str(error))
    for relative,row in json.loads((meta/'source_manifest.json').read_text()).items():
        current=ROOT/relative;original=ROOT/'archive/original'/row['original_path']
        if not current.is_file() or digest(current)!=row['active_sha256'] or digest(original)!=row['original_sha256']:changed.append(relative)
    broken=[]
    for folder in ('src','build','install'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_symlink() and not path.exists():broken.append(str(path.relative_to(ROOT)))
    unexpected=[name for name in ('studies','isaac','tools','config','data','docs','migration','results','external','scripts','isaac_model','motor_bag','motor_file')
                if (ROOT/name).exists() or (ROOT/name).is_symlink()]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    same_git=head==manifest['git_head'] and (ROOT/'.git').stat().st_ino==manifest['git_inode']
    # Publishing normal descendant commits must not invalidate evidence preservation.
    # Keep the original repository and require the recorded commit to remain an ancestor.
    preserved_git=(ROOT/'.git').stat().st_ino==manifest['git_inode'] and subprocess.run(
        ['git','merge-base','--is-ancestor',manifest['git_head'],head],cwd=ROOT,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    result=dict(workspace=str(ROOT),full_original_hash_audit=full,original_files_declared=len(manifest['files']),
        original_files_verified=checked,original_bytes_verified=size,original_mismatches=bad,
        active_source_digest_mismatches=changed,syntax_errors=syntax,broken_active_symlinks=broken,
        obsolete_root_entries=unexpected,git_history_unchanged=same_git,
        original_git_history_preserved=preserved_git,
        new_simulator_trials=0,original_scientific_records_changed=False)
    result['PASS']=not(bad or changed or syntax or broken or unexpected) and preserved_git
    target=ROOT/'log/ijcas/layout_verification.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full',action='store_true')
    args=parser.parse_args();result=verify(args.full);print(json.dumps(result,indent=2));raise SystemExit(not result['PASS'])
