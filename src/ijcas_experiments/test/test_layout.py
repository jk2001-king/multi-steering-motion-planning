"""Current layout and historical-name compatibility, without simulator access."""
from pathlib import Path
import hashlib
import pytest
import ijcas_paths as paths


@pytest.mark.parametrize('legacy,current', [
    ('studies/IJCAS/scripts','src/ijcas_experiments/runtime/study'),
    ('isaac/scripts','src/ijcas_experiments/runtime/sdk'),
    ('isaac/scenes','src/ijcas_experiments/assets/isaac/scenes'),
    ('studies/IJCAS/config','src/ijcas_experiments/config/study'),
    ('studies/IJCAS/results','archive/original/studies/IJCAS/results'),
    ('migration','archive/original/migration'),
    ('results/experiments','log/ijcas/experiments'),
])
def test_old_symbolic_paths_use_current_layout(legacy,current):
    assert paths.layout_path(paths.ROOT,legacy)==paths.ROOT/current


def test_unrelated_fixture_root_is_not_remapped(tmp_path):
    assert paths.layout_path(tmp_path,'studies','IJCAS')==tmp_path/'studies/IJCAS'


def test_archived_run_and_new_output_are_separate():
    old=paths.ROOT/'results/experiments'
    assert paths.resolve_archived_path(old)==paths.ROOT/'archive/original/results/experiments'
    assert paths.layout_path(paths.ROOT,'results/experiments')==paths.ROOT/'log/ijcas/experiments'


def test_source_zip_names_accept_exactly_one_historical_or_current_name():
    current=paths.layout_path(paths.ROOT,'studies/IJCAS/scripts/hospital_mission_sequence.py')
    old='studies/IJCAS/scripts/hospital_mission_sequence.py'
    new=current.relative_to(paths.ROOT).as_posix()
    assert paths.archive_member_path(current,[old])==old
    assert paths.archive_member_path(current,[new])==new
    with pytest.raises(ValueError):paths.archive_member_path(current,[old,new])
    with pytest.raises(ValueError):paths.archive_member_path(current,[])


def test_original_source_view_requires_both_recorded_hashes(monkeypatch):
    file=paths.layout_path(paths.ROOT,'studies/IJCAS/scripts/hospital_mission_sequence.py')
    relative=str(file.relative_to(paths.ROOT));entry=paths.layout_manifest()[relative]
    original=paths.original_layout_source(file)
    assert hashlib.sha256(original).hexdigest()==entry['original_sha256']
    monkeypatch.setattr(paths,'layout_manifest',lambda:{relative:dict(entry,active_sha256='0'*64)})
    with pytest.raises(ValueError,match='Active source differs'):paths.original_layout_source(file)


def test_rebased_USD_view_is_hash_verified_and_never_used_to_replace_live_asset():
    file=paths.layout_path(paths.ROOT,'isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd')
    entry=paths.layout_manifest()[str(file.relative_to(paths.ROOT))]
    assert entry['verified_transform'].startswith('USD_EXTERNAL_REFERENCE_PATHS_ONLY')
    assert hashlib.sha256(file.read_bytes()).hexdigest()==entry['active_sha256']
    assert hashlib.sha256(paths.original_layout_source(file)).hexdigest()==entry['original_sha256']


def test_extracted_original_route_binds_layout_helpers_before_any_execution():
    import builtins,dis,types
    import hospital_task_runtime as runtime
    source=runtime.task.shared.ENTRY.read_text()
    module=compile(source,str(runtime.task.shared.ENTRY),'exec',dont_inherit=True)
    required=set()
    def collect(code):
        required.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
        for c in code.co_consts:
            if isinstance(c,types.CodeType):collect(c)
    for code in module.co_consts:
        if isinstance(code,types.CodeType) and code.co_name in runtime.ROUTE_FUNCTIONS:collect(code)
    scope={name:object() for name in required-set(vars(builtins))-{'_ijcas_path','_ijcas_workspace'}}
    bound=runtime.compile_route_functions(scope)
    assert bound['_ijcas_path'] is paths.layout_path
    assert bound['_ijcas_workspace'] is paths.workspace_root


def test_complete_route_function_AST_differs_only_by_explicit_path_joins():
    import ast,copy
    import hospital_task_runtime as runtime
    current=runtime.task.shared.ENTRY
    original=paths.ROOT/'archive/original'/paths.legacy_relative_path(current)
    class RestoreJoins(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_ijcas_path':
                result=copy.deepcopy(node.args[0])
                for arg in node.args[1:]:result=ast.BinOp(left=result,op=ast.Div(),right=self.visit(copy.deepcopy(arg)))
                return result
            return self.generic_visit(node)
    before={n.name:n for n in ast.parse(original.read_text()).body if hasattr(n,'name')}
    after={n.name:n for n in ast.parse(current.read_text()).body if hasattr(n,'name')}
    for name in runtime.ROUTE_FUNCTIONS:
        restored=RestoreJoins().visit(copy.deepcopy(after[name]))
        assert ast.dump(restored,include_attributes=False)==ast.dump(before[name],include_attributes=False),name
