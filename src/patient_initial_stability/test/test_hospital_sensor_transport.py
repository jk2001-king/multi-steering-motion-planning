"""Pure lossless transport fixtures, not actual sensor/ROS/physics trials."""

import copy
import pytest
import hospital_sensor_transport as transport
from test_hospital_sensor_runtime import sensor_sequence_fixture


def frame_fixture():
    stream,_,_,_=sensor_sequence_fixture()
    return stream.capture()


def test_lossless_raw_source_roundtrip_keeps_entire_original_frame_and_every_hit():
    frame=frame_fixture();before=copy.deepcopy(frame)
    compact=transport.pack_frame(frame)
    assert 'decoded_rays' not in compact['scan_source']
    assert compact['scan_source']['raw_records']==frame['scan_packet']['raw_records']
    assert transport.unpack_frame(compact).frame==frame
    assert frame==before


@pytest.mark.parametrize('mutation',['range','stamp','role','omitted_ray','encoder_stamp','decoded_substitute','extra'])
def test_malformed_or_relabelled_raw_source_never_becomes_a_verified_frame(mutation):
    compact=transport.pack_frame(frame_fixture())
    if mutation=='range':compact['expected_scan_sha256']='0'*64
    if mutation=='stamp':compact['scan_source']['acquisition']['stamp_ns']+=1
    if mutation=='role':compact['scan_source']['acquisition']['source_role']='GROUND_TRUTH_IS_SENSOR'
    if mutation=='omitted_ray':compact['scan_source']['raw_records'].pop()
    if mutation=='encoder_stamp':compact['frame_fields']['native_encoder']['stamp_ns']+=1
    if mutation=='decoded_substitute':compact['frame_fields']['scan_packet']={}
    if mutation=='extra':compact['unreviewed']=True
    with pytest.raises((ValueError,KeyError)):transport.unpack_frame(compact)


def test_wire_json_cannot_supply_an_in_process_verification_receipt():
    with pytest.raises(ValueError):transport.VerifiedFrame(frame_fixture())


def test_raw_archive_checks_all_original_source_frames_after_lossless_readback(tmp_path):
    stream,boundary,encoder,_=sensor_sequence_fixture()
    archive=transport.RawSourceFrameArchive(tmp_path/'raw.jsonl.gz',maximum_frames=2,maximum_compressed_bytes=1024*1024)
    archive(stream.capture())
    boundary['timeline']['current_time_s']+=.2;encoder['stamp_ns']+=200000000
    archive(stream.capture());result=archive.finish()
    assert result['original_raw_frames']==2 and result['all_original_native_hits_metadata_and_frame_fields_retained']
    with pytest.raises(ValueError):archive.finish()
    with pytest.raises(FileExistsError):transport.RawSourceFrameArchive(tmp_path/'raw.jsonl.gz',maximum_frames=2,maximum_compressed_bytes=1)
