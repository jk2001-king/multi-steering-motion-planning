"""Lossless RAW-source transport/storage for native sensor observations.

All original hits, their native metadata, the acquisition pose, configuration,
encoders and frame fields are retained. Derived scan/decoded geometry is
reconstructed with the pinned decoder, not transmitted three times. This is
an I/O component: no SDK, ROS, timeline, actuation, pose reset or new trial.
"""

import hashlib
import json
from pathlib import Path

import hospital_runtime_evidence_io as evidence
import hospital_sensor_messages as messages


VERSION='LOSSLESS_NATIVE_RAW_SOURCE_FRAME_v1'
SOURCE_FIELDS=('version','kind','sequence_number','acquisition','configuration','raw_records')


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pack_frame(frame):
    """Public checked compaction; never accept fabricated decoded ranges."""
    messages.validate_packet(frame['scan_packet'])
    return _pack_immediately_verified_frame(frame)


def _pack_immediately_verified_frame(frame):
    """Only immediately after native SensorSequence.capture()/full validation.

    The private fast path changes no source values. It only avoids verifying
    the SAME in-process, unmodified packet at consecutive I/O sinks.
    """
    packet=frame['scan_packet']
    return evidence.json_value(dict(version=VERSION,
        scan_source={key:packet[key] for key in SOURCE_FIELDS},
        expected_scan_sha256=_sha(packet['scan']),
        frame_fields={key:value for key,value in frame.items() if key!='scan_packet'}))


class VerifiedFrame:
    """Private verification receipt; external wire JSON cannot supply one."""
    def __init__(self,frame,*,_token=None):
        if _token is not _RECEIPT_TOKEN:raise ValueError('Reconstruct and verify the actual raw wire frame first')
        self.frame=frame


_RECEIPT_TOKEN=object()


def unpack_frame(compact):
    """Validate/reconstruct ONCE at the receiving process boundary."""
    compact=evidence.json_value(compact)
    if (type(compact) is not dict or set(compact)!={'version','scan_source','expected_scan_sha256','frame_fields'}
            or compact['version']!=VERSION or set(compact['scan_source'])!=set(SOURCE_FIELDS)
            or type(compact['frame_fields']) is not dict or 'scan_packet' in compact['frame_fields']):
        raise ValueError('Exact lossless native raw-source frame required')
    source=compact['scan_source']
    if source['version']!=messages.VERSION or source['kind']!='INSTANT_SCAN':
        raise ValueError('Pinned instantaneous scan source format required')
    packet=messages.scan_packet(source['acquisition'],source['configuration'],source['raw_records'],source['sequence_number'])
    if _sha(packet['scan'])!=compact['expected_scan_sha256']:
        raise ValueError('Original transmitted LaserScan differs from reconstructed raw returns')
    frame=dict(compact['frame_fields'],scan_packet=packet)
    if frame['native_encoder']['stamp_ns']!=packet['acquisition']['stamp_ns']:
        raise ValueError('Exact original encoder/scan acquisition stamp required')
    return VerifiedFrame(frame,_token=_RECEIPT_TOKEN)


class RawSourceFrameArchive:
    """Exclusive bounded lossless stream; expensive readback is SDK-free."""
    def __init__(self,path,*,maximum_frames,maximum_compressed_bytes):
        import gzip
        if (type(maximum_frames) is not int or not 1<=maximum_frames<=3010
                or type(maximum_compressed_bytes) is not int or not 1<=maximum_compressed_bytes<=640*1024*1024):
            raise ValueError('Explicit bounded raw-source frame/byte budget required')
        self.path=Path(path).resolve();self.maximum_frames=maximum_frames
        self.maximum_bytes=maximum_compressed_bytes;self.raw=self.path.open('xb')
        self.stream=gzip.GzipFile(fileobj=self.raw,mode='wb',mtime=0,compresslevel=1)
        self.count=0;self.finished=False

    def __call__(self,frame):
        # Caller must immediately supply the native sequence's verified
        # in-process result. Receiver/readback independently reconstructs ALL
        # raw returns, near-obstacle rejection and exact native frame fields.
        if self.finished or self.count>=self.maximum_frames:raise ValueError('Closed/full raw-source archive')
        compact=_pack_immediately_verified_frame(frame)
        line=json.dumps(compact,separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(line)>evidence.MAX_BYTES:raise ValueError('Bounded complete raw-source line required')
        self.stream.write(line);self.count+=1

    def finish(self):
        import gzip,os
        if self.finished:raise ValueError('ONE exclusive raw-source finalization only')
        self.finished=True
        try:self.stream.close();self.raw.flush();os.fsync(self.raw.fileno())
        finally:self.raw.close()
        if self.path.stat().st_size>self.maximum_bytes:raise ValueError('Raw-source compressed byte budget exceeded')
        previous=None;count=0
        with gzip.open(self.path,'rb') as stream:
            while True:
                line=stream.readline(evidence.MAX_BYTES+1)
                if not line:break
                if len(line)>evidence.MAX_BYTES or count>=self.maximum_frames:raise ValueError('Raw-source readback exceeds bounds')
                frame=unpack_frame(json.loads(line)).frame;packet=frame['scan_packet']
                identity=(packet['acquisition']['actor_root'],packet['acquisition']['capture_id'])
                current=(identity,packet['sequence_number'],packet['acquisition']['stamp_ns'])
                if previous is not None and (identity!=previous[0] or current[1]!=previous[1]+1
                        or not 0<current[2]-previous[2]<=250000000):
                    raise ValueError('Raw-source acquisition identity/order/clock changed')
                previous=current;count+=1
        if count!=self.count:raise ValueError('Lossless raw-source readback count differs')
        return dict(path=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            format=VERSION,original_raw_frames=count,compressed_bytes=self.path.stat().st_size,
            all_original_native_hits_metadata_and_frame_fields_retained=True,
            exact_scan_encoder_stamps_and_raw_packets_verified=True,
            derived_geometry_reconstructed_NOT_a_dropped_or_synthetic_measurement=True)


class BoundedJSONLArchive:
    """Exclusive lossless small-record receipt stream, with byte-exact replay."""
    def __init__(self,path,*,maximum_records,maximum_record_bytes=65536):
        import gzip
        if (type(maximum_records) is not int or not 1<=maximum_records<=12010
                or type(maximum_record_bytes) is not int or not 1<=maximum_record_bytes<=65536):
            raise ValueError('Explicit finite receipt record/count limits required')
        self.path=Path(path).resolve();self.raw=self.path.open('xb')
        self.stream=gzip.GzipFile(fileobj=self.raw,mode='wb',mtime=0,compresslevel=1)
        self.maximum_records=maximum_records;self.maximum_record_bytes=maximum_record_bytes
        self.count=0;self.digest=hashlib.sha256();self.finished=False

    def __call__(self,value):
        if self.finished or self.count>=self.maximum_records:raise ValueError('Closed/full exclusive receipt stream')
        payload=json.dumps(evidence.json_value(value),separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(payload)>self.maximum_record_bytes:raise ValueError('Receipt exceeds exact record byte limit')
        self.stream.write(payload);self.digest.update(payload);self.count+=1

    def finish(self):
        import gzip,os
        if self.finished:raise ValueError('ONE receipt finalization only')
        self.finished=True
        try:self.stream.close();self.raw.flush();os.fsync(self.raw.fileno())
        finally:self.raw.close()
        if self.path.stat().st_size>256*1024*1024:raise ValueError('Compressed receipt bound exceeded')
        digest=hashlib.sha256();count=0
        with gzip.open(self.path,'rb') as stream:
            while True:
                line=stream.readline(self.maximum_record_bytes+1)
                if not line:break
                if len(line)>self.maximum_record_bytes or count>=self.maximum_records:raise ValueError('Receipt readback bounds exceeded')
                evidence.json_value(json.loads(line));digest.update(line);count+=1
        if count!=self.count or digest.hexdigest()!=self.digest.hexdigest():raise ValueError('Receipt bytes/count differ on independent readback')
        return dict(path=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            records=count,uncompressed_stream_sha256=digest.hexdigest(),exact_original_records_durably_verified=True)
