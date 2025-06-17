import tarfile
import io
import os
import importlib.machinery
import pytest

pytest.importorskip("numpy")

loader = importlib.machinery.SourceFileLoader('metrics', os.path.join('bin', 'benchmark_metrics.py'))
metrics = loader.load_module()

def create_gen_tar(tmp_path):
    tar_path = tmp_path / 'gen.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        content = '123\n456\ndata_test\n_cell_length_a 1\n'
        data = content.encode('utf-8')
        info = tarfile.TarInfo('sample.cif')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return tar_path


def test_read_generated_cifs_strip(tmp_path):
    tar_path = create_gen_tar(tmp_path)
    data = metrics.read_generated_cifs(str(tar_path))
    assert list(data.keys()) == ['sample']
    assert data['sample'][0].startswith('data_test')