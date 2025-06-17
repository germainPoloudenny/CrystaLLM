import tarfile
import io
import os
import importlib.machinery

import pytest

# load the extract script as a module
loader = importlib.machinery.SourceFileLoader('extract', os.path.join('bin', 'extract_cifs_from_amp.py'))
extract = loader.load_module()


def create_tar_with_prefix(tmp_path):
    tar_path = tmp_path / 'in.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        content = '123\n456\ndata_test\n_cell_length_a 1\n'
        data = content.encode('utf-8')
        info = tarfile.TarInfo('sample.cif')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return tar_path


def test_load_entries_from_tar(tmp_path):
    tar_path = create_tar_with_prefix(tmp_path)
    entries = extract.load_entries_from_tar(str(tar_path))
    assert entries == [('sample', '123\n456\ndata_test\n_cell_length_a 1\n')]


def test_strip_prefix(tmp_path):
    tar_path = create_tar_with_prefix(tmp_path)
    out_path = tmp_path / 'out.tar.gz'
    extract.main([str(tar_path), '--out', str(out_path)])
    with tarfile.open(out_path, 'r:gz') as tar:
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        text = f.read().decode('utf-8')
    assert text.startswith('data_test')