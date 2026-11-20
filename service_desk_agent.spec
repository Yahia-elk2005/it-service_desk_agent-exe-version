# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

datas = [('Download the ICO file.ico', '.')]
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')
datas += collect_data_files('safehttpx')
datas += collect_data_files('groovy')
datas += collect_data_files('rfc3987_syntax')
datas += copy_metadata('gradio')
datas += copy_metadata('gradio_client')


a = Analysis(
    ['service_desk_agent.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pyarrow', 'matplotlib', 'scipy', 'torch'],
    noarchive=False,
    optimize=0,
    module_collection_mode={
        'gradio': 'py',
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='service_desk_agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['Download the ICO file.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='service_desk_agent',
)
