# PyInstaller 打包配置（Linux / Windows 通用）：pyinstaller optsim.spec
a = Analysis(['optsim_app.py'], pathex=['.'],
             datas=[('optsim_gui/glasses.json', 'optsim_gui')],
             hiddenimports=['matplotlib.backends.backend_qtagg', 'scipy.optimize', 'scipy.special._cdflib'],
             excludes=['tkinter', 'PyQt5', 'PyQt6', 'IPython', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
                       'PySide6.Qt3DCore', 'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtMultimedia', 'PySide6.QtPdf'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='optsim', console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name='optsim', upx=False)
