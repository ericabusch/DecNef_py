"""Optional Qt for the vendored RTPSpy modules.

RTPSpy's volreg/regress classes carry GUI code (progress dialogs, parameter
widgets, live motion plots). The real-time pipeline never shows them, so PyQt5
is an optional extra: when it is missing, these stand-ins let the modules
import, and any attempt to actually open a widget raises a clear error.
"""
from __future__ import annotations

try:  # pragma: no cover - depends on the environment
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False

    class _Missing:
        """Any attribute access or instantiation reports the missing extra."""

        def __init__(self, *a, **k):
            raise RuntimeError("PyQt5 is not installed: install the 'gui' extra to use RTPSpy's GUI features")

        def __getattr__(self, name):
            raise RuntimeError("PyQt5 is not installed: install the 'gui' extra to use RTPSpy's GUI features")

    class _QObject:  # base class for nested helper classes (never instantiated without Qt)
        def __init__(self, *a, **k):
            raise RuntimeError("PyQt5 is not installed: install the 'gui' extra to use RTPSpy's GUI features")

    def _signal(*a, **k):
        return None

    class _Namespace:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):
            return _Missing

    QtCore = _Namespace(QObject=_QObject, QThread=_QObject, pyqtSignal=_signal)
    QtWidgets = _Namespace(QDialog=_QObject, QMainWindow=_QObject)
    QtGui = _Namespace()

try:  # pragma: no cover
    import serial  # type: ignore
except ImportError:  # pragma: no cover
    serial = None

string_types = (str,)
