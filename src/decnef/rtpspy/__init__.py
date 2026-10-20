"""Vendored RTPSpy modules (Misaki et al.; https://github.com/mamisaki/RTPSpy):
RtpVolreg (AFNI 3dvolreg via librtp), RtpRegress (online nuisance regression),
RetroTS physio regressors. Unchanged numerics; only the imports were made
optional (PyQt5 / pyserial, see _qt.py) and `six` removed. librtp.so/.dylib are
compiled from afni_module/ (a build step replaces the committed binaries later).
"""
