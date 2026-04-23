{pkgs}: {
  deps = [
    pkgs.ghostscript
    pkgs.poppler
    pkgs.tesseract
    pkgs.xorg.libxcb
  ];
}
