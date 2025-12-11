{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
    };
  },
}:
let
  rocmEnv = pkgs.symlinkJoin {
    name = "rocm-combined";
    paths = with pkgs.rocmPackages; [
      rocblas
      hipblas
      clr
    ];
  };
in
pkgs.mkShell rec {
  packages = with pkgs; [
    cargo
    rust-analyzer
    rustc
    rustfmt
    uv

    vulkan-headers
    vulkan-loader
    vulkan-tools
    vulkan-tools-lunarg
    vulkan-extension-layer
    vulkan-validation-layers

    stdenv.cc
    binutils
    ncurses

    clang
    stdenv.cc.cc.lib

    rocmEnv
  ];

  LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath packages}";
}
