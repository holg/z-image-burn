{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  },
}:
let
  cudaPackages = pkgs.cudaPackages_13_0;
  nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.beta;
  libtorch = pkgs.libtorch-bin;
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

    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    nvidiaPackage

    libtorch
  ];

  LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath packages}";
  shellHook = ''
    export CUDA_PATH=${cudaPackages.cudatoolkit}
    export EXTRA_LDFLAGS="-L/lib -L${nvidiaPackage}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
  LIBTORCH_LIB = "${libtorch}";
  LIBTORCH_INCLUDE = "${libtorch.dev}";
}
