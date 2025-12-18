{
  inputs = {
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs =
    {
      fenix,
      nixpkgs,
      ...
    }:
    let
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];

      forAllSystems =
        f:
        builtins.listToAttrs (
          map (system: {
            name = system;
            value = f system;
          }) systems
        );
    in
    {
      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-tree);

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ fenix.overlays.default ];
          };
          commonPackages = with pkgs; [
            (
              with pkgs.fenix.stable;
              pkgs.fenix.combine [
                cargo
                clippy
                rust-src
                rustc
                rustfmt
              ]
            )

            llvmPackages.libstdcxxClang
            rust-analyzer
            stdenv.cc.cc.lib
            uv
            vulkan-loader
          ];
        in
        {
          default = pkgs.mkShell rec {
            packages = commonPackages;

            LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath packages}";
          };

          rocm = pkgs.mkShell rec {
            packages =
              with pkgs;
              commonPackages
              ++ [
                (pkgs.symlinkJoin {
                  name = "rocm-combined";
                  paths = with pkgs.rocmPackages; [
                    rocblas
                    hipblas
                    clr
                  ];
                })
              ];

            LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath packages}";
          };
        }
      );
    };
}
