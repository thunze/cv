{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (
        system: f {
          pkgs = import nixpkgs {
            inherit system;
          };
        }
      );
    in
    {
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {

          packages = with pkgs; [
            bashInteractive
            python313
            uv
          ];

          env = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
              stdenv.cc.cc.lib
              glib
              libGL
              zlib
            ]);
            UV_NO_MANAGED_PYTHON = "1";
            UV_PYTHON = pkgs.python313;
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            unset VIRTUAL_ENV
            ${pkgs.uv}/bin/uv venv
            source .venv/bin/activate
          '';
        };
      });

      formatter = forAllSystems ({ pkgs, ... }: pkgs.nixpkgs-fmt);
    };
}
