{
  description = "TheEye";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # Disclaimer: Uv2nix is new and experimental.
  # Users are expected to be able to contribute fixes.
  #
  # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
  # It's using https://nix-community.github.io/pyproject.nix/build.html

  outputs =
    {
      uv2nix,
      pyproject-nix,
      nixpkgs,  
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = { 
          allowUnfree = true; 
          cudaSupport = true;
        };
      };
      # Load a uv workspace from a workspace root.
      # Uv2nix treats all uv projects as workspace projects.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Create package overlay from workspace.
      overlay = workspace.mkPyprojectOverlay {
        # Prefer prebuilt binary wheels as a package source.
        # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
        # Binary wheels are more likely to, but may still require overrides for library dependencies.
        sourcePreference = "wheel"; # or sourcePreference = "sdist";
        # Optionally customise PEP 508 environment
        # environ = {
        #   platform_release = "5.10.65";
        # };
      };

      # Extend generated overlay with build fixups
      #
      # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
      # This is an additional overlay implementing build fixups.
      # See:
      # - https://adisbladis.github.io/uv2nix/FAQ.html
      pyprojectOverrides = final: prev: {
        nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old:  {
          buildInputs = with pkgs; [cudatoolkit libGLU libGL] ++ (old.buildInputs or [ ]);
          });

        torch = prev.torch.overrideAttrs (old:  {
          buildInputs = with pkgs; [cudaPackages.cudatoolkit libGLU libGL cudaPackages.cudnn cudaPackages.nccl linuxPackages.nvidia_x11] ++ (old.buildInputs or [ ]);
        });


        tbbpool = prev.tbbpool.overrideAttrs (old:  {
          buildInputs = with pkgs; [tbb gcc ] ++ (old.buildInputs or [ ]);
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoPatchelfHook ];
          autoPatchelfExtraLibDirs = [ "${pkgs.tbb}/lib" ];
        });

        numba = prev.numba.overrideAttrs (old:  {
          buildInputs = with pkgs; [tbb gcc 
            llvmPackages_14.llvm
            llvmPackages_14.libclang
          ] ++ (old.buildInputs or [ ]);
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ 
            pkgs.autoPatchelfHook
            pkgs.llvmPackages_14.libclang
          ];
          autoPatchelfExtraLibs = [ "${pkgs.tbb}/lib" ];
          autoPatchelfIgnoreMissingDeps = [ "libtbb.so.12" ];
          postFixup = ''
            find ${pkgs.tbb}/lib -name "libtbb*" -exec file {} \;
            patchelf --set-rpath "${pkgs.lib.makeLibraryPath ([
              pkgs.tbb
            ] ++ (old.buildInputs or []))}" \
            $out/lib/python*/site-packages/numba/np/ufunc/tbbpool.cpython-*-linux-gnu.so
          '';
        });


        pyaudio = prev.pyaudio.overrideAttrs (old:  {
          buildInputs = with pkgs; [ portaudio ] ++ (old.buildInputs or [ ]);
        });


        numpy = prev.numpy.overrideAttrs (old:  {
          buildInputs = with pkgs; [ zlib ] ++ (old.buildInputs or [ ]);
          });

        nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old:  {
          buildInputs = with pkgs; [ cudaPackages.libcusparse ] ++ (old.buildInputs or [ ]);
          });
      };

      # Use Python 3.12 from nixpkgs
      python = pkgs.python312;

      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (pkgs.lib.composeExtensions overlay pyprojectOverrides);

    in
    {
      # Package a virtual environment as our main application.
      # packages.x86_64-linux.default = pythonSet.mkVirtualEnv "bulletsnake-env" {
      # };

      # This example provides two different modes of development:
      # - Impurely using uv to manage virtual environments
      # - Pure development using uv2nix to manage virtual environments
      devShells.x86_64-linux = {
        # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
        # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
        impure = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
          ];
          shellHook = ''
            unset PYTHONPATH
          '';
        };

        nixConfig = {
          extra-substituters = [
            "https://nix-community.cachix.org"
            "https://cache.nixos.org"
          ];
          extra-trusted-public-keys = [
            "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
          ];
        };

        # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
        # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
        #
        # This means that any changes done to your local files do not require a rebuild.
        default =
          let
            # Create an overlay enabling editable mode for all local dependencies.
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              # Use environment variable
              root = "$REPO_ROOT";
              # Optional: Only enable editable for these packages
              # members = [ "hello-world" ];
            };

            # Override previous set with our overrideable overlay.
            editablePythonSet = pythonSet.overrideScope editableOverlay;

            # Build virtual environment
            virtualenv = editablePythonSet.mkVirtualEnv "audionet" {
              audionet = [ ];
            };
            myBuildInputs = with pkgs;[
              uv git cudatoolkit
              linuxPackages.nvidia_x11
              ncurses5
              tbb
              libGLU libGL
              xorg.libXi xorg.libXmu freeglut
              xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
              cudaPackages.cudnn
              cudaPackages.nccl
              pyright
            ] ++ [ virtualenv ];
          in
          pkgs.mkShell {
            name = "AudioNet";
            buildInputs = myBuildInputs;
            shellHook = ''
              # Undo dependency propagation by nixpkgs.
              unset PYTHONPATH
              export CUDA_PATH=${pkgs.cudatoolkit}
              # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.lib.makeLibraryPath myBuildInputs}
              export CUDA_PATH=${pkgs.cudatoolkit}
              # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
              export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
              export EXTRA_CCFLAGS="-I/usr/include"
            '';
          };
      };
    };
}
