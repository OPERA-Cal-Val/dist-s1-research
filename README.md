# DIST-S1 Research

Notebooks to provide trade studies and visualizations for developing the final OPERA `dist-s1` algorithm

## Install

```
mamba env update -f environment.yml
conda activate dist-s1
python -m ipykernel install --user --name dist-s1
```

### For bm3d

To use the well-known denoiser, please use Rosetta.

```
CONDA_SUBDIR=osx-64 conda create -n dist-s1-intel 
conda activate dist-s1-intel
python -c "import platform;print(platform.machine())"  # Confirm that the correct values are being used.
conda config --env --set subdir osx-64 
```

## Notes

Create your own directory with your last name and do as you please. Don't mess with other people's work. This is a poorly versioned controlled repository and meant to provide sample code and prototypes that can be distilled down later.